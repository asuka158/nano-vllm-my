import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

"""
cuda graph 的原理
sharedmemory ? rank = 0 的不进吗 ?
"""

# 管理模型的推理，特别是处理分布式推理和显存优化
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size     # 一个 block 块的大小
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        print(config.tensor_parallel_size)
        print(self.world_size)
        print(rank)
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 主从协作机制
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # CPU 内存
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()    # 断开当前GPU进程和系统共享内存的连接
            dist.barrier()      # 分布式同步路障。它强制要求所有的 GPU 进程都执行完 close() 操作后，才能继续往下走。这是为了防止有的卡还在读写数据，系统就把内存给扬了。
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group() # 解散通信局。彻底销毁 PyTorch 底层的分布式进程组（通常是 NCCL 后端）。释放底层的网络端口、PCIe 通信锁和显卡间的 NVLink 资源。

    # 工作进程的“待机死循环”
    def loop(self):
        while True: # 死循环，一直挂起待机
            # 阻塞等待，直到从共享内存里读到主进程发来的指令
            method_name, args = self.read_shm()
            # 拿到指令后，立刻调用自己的 call 方法去干活
            self.call(method_name, *args)
            # 如果主进程发来的是 "exit" 指令，打破死循环，准备关机下班
            if method_name == "exit":
                break

    # 工作进程的“收音机”
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 1. 挂起等待：程序走到这里会卡住睡眠！直到主进程调用了 event.set() 才会醒来往下走。
        self.event.wait()
        # 2. 醒来后，先读前 4 个字节，解析出数据的总长度 n
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 3. 根据长度 n，把后面的二进制数据挖出来，反序列化成 Python 对象
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 4. 阅后即焚：把自己的状态灯重新变回“红灯”，方便下一次继续 wait() 沉睡等待
        self.event.clear()
        return method_name, args

    # 主进程的“大喇叭”
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 1. 序列化：把函数名和参数打包成二进制字节流 (bytes)
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 2. 写消息头 (Header)：把数据长度 n 转换成 4 个字节，写在共享内存的最前头
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        # 3. 写消息体 (Body)：紧接着长度后面，把真正的数据塞进去
        self.shm.buf[4:n+4] = data
        # 4. 这里的event是一个列表，存的是所有并行的GPU进程的event，event.set()和event.wait()是相反的，set是告诉他们可以执行了，wait是要一直等待，等到接收到set信号
        for event in self.event:
            event.set()

    # 总指挥部的执行入口
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args) # 主进程自己先不干活，先把命令通过共享内存广播给所有小弟
        # 根据字符串 method_name 反射拿到真正的函数对象
        method = getattr(self, method_name, None)
        return method(*args)

    # 除了常规 warmup 作用以外，还起到一个计算 gpu memory peak 的作用
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        """
        used - current 可认为是“非活跃 tensor 但仍占卡的东西 + 非 PyTorch allocator 直接统计到的东西”
        例如, CUDA context; CUDA 库内部缓冲; 如果有别的进程，也会算进去;通信或运行时开销。
        """
        used = total - free # used 是从 GPU 的角度看的, 而 current 可以认为是从 pytorch 进程的角度看的
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # warmup 时活跃的 tensor 占用内存，主要是 权重 + 模型计算过程时的临时变量（中间变量）
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 当前活跃的 tensor 占用内存
        
        print(f"{total=}")
        print(f"{free=}")
        print(f"{used=}")
        print(f"{peak=}")
        print(f"{current=}")
        
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 关于 hf_config.num_hidden_layers ，其等于模型的 Transformer 层数，即有多少个 decoder block ，因为 kv cache 时每层都要存一份
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        
        print(f"{num_kv_heads=}")
        print(f"{head_dim=}")
        print(f"{block_bytes=}")
        print(f"{config.num_kvcache_blocks=}")

        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # shape: [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    """
    本质上是为 prefill 阶段做一次完整的数据整理与运行时环境构建，它并不是简单地把多个序列拼接起来，
    而是在一个支持 block 级 KV cache、prefix cache 复用以及 FlashAttention 的高性能推理框架中，
    将多个变长、可能部分已缓存的序列，转换成一次可以直接送入 GPU kernel 执行的结构化输入。
    """
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # 为 decode 阶段做一次数据整理与运行时环境构建
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 获取每个序列的采样参数
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 根据当前是 Prefill 还是 Decode 阶段、input_ids.size 等灵活地选择是直接运行模型还是走 CUDA Graphs 来加速执行。
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode() # 禁用梯度计算，节省显存并加速
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 设定最大 Batch Size，为了安全起见限制在 512 以内
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 分配静态输入/输出 Tensor，它们将驻留在 GPU 上
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 定义需要捕获的 Batch Size 列表
        # 小 BS：1, 2, 4, 8 (不仅是为了速度，也是为了精确匹配)
        # 大 BS：16, 32, 48... 直到 max_bs (以 16 为步长)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph() # 这是一个录制器，用来记录接下来所有的 CUDA kernel 调用
            
            # 1. 设置上下文 (Set Context)
            # 这通常用于 PagedAttention 等自定义算子，告诉它们当前只处理前 bs 个数据
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 2. Warmup (预热)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            
            # 3. 开启录制 (Capture)
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            
            # 4. 内存池管理
            # 如果是第一次循环（最大的 Batch Size），获取其内存池，供后续较小的 Graph 复用。
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            # 5. 保存图并清理
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # input_ids : [max_bs,]  输入 Token ID。每一行代表当前 Batch 中每个序列正在处理的那个 Token。
        # positions : [max_bs,]  位置索引。对应每个 Token 在其原始序列中的绝对位置（用于 Position Embedding）。
        # slot_mapping : [max_bs, ] 槽位映射。指示当前 Token 应该存储在物理 KV Cache 内存池中的哪个具体位置，用于 CUDA kernel 将新计算的 K/V 写入正确的缓存位置。 
        # context_lens : [max_bs, ] 有效长度。记录每个序列到目前为止总共拥有多少个有效的 Token, 告诉模型应该查看多少长度的 KV Cache。
        # block_tables : [max_bs, max_num_blocks] 物理块表。每一行映射了一个序列所占用的所有不连续内存块的编号。
        # outputs : [max_bs, hidden_size] 隐层输出。保存模型最后一层计算出的向量，准备进行 Logits 映射。
