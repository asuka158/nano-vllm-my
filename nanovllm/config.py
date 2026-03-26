import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # 单 batch 处理的最大 token 数量
    max_num_batched_tokens: int = 16384
    # 单 batch 的最大 sequence 数量
    max_num_seqs: int = 512
    # 模型支持的最大输入长度, 如果一个请求的 prompt 对应的 token 长度大于了这个数值，就要被拆成两个 seq
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    # False 表示可以启用 cuda graph, True 的话一定不启用
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    # 一个 kv cache 块存256个 token
    kvcache_block_size: int = 256
    # kv cache 中块的数量 
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
