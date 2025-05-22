from vllm.config import CacheConfig, ModelConfig, ParallelConfig, DeviceConfig
from vllm.worker.cache_engine import CacheEngine

cache_config = CacheConfig(
    block_size=32,
    gpu_memory_utilization=0.75,
    swap_space=4,
    cache_dtype="auto"
)
model_config = ModelConfig(model="mistralai/mistral-7b-v0.1", max_model_len=4096)
parallel_config = ParallelConfig()
block_size_bytes = CacheEngine.get_cache_block_size(cache_config, model_config, parallel_config)
print(block_size_bytes)