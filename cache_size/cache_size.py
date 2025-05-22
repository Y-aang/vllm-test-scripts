from vllm import LLM, SamplingParams
import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()

model_name = "mistralai/mistral-7b-v0.1"

llm = LLM(model=model_name, 
          gpu_memory_utilization=0.95,      # 0.657: 134 CUDA blocks
          max_model_len=4096, 
          block_size=32, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()

prompt = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."

print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")