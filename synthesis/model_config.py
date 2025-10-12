

MODEL_CONFIGS = {
    "Qwen2.5_wikiQA": {
        "avg_prompt_len": 11170.23,
        "CP_ratio": None,
        "block_size": 16,
        "model_weight": 2.89,
        "non_torch_memory": 0.04,
        "torch_activation": 1.55,
        "kv_per_16_tokens": 4.27e-4,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_model_len": 25000,
    },
    "Qwen2.5_Squad": {
        "avg_prompt_len": 180.78,
        "CP_ratio": None,
        "block_size": 16,
        "model_weight": 2.89,
        "non_torch_memory": 0.04,
        "torch_activation": 1.39,
        "kv_per_16_tokens": 4.27e-4,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_model_len": 1024,
    },
    "DeepseekR1_QuALITY": {
        "avg_prompt_len": 5768.10,
        "CP_ratio": None,
        "block_size": 16,
        "model_weight": 3.3460,
        "non_torch_memory": 0.04,
        "torch_activation": 1.39,
        "kv_per_16_tokens": 4.27e-4,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_model_len": 25000,
    },
    "DeepseekR1_Wild": {
        "avg_prompt_len": 5768.10,
        "CP_ratio": None,
        "block_size": 16,
        "model_weight": 3.35,
        "non_torch_memory": 0.04,
        "torch_activation": 1.55,
        "kv_per_16_tokens": 4.27e-4,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "max_model_len": 25000,
    },

    # Add others like
    # "Mistral7B_SQuAD": {
    #     "avg_prompt_len": 5120,
    #     ...
    # },
}