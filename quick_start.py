from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is (please generate more than 128 words)",
    "The president of the United States is (please generate more than 128 words)",
    "The capital of France is (please generate more than 128 words)",
    "The future of AI is (please generate more than 128 words)",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

llm = LLM(model="facebook/opt-125m", 
          max_model_len=512,
          block_size=8,
          disable_sliding_window=True
          )
tokenizer = llm.get_tokenizer()

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f"Generated token length: {len(tokenizer(generated_text)["input_ids"])} characters")