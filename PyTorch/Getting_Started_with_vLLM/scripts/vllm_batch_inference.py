from vllm import LLM, SamplingParams
import torch

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# initialize
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=30)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enforce_eager=True)

# perform the inference
outputs = llm.generate(prompts, sampling_params)

# print outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
