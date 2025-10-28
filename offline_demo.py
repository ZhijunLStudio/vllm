from vllm import LLM, SamplingParams

# model_name_or_path = "/home/aistudio/config_folder"
model_name_or_path = "/home/aistudio/data/models/localdisk/264907/models/Kimi-K2-Instruct-0905"

# 超参设置
sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
llm = LLM(model=model_name_or_path, tensor_parallel_size=4, trust_remote_code=True, enforce_eager=True)
output = llm.generate(prompts="who are you？", sampling_params=sampling_params)

print(output)


