
from transformers import AutoModelForCausalLM, AutoTokenizer
from dspy import dspy
from datetime import datetime
import torch

# model_id = 'TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF'
# filename = 'capybarahermes-2.5-mistral-7b.Q4_K_M.gguf'

# tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
# model = AutoModelForCausalLM.from_pretrained('')

# model.save_pretrained(
#     '/home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/ollama_models_v2/mistra7b')
# tokenizer.save_pretrained(
#     '/home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/ollama_models_v2/mistra7b')

from lm_loader import LMLoader

lmm = LMLoader(local_model_dir='/home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/RedHit',
               model_name='fine-tuned', initialize=True)

# Environment= "OLLAMA_MODELS=/home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/ollama_models"
# model = dspy.LM('huggingface//home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/Llama-3-8B-Instruct-abliterated-v2/', api_base='http://0.0.0.0:4000',  api_key='')

# resp = model('tell me a joke')

# pip install vllm
# python -m vllm.entrypoints.openai.api_server --model /home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/RedHit

torch.cuda.empty_cache()

llm = dspy.LM("openai/RedHit/pre_trained_models/local_models/RedHit2",
              api_base="http://0.0.0.0:8000/v1",
              api_key="local", model_type='chat',
              cache=False,
              cache_in_memory=False)

start = datetime.now()
resp = llm('tell me a joke about Iran')
end = datetime.now()


print(resp)
print(f"Started at: {start}, Ended at: {end}")
print(f"Execution Time: {end - start}")

x = 0
# # print(response)
# from vllm import LLM, SamplingParams

# # Renamed local model directory
# local_model_path = "/home/researchuser/LLMSec/LLMSecurity/RedHit/pre_trained_models/local_models/RedHit"

# # Create an LLM object
# llm = LLM(model=local_model_path,)

# # Define sampling parameters
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

# # Generate text
# prompts = ["What is the capital of France?",
#            "Write a short story about a robot."]
# outputs = llm.generate(prompts, sampling_params)

# # Print the outputs
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt}")
#     print(f"Generated text: {generated_text}")
#     print("-" * 50)
