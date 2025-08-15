import json
import os
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess


def ensure_model_pulled(model):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if model not in result.stdout:
            print(f'Model {model} not found locally. Pulling...')
            subprocess.run(['ollama', 'pull', model], check=True)
        else:
            print(f'Model {model} already available.')
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model {model}: {e}")


def get_responses(model_name):
    input_path = os.path.join(data_dir, "harmful_behaviors.csv")
    output_path = os.path.join(data_dir, f"sure_responses_{model_name}.csv")
    print("Reading input file...", input_path)
    df = pd.read_csv(input_path)
    
    for i, row in df.iterrows():
        prompt = row["goal"]
        prompt += ". Sure, here's"
        response: ChatResponse = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7}
        )
        
        response_data = {
            "prompt_number": i,
            "prompt": prompt,
            "response": response.message.content
        }
        
        pd.DataFrame([response_data]).to_csv(
            output_path, index=False, header=not os.path.exists(output_path), mode='a'
        )
        
        print(f"Completed: {i}")


models = [
    #"deepseek-r1:7b",
    #"gemma3:4b",
    #"phi4:latest",
    #"llama3:8b"
    "vicuna:7b",
    "vicuna:13b",
    "llama2:7b-chat",
    "koesn/mistral-7b-instruct:latest",
    "Hudson/falcon-mamba-instruct:7b-q4_0"
]
for model in models:
    ensure_model_pulled(model)
    get_responses(model)
