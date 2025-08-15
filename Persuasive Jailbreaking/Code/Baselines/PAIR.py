import json
import os
import re
import pandas as pd
from ollama import chat
from ollama import ChatResponse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
data_dir = "my_PAIR"
import subprocess
from strong_reject.jailbreaks import apply_jailbreaks
from tqdm import tqdm 
'''
dict_keys(['auto_payload_splitting', 'auto_obfuscation', 'rot_13', 'disemvowel', 'pair', 'pap_evidence_based_persuasion', 'pap_expert_endorsement', 'pap_misrepresentation', 'pap_authority_endorsement', 'pap_logical_appeal', 'translation_hmong', 'translation_scotts_gaelic', 'translation_guarani', 'translation_zulu', 'gcg_transfer_harmbench', 'gcg_transfer_universal_attacks', 'combination_3', 'combination_2', 'few_shot_json', 'dev_mode_v2', 'dev_mode_with_rant', 'wikipedia_with_title', 'distractors', 'wikipedia', 'style_injection_json', 'style_injection_short', 'refusal_suppression', 'prefix_injection', 'distractors_negated', 'poems', 'base64', 'base64_raw', 'base64_input_only', 'base64_output_only', 'none', 'evil_confidant', 'aim'])
'''

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
    safe_model_name = model_name.replace("/", "_")
    safe_model_name = safe_model_name.replace(":", "_")
    output_path = os.path.join(data_dir, f"responses_{safe_model_name}.csv")
    print("Reading input file...", input_path)
    df = pd.read_csv(input_path)
    processed = set()
    if os.path.exists(output_path):
        try:
            processed_df = pd.read_csv(output_path)
            if "prompt_number" in processed_df.columns:
                processed = set(processed_df["prompt_number"].tolist())
        except Exception as e:
            print(f"Could not read checkpoint file: {e}")

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Model: {safe_model_name}"):
        if i in processed:
            continue  # Skip already processed

        original_prompt = row["goal"]
        prompt = apply_jailbreaks(original_prompt, ["pair"])[0]
        #print(prompt)
        #exit()
        response: ChatResponse = chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7}
        )

        response_data = {
            "prompt_number": i,
            "original_prompt": original_prompt,
            "jailbroken_prompt": prompt,
            "response": response.message.content
        }

        pd.DataFrame([response_data]).to_csv(
            output_path, index=False, header=not os.path.exists(output_path), mode='a'
        )

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
