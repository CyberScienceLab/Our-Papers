import numpy as np
import random
import torch
import logging


def disable_logs():
    loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)


def stop_garak_ollama():
    import subprocess
    import yaml

    yaml_file_path = "/home/researchuser/LLMSec/LLMSecurity/garak/resources/garak.core.yaml"

    try:
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
            model_name = data["plugins"]["model_name"]

            subprocess.run(["ollama", "stop", model_name], check=True)
            print(f"Ollama {model_name} stopped successfully.")

    except Exception as ex:
        print(f"Error in stopping ollama: {ex}")


def run_garak_ollama():
    import subprocess
    import yaml

    yaml_file_path = "/home/researchuser/LLMSec/LLMSecurity/garak/resources/garak.core.yaml"

    try:
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
            model_name = data["plugins"]["model_name"]

            subprocess.run(["ollama", "run", model_name], check=True)
            print(f"Ollama {model_name} started successfully.")

    except Exception as ex:
        print(f"Error in stopping ollama: {ex}")


def seed_torch(seed=1373):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
