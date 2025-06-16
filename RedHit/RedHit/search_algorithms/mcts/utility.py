
import re
from RedHit.__main__ import server as vllm_server
from RedHit.utility import run_garak_ollama, stop_garak_ollama


def stop_llms():
    vllm_server.stop_vllm_server()
    stop_garak_ollama()


def serve_llms():
    vllm_server.start_vllm_server()


def memory_safe_fine_tuning(dpo_trainer, trainset):
    stop_llms()
    dpo_trainer(trainset)
    serve_llms()


def inject_histories(cot_prompt: str, histories: str) -> str: return re.sub(
    r"<Histories>.*?</Histories>", f"<Histories>{histories}</Histories>", cot_prompt, flags=re.DOTALL)
