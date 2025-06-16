import requests
from transformers import AutoTokenizer

from ollama import Client


class OllamaModelWrapper:
    def __init__(self, model_name, tokenizer_name, ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.client = Client(host=ollama_url)

    def generate(self, prompt):

        response = self.client.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                f'content': prompt,
            },
        ])

        return response

    # add max_new_tokens to prevent very long responses.
    def __call__(self, prompt, max_new_tokens=512):

        full_response = self.generate(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.tokenizer(full_response, return_tensors="pt")

        decoded_response = self.tokenizer.decode(
            outputs['input_ids'][0][:max_new_tokens], skip_special_tokens=True)
        return decoded_response
