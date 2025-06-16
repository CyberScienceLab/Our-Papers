# from dspy import LM
# from huggingface_hub import Repository, create_repo


# class CustomLMClient(LM):
#     def __init__(self, model_endpoint):
#         repo = Repository(local_dir="")
#         repo.client.
#         self.provider = "default"
#         self.client = InferenceClient(model=model_endpoint)

#         self.history = []

#     def basic_request(self, prompt, **kwargs):
#         output_text = self.client.text_generation(prompt=prompt, **kwargs)
#         self.history.append({
#             "prompt": prompt,
#             "response": output_text,
#             "kwargs": kwargs
#         })
#         return output_text

#     def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
#         response = self.basic_request(prompt, **kwargs)
#         completions = [response]
#         return completions
