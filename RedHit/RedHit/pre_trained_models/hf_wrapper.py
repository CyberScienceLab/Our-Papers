import dspy

from pre_trained_models.lm_loader import LMLoader


class BasicHFWrapper():
    def __init__(self, model_name, max_new_tokens=7000, temperature=1, max_tokens=2048, **kwargs):

        llm_loader = LMLoader(model_name, initialize=True)
        self.model = llm_loader.model
        self.tokenizer = llm_loader.tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        tokenized = self.tokenizer.encode(prompt,  return_tensors="pt")
        outputs = self.model.generate(tokenized, max_new_tokens=7000)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class HuggingFaceDSPy(dspy.LM):
    def __init__(self, model_name, model, tokenizer, max_new_tokens=7000, temperature=1, max_tokens=2048, **kwargs):
        super().__init__(model=model_name)
        self.model = model_name
        self._model = model
        self._tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.kwargs = dict(temperature=temperature,
                           max_completion_tokens=max_tokens, **kwargs)

    def __call__(self, **kwargs):
        messages = kwargs['messages']
        tokenized_chat = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = self._model.generate(tokenized_chat, max_new_tokens=7000)
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
