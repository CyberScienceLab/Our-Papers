class Example:
    def __init__(self, prompt: str, response: str, reasoning: str):
        self.prompt = prompt
        self.response = response
        self.reasoning = reasoning


class DPOExample:
    def __init__(self, prompt, chosen, rejected):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected
