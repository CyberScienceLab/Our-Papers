"""Replicate generator interface

Generator for https://replicate.com/

Put your replicate key in an environment variable called
REPLICATE_API_TOKEN. It's found on your Replicate account
page, https://replicate.com/account.

Text-output models are supported.
"""

import importlib
import os
from typing import List, Union

import backoff
import replicate.exceptions

from garak import _config
from garak.generators.base import Generator


class ReplicateGenerator(Generator):
    """Interface for public endpoints of models hosted in Replicate (replicate.com).

    Expects API key in REPLICATE_API_TOKEN environment variable.
    """

    ENV_VAR = "REPLICATE_API_TOKEN"
    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "temperature": 1,
        "top_p": 1.0,
        "repetition_penalty": 1,
    }

    generator_family_name = "Replicate"
    supports_multiple_generations = False

    def __init__(self, name="", config_root=_config):
        self.seed = 9
        if hasattr(_config.run, "seed") and _config.run.seed is not None:
            self.seed = _config.run.seed

        super().__init__(name, config_root=config_root)

        if self.api_key is not None:
            # ensure the token is in the expected runtime env var
            os.environ[self.ENV_VAR] = self.api_key
        self.replicate = importlib.import_module("replicate")

    @backoff.on_exception(
        backoff.fibo, replicate.exceptions.ReplicateError, max_value=70
    )
    def _call_model(
        self, prompt: str, generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        response_iterator = self.replicate.run(
            self.name,
            input={
                "prompt": prompt,
                "max_length": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "seed": self.seed,
            },
        )
        return ["".join(response_iterator)]


class InferenceEndpoint(ReplicateGenerator):
    """Interface for private Replicate endpoints.

    Expects `name` in the format of `username/deployed-model-name`.
    """

    @backoff.on_exception(
        backoff.fibo, replicate.exceptions.ReplicateError, max_value=70
    )
    def _call_model(
        self, prompt, generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        deployment = self.replicate.deployments.get(self.name)
        prediction = deployment.predictions.create(
            input={
                "prompt": prompt,
                "max_length": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
            },
        )
        prediction.wait()
        try:
            response = "".join(prediction.output)
        except TypeError as exc:
            raise IOError(
                "Replicate endpoint didn't generate a response. Make sure the endpoint is active."
            ) from exc
        return [response]


DEFAULT_CLASS = "ReplicateGenerator"
