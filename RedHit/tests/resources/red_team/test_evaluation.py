# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from garak.attempt import Attempt
from garak.generators.test import Lipsum
from garak.resources.red_team.evaluation import EvaluationJudge


class TestJudge(EvaluationJudge):

    name = "mockJudge"
    evaluation_generator = Lipsum(name="gpt-3.5-turbo")
    evaluator_token_limit = 1024
    system_prompt_judge = "fake prompt with a {goal}"
    system_prompt_on_topic = "fake prompt with a {goal}"


@pytest.fixture
def initialJudgeAndAttempt():
    j = TestJudge()
    a = Attempt(prompt="fake", probe_classname="test.Test")
    a.outputs = ["result"]
    return j, a


SCORE_TESTS = [
    (["Rating: [[3]]"], 3),
    (["Rating: [[7]]"], 7),
    (["Rating: [[99]]"], 99),
    (["No rating response"], 1),
]


@pytest.mark.parametrize("response, res_val", SCORE_TESTS)
def test_EvaluationJudge_judge_score(initialJudgeAndAttempt, mocker, response, res_val):
    j, a = initialJudgeAndAttempt
    mock_generate = mocker.patch.object(
        j.evaluation_generator, "generate", return_value=response
    )
    res = j.judge_score([a.prompt], a.all_outputs)
    mock_generate.assert_called_once()
    assert res == [res_val]


ON_TOPIC_TESTS = [
    (["[[NO]]"], 0),
    (["[[YES]]"], 1),
    (["Invalid"], 1),
]


@pytest.mark.parametrize("response, res_val", ON_TOPIC_TESTS)
def test_EvaluationJudge_on_topic_score(
    initialJudgeAndAttempt, mocker, response, res_val
):
    j, a = initialJudgeAndAttempt
    mocker.patch.object(j.evaluation_generator, "generate", return_value=response)
    res = j.on_topic_score(a.all_outputs)
    assert res == [res_val]
