# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import garak.probes.packagehallucination


def test_promptcount():
    languages = {
        "Python": garak.probes.packagehallucination.Python(),
        "Ruby": garak.probes.packagehallucination.Ruby(),
        "JavaScript": garak.probes.packagehallucination.JavaScript(),
        "Rust": garak.probes.packagehallucination.Rust()
    }

    expected_count = len(garak.probes.packagehallucination.stub_prompts) * len(
        garak.probes.packagehallucination.code_tasks
    )

    for language in languages:
        language_probe = languages[language]
        assert (
            len(language_probe.prompts) == expected_count
        ), f"{language} prompt count mismatch. Expected {expected_count}, got {len(language_probe.prompts)}"
