"""Always detectors

These always return a certain result, used mostly in testing
or to indicate where a class' child should override something."""

from typing import List

from garak.attempt import Attempt
from garak.detectors.base import Detector


class Fail(Detector):
    """Always Fail (for testing & prototyping)"""

    bcp47 = "*"

    def detect(self, attempt: Attempt) -> List[float]:
        return [1.0] * len(attempt.all_outputs)


class Pass(Detector):
    """Always Pass (for testing & prototyping)"""

    bcp47 = "*"

    def detect(self, attempt: Attempt) -> List[float]:
        return [0.0] * len(attempt.all_outputs)


class Passthru(Detector):
    """Send existing attempt detection results; use first detector"""

    bcp47 = "*"

    DEFAULT_PARAMS = Detector.DEFAULT_PARAMS | {
        "skip": True,
    }

    def detect(self, attempt: Attempt) -> List[float]:
        assert (
            len(attempt.detector_results) > 0
        ), "Only use passthru on attempts that already have detector scores"
        detector_chosen = sorted(list(attempt.detector_results.keys()))[0]
        return attempt.detector_results[detector_chosen]
