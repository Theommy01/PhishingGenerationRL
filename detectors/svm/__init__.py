"""The held-out detector: TF-IDF + linear SVM, trained on public corpora.

Never part of the loop. Nothing is trained against it and it contributes no
reward, so it answers the question ScamLLM cannot answer about itself: do the
messages that evade the detector in the loop also evade one that was not?

The pipeline is the one `metrics.analysis` builds and fits — `TfidfVectorizer`
into `SVC(kernel="linear")` — loaded from `config.SVM_MODEL_PATH`, overridable
with the `PHISHNET_SVM_MODEL` environment variable.

Scores here are **not** probabilities. The SVC is fitted with
`probability=False`, so there is no `predict_proba`; `decision_function` gives a
signed distance from the hyperplane, which `score_messages` squashes through a
logistic to land in 0-1. That is monotone in the margin and nothing more —
useful for ranking and for comparing rounds of this detector against each other,
not for reading as a likelihood, and not for comparing against ScamLLM's score.
`label_messages` is the honest cross-detector signal.
"""

import math
import os
from typing import List, Optional

from metrics import config

# Matches ScamLabeller's convention: the positive class is phishing, so a
# message is "safe" (the filter was evaded) when the detector says 0.
PHISHING_CLASS = 1

_SHARED_DETECTOR = None


class SvmDetector:
    """A fitted sklearn pipeline behind the two-call detector interface."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.SVM_MODEL_PATH
        self._pipeline = None

    @property
    def pipeline(self):
        """Loaded on first use, with an actionable error when it is missing."""
        if self._pipeline is None:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"no SVM detector at {self.model_path}. It is gitignored "
                    "(*.pkl), so a fresh checkout does not have one. Either "
                    "point PHISHNET_SVM_MODEL at an existing pipeline, or fit "
                    "one with metrics.analysis.run_svm_training(), which needs "
                    f"the CSVs listed in config.DETECTION_DATASET_PATHS under "
                    f"{config.DETECTION_SOURCES_DIR}."
                )
            import joblib

            self._pipeline = joblib.load(self.model_path)
        return self._pipeline

    def label_messages(self, bodies: List[str]) -> List[bool]:
        """True where the SVM does *not* call a message phishing."""
        if not bodies:
            return []
        predictions = self.pipeline.predict(list(bodies))
        return [int(prediction) != PHISHING_CLASS for prediction in predictions]

    def score_messages(self, bodies: List[str]) -> List[float]:
        """A 0-1 "safe" score: the logistic of the negated decision margin.

        Uncalibrated by construction — see the module docstring. Falls back to
        the hard label when the pipeline exposes no `decision_function`.
        """
        if not bodies:
            return []

        classifier = self.pipeline[-1] if hasattr(self.pipeline, "__getitem__") else None
        if classifier is None or not hasattr(classifier, "decision_function"):
            return [1.0 if safe else 0.0 for safe in self.label_messages(bodies)]

        margins = self.pipeline.decision_function(list(bodies))
        # positive margin = phishing, so negate to make "higher means safer"
        return [1.0 / (1.0 + math.exp(min(60.0, max(-60.0, margin)))) for margin in margins]


def get_svm_detector(model_path: Optional[str] = None) -> SvmDetector:
    """One shared detector, so the pipeline is unpickled once."""
    global _SHARED_DETECTOR
    if _SHARED_DETECTOR is None or (
        model_path is not None and model_path != _SHARED_DETECTOR.model_path
    ):
        _SHARED_DETECTOR = SvmDetector(model_path)
    return _SHARED_DETECTOR


def svm_available(model_path: Optional[str] = None) -> bool:
    """Whether a fitted pipeline is on disk — the loop skips scoring if not."""
    return os.path.isfile(model_path or config.SVM_MODEL_PATH)


def unload_svm_detector() -> None:
    global _SHARED_DETECTOR
    _SHARED_DETECTOR = None
