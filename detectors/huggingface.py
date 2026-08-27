"""Held-out detectors: any HuggingFace text-classification model.

One class, parameterised by model id and by which of its labels means phishing,
so a new held-out detector is a `register(...)` call rather than a new module.

The label mapping is given explicitly and never inferred from label order.
That is not caution for its own sake: this project has already been bitten by
exactly that mistake — three copies of ScamLLM's label-to-score mapping, two of
them disagreeing, and the published BCO/KTO results computed with the inverted
one (PORTING_NOTES.md §1). `verify_direction` re-checks the mapping against two
probe texts, so a wrong one fails loudly instead of quietly inverting a result.
"""

from typing import Dict, List, Optional, Sequence

from metrics import config

# Probes for `verify_direction`. Deliberately unambiguous — this checks that a
# model points the way its config says, not that it is good at its job.
PROBE_PHISHING = (
    "Dear customer, your account has been suspended. Verify your identity "
    "immediately at <URL> or your access will be permanently revoked."
)
PROBE_BENIGN = (
    "Hi Tom, moving our Tuesday sync to 3pm as the room was double booked. "
    "The agenda is unchanged; shout if that clashes with anything."
)


class HuggingFaceDetector:
    """A text-classification pipeline behind the detector protocol."""

    def __init__(
        self,
        model_id: str,
        phishing_label: str,
        threshold: float = config.SAFE_THRESHOLD,
        max_length: int = 512,
        device: Optional[int] = None,
    ):
        self.model_id = model_id
        self.phishing_label = phishing_label
        self.threshold = threshold
        self.max_length = max_length
        self._device = device
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            device = self._device
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_id,
                device=device,
                truncation=True,
                max_length=self.max_length,
                top_k=None,
            )
        return self._pipeline

    def _phishing_probability(self, body: str) -> float:
        scores = self.pipeline(body or "")[0]
        for entry in scores:
            if entry["label"].lower() == self.phishing_label.lower():
                return float(entry["score"])
        raise KeyError(
            f"{self.model_id} produced labels {[e['label'] for e in scores]}, "
            f"none of them {self.phishing_label!r} — check the mapping"
        )

    def score_messages(self, bodies: Sequence[str]) -> List[float]:
        """Probability the message is *safe*, so higher means evaded."""
        return [1.0 - self._phishing_probability(body) for body in bodies]

    def label_messages(self, bodies: Sequence[str]) -> List[bool]:
        return [score >= self.threshold for score in self.score_messages(bodies)]

    def verify_direction(self) -> Dict[str, float]:
        """Check the label mapping against two unambiguous probes.

        Returns the two safe-scores. A correctly mapped detector scores the
        phishing probe low and the benign probe high; if that is not what comes
        back, the mapping is wrong and every number derived from it is inverted.
        """
        phishing, benign = self.score_messages([PROBE_PHISHING, PROBE_BENIGN])
        if not phishing < benign:
            raise ValueError(
                f"{self.model_id}: label mapping looks inverted — the phishing "
                f"probe scored {phishing:.3f} safe and the benign probe "
                f"{benign:.3f}. Check `phishing_label`."
            )
        return {"phishing_probe": phishing, "benign_probe": benign}

    def unload(self) -> None:
        self._pipeline = None
