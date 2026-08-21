from phishnet_feature_engineering.labelling.labellers.Labeller import Labeller
from detectors.scamllm.ScamAuxiliaryModel import ScamAuxiliaryModel
from detectors.scamllm.ScamLabel import ScamLabel
from typing import List


class ScamLabeller(Labeller):
    """
    Class for labelling messages with their ScamLLM score.

    This is the only ScamLLM scorer in the project. Everything that needs a
    number out of ScamLLM — the training-set builder, the RL loop, the
    evaluation report — goes through `label_messages` or `score_messages`, so
    the label-to-score mapping is defined once, in `parse_model_output`.
    """

    auxiliary_model: ScamAuxiliaryModel = ScamAuxiliaryModel()

    def parse_model_output(self, labels: list) -> list[ScamLabel]:
        """
        Parses the output of the auxiliary model and returns a list of ScamLabel
        instances.

        labels: one list of {"label", "score"} dicts per message, as returned by
                ScamAuxiliaryModel.predict
        returns: list of ScamLabel instances holding the safe (LABEL_0)
                 probability, matching every other ScamLLM path in the codebase
        """
        scam_labels = []

        for label in labels:
            # The pipeline runs with top_k=None, so each entry is a list of
            # per-label dicts rather than a single {label: score} mapping.
            scores = {item["label"]: item["score"] for item in label}

            if "LABEL_0" in scores:
                safe_score = scores["LABEL_0"]  # LABEL_0 = safe, per the model card
            elif "LABEL_1" in scores:
                # softmax pair, so either label recovers the other
                safe_score = 1.0 - scores["LABEL_1"]
            else:
                raise ValueError(
                    f"ScamLLM returned neither LABEL_0 nor LABEL_1: {scores}. "
                    "The classifier must run with top_k=None."
                )

            scam_labels.append(ScamLabel(value=safe_score))

        return scam_labels

    def label_messages(self, message_bodies: List[str]) -> List[ScamLabel]:
        labels = self.auxiliary_model.predict(message_bodies=message_bodies)
        labels = self.parse_model_output(labels=labels)
        return labels

    def score_messages(self, message_bodies: List[str]) -> List[float]:
        """
        Safe probability per message, for callers that want plain floats.

        message_bodies: texts to be evaluated; a text may arrive wrapped as
                        ['body'] from the TRL-style call sites
        returns: one float per message, higher meaning the filter was evaded
        """
        bodies = [
            str(body[0]) if isinstance(body, list) else str(body)
            for body in message_bodies
        ]
        return [label.value for label in self.label_messages(message_bodies=bodies)]


_SHARED_LABELLER = None


def get_scam_labeller() -> ScamLabeller:
    """
    The process-wide ScamLabeller.

    ScamAuxiliaryModel builds its pipeline in a cached_property, and pydantic
    copies the field default per instance, so constructing ScamLabeller() in
    several places would put several copies of ScamLLM in VRAM. Call this
    instead.
    """
    global _SHARED_LABELLER

    if _SHARED_LABELLER is None:
        _SHARED_LABELLER = ScamLabeller()

    return _SHARED_LABELLER


def unload_scam_labeller() -> None:
    """
    Drop the shared labeller so ScamLLM leaves VRAM.

    Called between generation and training, where an 8B adapter needs the room.
    The next get_scam_labeller() rebuilds it.
    """
    global _SHARED_LABELLER

    _SHARED_LABELLER = None
