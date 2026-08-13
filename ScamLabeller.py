from phishnet_feature_engineering.labelling.labellers.Labeller import Labeller
from ScamAuxiliaryModel import ScamAuxiliaryModel
from ScamLabel import ScamLabel
from typing import List


class ScamLabeller(Labeller):
    """
    Class for labelling messages with their ScamLLM score.
    """

    auxiliary_model: ScamAuxiliaryModel = ScamAuxiliaryModel()

    def parse_model_output(self, labels: list) -> list[ScamLabel]:
        """
        Parses the output of the auxiliary model and returns a list of ScamLabel
        instances.

        labels: one list of {"label", "score"} dicts per message, as returned by
                ScamAuxiliaryModel.predict
        returns: list of ScamLabel instances
        """
        scam_labels = []

        for label in labels:
            # The pipeline runs with top_k=None, so each entry is a list of
            # per-label dicts rather than a single {label: score} mapping.
            scores = {item["label"]: item["score"] for item in label}
            malicious_score = scores.get("LABEL_1", 0.0)
            scam_labels.append(ScamLabel(value=malicious_score))

        return scam_labels

    def label_messages(self, message_bodies: List[str]) -> List[ScamLabel]:
        labels = self.auxiliary_model.predict(message_bodies=message_bodies)
        labels = self.parse_model_output(labels=labels)
        return labels
