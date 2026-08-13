from phishnet_feature_engineering.labelling.labels.Label import Label


class ScamLabel(Label):
    """
    Class for ScamLLM labels.

    The value is the LABEL_1 probability. Per the reward function's own comment
    ("LABEL_0 = Safe, LABEL_1 = Malicious") that is the probability of the
    message being malicious — but see PORTING_NOTES.md §1: the notebook's two
    scorers disagree about which label means safe, so confirm the polarity
    against the model card before relying on it.
    """

    value: float
