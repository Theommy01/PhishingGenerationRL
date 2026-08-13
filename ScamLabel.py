from phishnet_feature_engineering.labelling.labels.Label import Label


class ScamLabel(Label):
    """
    Class for ScamLLM labels.

    The value is the **safe probability**: ScamLLM's LABEL_0 score, i.e. how
    likely the message is to be judged non-malicious. 1.0 means the filter was
    fully evaded, 0.0 means it was caught.

    LABEL_0 = safe is confirmed by the model card ("Label 1 ... a phishing
    attempt, while Label 0 ... safe and non-malicious") and behaviourally:
    across 20 unambiguous emails it separates them 18/20, versus 2/20 under the
    inverse reading.

    ScamLabeller is the only ScamLLM scorer in the project, so every ScamLLM
    value in the codebase is this scale. It previously stored the LABEL_1
    (malicious) probability while metrics/ held two further, mutually
    contradictory copies of the mapping; see PORTING_NOTES.md §1.
    """

    value: float
