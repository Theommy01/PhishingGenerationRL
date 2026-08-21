"""The detectors a generated message is scored against.

    scamllm/   the detector the loop trains against — its score is the reward
    svm/       a TF-IDF + linear SVM trained on public phishing corpora, never
               part of the loop

Two, deliberately. If evasion of ScamLLM transfers to a detector that was never
in the loop, the policy learned something about phishing text. If it does not,
the policy learned something about *ScamLLM* — which is a finding in its own
right, but a different one, and only a second detector can tell them apart.

Both expose the same two calls, so a caller can score with either:

    score_messages(bodies) -> list[float]   higher = safer, i.e. evaded
    label_messages(bodies) -> list[bool]    True = judged safe, i.e. evaded

The scores are not on a common scale — ScamLLM's is a probability, the SVM's a
squashed margin — so compare the labels across detectors, and the scores only
within one.
"""

from detectors.scamllm import get_scam_labeller, unload_scam_labeller

__all__ = ["get_scam_labeller", "unload_scam_labeller", "get_svm_detector"]


def get_svm_detector(*args, **kwargs):
    """Imported lazily: it pulls in scikit-learn, which ScamLLM does not need."""
    from detectors.svm import get_svm_detector as build

    return build(*args, **kwargs)
