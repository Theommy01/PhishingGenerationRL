"""The detectors a generated message is scored against.

One is `in_loop` — ScamLLM, whose score is the reward that produced the training
labels. The rest are controls, and they exist for the question the in-loop
detector cannot answer about itself: when evasion improves, did the policy learn
something about phishing text, or something about *this detector*?

Those two have very different consequences. If the gain is detector-specific,
an attack needs access to the deployed detector. If it transfers, any surrogate
will do as a training signal and the resulting generator outlives the detector
it was trained against. `metrics/transfer.py` measures the split rather than
picking a side.

The registry is open: register a spec, run the backfill, and the analysis picks
it up. Verdicts live under `detector_scores.<name>` on each message, so adding
one never changes the schema.

    from detectors import available, get_detector

    available()                              # what can run here
    get_detector("scamllm").score_messages(bodies)
"""

from detectors.base import (  # noqa: F401
    Detector,
    DetectorSpec,
    available,
    get_detector,
    get_spec,
    in_loop_detector,
    register,
    registry,
    score_with,
    unload_all,
)
from detectors.scamllm import get_scam_labeller, unload_scam_labeller  # noqa: F401


def _build_scamllm():
    return get_scam_labeller()


def _build_bert_phishing():
    from detectors.huggingface import HuggingFaceDetector

    return HuggingFaceDetector(
        model_id="ealvaradob/bert-finetuned-phishing", phishing_label="phishing"
    )


def _build_svm():
    from detectors.svm import get_svm_detector

    return get_svm_detector()


def _svm_available() -> bool:
    from detectors.svm import svm_available

    return svm_available()


register(
    DetectorSpec(
        name="scamllm",
        build=_build_scamllm,
        description=(
            "phishbot/ScamLLM. The detector the loop trains against — its score "
            "is the reward and the source of the desirable/undesirable labels."
        ),
        kind="transformer",
        in_loop=True,
    )
)

register(
    DetectorSpec(
        name="bert-phishing",
        build=_build_bert_phishing,
        description=(
            "ealvaradob/bert-finetuned-phishing. Held out: trained on different "
            "data, never optimised against. Its label mapping was checked "
            "empirically, not read off the model card."
        ),
        kind="transformer",
    )
)

register(
    DetectorSpec(
        name="svm",
        build=_build_svm,
        description=(
            "TF-IDF + linear SVM from metrics.analysis. Held out, and a "
            "bag-of-words model rather than a transformer, so shared artefacts "
            "with ScamLLM are unlikely — which makes transfer to it strong "
            "evidence. Needs a fitted pipeline on disk."
        ),
        kind="sklearn",
        is_available=_svm_available,
    )
)
