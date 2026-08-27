"""The detector protocol and the registry of them.

Adding a detector is a registration and a backfill pass — never a schema
change. Messages store their verdicts under `detector_scores.<name>` and
`detector_labels.<name>`, so the set can grow after a run has finished and the
analysis code takes detector names as parameters rather than having any of them
built in.

Every detector answers the same two questions about a message body:

    score_messages(bodies) -> list[float]   higher = safer, i.e. evaded
    label_messages(bodies) -> list[bool]    True = judged safe, i.e. evaded

"Higher means safer" follows ScamLLM's convention, so every detector points the
same way. The scores are *not* on a common scale — one is a softmax
probability, another a squashed SVM margin — so compare labels across detectors
and scores only within one.

Exactly one detector is `in_loop`: its score is the reward that produced the
training labels. Every other detector is a control, and the point of having
them is the question the in-loop one cannot answer about itself — whether
evasion transfers to a detector that was never optimised against.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence


class Detector(Protocol):
    """What every detector implements."""

    def score_messages(self, bodies: Sequence[str]) -> List[float]:
        ...

    def label_messages(self, bodies: Sequence[str]) -> List[bool]:
        ...


@dataclass
class DetectorSpec:
    """How to build a detector, and what it is.

    `build` is called lazily and its result cached, so registering a detector
    costs nothing until something scores with it — which matters when one of
    them is a 400 MB transformer.
    """

    name: str
    build: Callable[[], Detector]
    description: str
    kind: str = "transformer"
    in_loop: bool = False
    is_available: Callable[[], bool] = field(default=lambda: True)
    _instance: Optional[Detector] = field(default=None, repr=False, compare=False)

    def get(self) -> Detector:
        if self._instance is None:
            self._instance = self.build()
        return self._instance

    def unload(self) -> None:
        self._instance = None


_REGISTRY: Dict[str, DetectorSpec] = {}


def register(spec: DetectorSpec) -> DetectorSpec:
    """Add a detector to the registry, replacing any of the same name."""
    _REGISTRY[spec.name] = spec
    return spec


def registry() -> Dict[str, DetectorSpec]:
    return dict(_REGISTRY)


def get_spec(name: str) -> DetectorSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector: {name!r} (have {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def get_detector(name: str) -> Detector:
    """The detector itself, built on first use."""
    return get_spec(name).get()


def available() -> List[str]:
    """Detectors that can actually run here.

    A registered detector may still be unusable — the SVM needs a fitted
    pipeline on disk, a transformer needs its weights downloadable — so the
    registry reports what it has and this reports what will work.
    """
    return sorted(name for name, spec in _REGISTRY.items() if spec.is_available())


def in_loop_detector() -> Optional[str]:
    """The detector whose score is the reward, if one is registered."""
    for name, spec in sorted(_REGISTRY.items()):
        if spec.in_loop:
            return name
    return None


def score_with(names: Sequence[str], bodies: Sequence[str]) -> Dict[str, Dict[str, list]]:
    """Score one batch of bodies with several detectors.

    Returns {detector: {"scores": [...], "labels": [...]}}. Detectors are used
    one at a time and released afterwards, because two transformers resident at
    once is exactly the squeeze this project keeps running into.
    """
    results = {}
    for name in names:
        spec = get_spec(name)
        detector = spec.get()
        results[name] = {
            "scores": list(detector.score_messages(bodies)),
            "labels": list(detector.label_messages(bodies)),
        }
    return results


def unload_all() -> None:
    for spec in _REGISTRY.values():
        spec.unload()
