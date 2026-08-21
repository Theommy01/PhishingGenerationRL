"""ScamLLM: the detector the loop is training against.

Follows the phishnet AuxiliaryModel/Labeller/Label pattern, hence the file
names — `ScamAuxiliaryModel` is the pipeline, `ScamLabeller` maps its labels to
a score, `ScamLabel` is the label itself.

`ScamLabeller.parse_model_output` is the only code in the project that turns
ScamLLM's labels into a number, and `get_scam_labeller()` returns a single
shared instance so the model loads into VRAM once. That single-source rule is
not cosmetic: the codebase used to carry three copies of that mapping, two of
which disagreed, and the published BCO/KTO results were computed with the
inverted one. See PORTING_NOTES.md §1.
"""

from scamllm.ScamAuxiliaryModel import ScamAuxiliaryModel
from scamllm.ScamLabel import ScamLabel
from scamllm.ScamLabeller import ScamLabeller, get_scam_labeller, unload_scam_labeller

__all__ = [
    "ScamAuxiliaryModel",
    "ScamLabel",
    "ScamLabeller",
    "get_scam_labeller",
    "unload_scam_labeller",
]
