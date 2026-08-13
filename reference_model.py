"""What the KL term in BCO/KTO is anchored to.

Shared by both trainers, which previously carried identical copies of this.

KTO and BCO regularise the policy against a reference policy, so the trainer
needs log P_policy(y|x) and log P_reference(y|x) for the same batch on every
step. That is the algorithm; it is not a choice.

What *was* a choice, and was wrong, is how the reference got into memory. The
policy and the reference share the identical frozen 4-bit base and differ only
by a LoRA adapter, so the reference is loaded as a *second adapter on the same
model* and TRL swaps between them. Handing TRL a checkpoint path as `ref_model`
instead makes it build an entire separate model with
`AutoModelForCausalLM.from_pretrained`.

Measured on checkpoint-2122: policy alone 5.50 GiB, policy + reference adapter
5.65 GiB. The second-model route needs 5.5 GiB more and OOMs on an 11 GB card.

The SFT LoRA is this project's baseline — round 0 generates from it and every
drift metric is measured against round 0 — so "sft" is the default anchor.
"base" anchors to the raw pretrained Llama instead, which is what the notebook
did implicitly; it is kept for reproducing those runs, but it measures
divergence from a model that plays no other part in the experiment.
"""

import os

REF_ADAPTER_NAME = "reference"

REF_MODES = ("sft", "previous", "base")


def resolve_ref_path(ref_mode: str, base_model: str, sft_path: str):
    """The checkpoint the KL term is measured against, or None.

    "sft"      -> the SFT checkpoint, pinned for every round, so divergence is
                  always measured from the tuned baseline. The default.
    "previous" -> the checkpoint this round trains from, so the anchor moves
                  and divergence from SFT compounds across rounds.
    "base"     -> None: no reference adapter is attached, so TRL disables the
                  LoRA and anchors to the raw pretrained Llama. Costs no extra
                  memory. What the notebook did implicitly.

    At round 1 "sft" and "previous" coincide, since the round trains from SFT.
    """
    if ref_mode == "sft":
        return sft_path
    if ref_mode == "previous":
        return base_model
    if ref_mode == "base":
        return None
    raise ValueError(f"unknown ref_mode: {ref_mode!r} {REF_MODES}")


def attach_reference(model, ref_mode: str, base_model: str, sft_path: str) -> dict:
    """Attach the reference adapter and return the TRL trainer kwargs for it.

    `ref_model` stays None in every mode: for "base" TRL disables the LoRA, and
    otherwise it switches to the adapter attached here rather than to a separate
    model. Returns kwargs to splat into KTOTrainer/BCOTrainer.
    """
    ref_path = resolve_ref_path(ref_mode, base_model, sft_path)

    if ref_path is None:  # "base": nothing to attach
        return {"ref_model": None}

    if not os.path.isdir(ref_path):
        raise FileNotFoundError(
            f"ref_mode={ref_mode!r} needs the checkpoint at {ref_path!r}, "
            "which does not exist"
        )

    if not hasattr(model, "peft_config"):
        raise TypeError(
            f"ref_mode={ref_mode!r} needs a PEFT model to attach the reference "
            "adapter to, but got a plain model."
        )

    policy_adapter = model.active_adapter
    if isinstance(policy_adapter, list):  # peft returns a list in some versions
        policy_adapter = policy_adapter[0]

    model.load_adapter(ref_path, adapter_name=REF_ADAPTER_NAME)
    # load_adapter can leave the new adapter active; training must continue on
    # the policy one, and TRL switches to the reference only when it needs it.
    model.set_adapter(policy_adapter)

    print(
        f"  reference adapter '{REF_ADAPTER_NAME}' loaded from {ref_path} "
        f"(policy adapter: '{policy_adapter}')"
    )

    return {
        "ref_model": None,
        "model_adapter_name": policy_adapter,
        "ref_adapter_name": REF_ADAPTER_NAME,
    }
