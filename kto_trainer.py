# KTO Training

import inspect

import pandas as pd
import torch
import gc
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import KTOConfig, KTOTrainer
from datasets import Dataset

from metrics import config
from reference_model import attach_reference

torch.cuda.empty_cache()
gc.collect()


def kto_class_weights(n_desirable: int, n_undesirable: int) -> tuple:
    """Weights that bring the effective desirable/undesirable ratio to ~1.

    TRL wants desirable_weight * n_desirable / (undesirable_weight *
    n_undesirable) near 1. The cumulative pool's balance shifts every round as
    successes accumulate, so this is derived from the counts rather than fixed.
    """
    if not n_desirable or not n_undesirable:
        return 1.0, 1.0
    if n_desirable < n_undesirable:
        return n_undesirable / n_desirable, 1.0
    return 1.0, n_desirable / n_undesirable




def train_kto(
    num_epochs=3,
    base_model=None,
    dataset_path=None,
    output_dir=None,
    ref_mode="sft",
    sft_path=None,
):
    print("\n" + "=" * 50)
    print(f"KTO Training (Epochs: {num_epochs})")
    print("=" * 50)

    # Nota: partiamo sempre dal modello di base SFT per l'addestramento KTO
    sft_path = sft_path or config.PATH_SFT
    path_sft = base_model or sft_path
    dataset_path = dataset_path or config.MASTER_TRAINING_DATASET
    save_dir = output_dir or f"{config.MODELS_DIR}/kto_model_ep{num_epochs}"

    print("Uploading training dataset...")
    df = pd.read_json(dataset_path, lines=True)

    formatted_data = []
    for index, row in df.iterrows():
        # KTO format: richiede specificamente il label boolean
        formatted_data.append(
            {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "label": row[
                    "label"
                ],  # True if ScamLLM returns Safe (Desirable), False if Malicious (Undesirable)
            }
        )

    dataset = Dataset.from_list(formatted_data)
    n_desirable = sum(1 for row in formatted_data if row["label"])
    n_undesirable = len(formatted_data) - n_desirable
    desirable_weight, undesirable_weight = kto_class_weights(n_desirable, n_undesirable)
    print(
        f" Read {len(dataset)} samples "
        f"({n_desirable} desirable / {n_undesirable} undesirable); "
        f"weights {desirable_weight:.3f} / {undesirable_weight:.3f}."
    )
    if not n_desirable or not n_undesirable:
        print(
            "WARNING: one class is empty. KTO needs both desirable and "
            "undesirable examples to contrast; training will not learn a "
            "useful signal from this dataset."
        )

    print("\nUpload starting model for KTO training...")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path_sft,
            max_seq_length=512,
            load_in_4bit=True,
        )

    except Exception as e:
        print(f"Training model uploading exception: {e}")
        return

    print(f"\nTrainer configuration (ref_mode={ref_mode})...")
    ref_kwargs = attach_reference(model, ref_mode, path_sft, sft_path)

    training_args = KTOConfig(
        per_device_train_batch_size=2,  # <--- MODIFICATO DA 1 A 2 (Obbligatorio per KTO)
        gradient_accumulation_steps=4,  # <--- MODIFICATO DA 8 A 4 (Per bilanciare la memoria)
        max_length=512,
        max_prompt_length=256,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=5e-6,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=save_dir,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=True,  # <--- FONDAMENTALE (lascialo sempre True!)
        # Parametri specifici di KTO
        beta=0.1,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        **ref_kwargs,
    )

    # Patch per Unsloth se necessario
    if "dataset" not in inspect.signature(trainer._get_train_sampler).parameters:
        original_sampler = trainer._get_train_sampler
        trainer._get_train_sampler = lambda dataset=None: original_sampler()

    print("\nStarting KTO training...")
    trainer.train()

    print(f"\nSaving FINAL KTO model in: {save_dir}")
    # Only the policy adapter. With a reference attached, peft would otherwise
    # save every adapter, leaving a stray reference/ copy of the anchor inside
    # each round's checkpoint.
    model.save_pretrained(
        save_dir, selected_adapters=[ref_kwargs.get("model_adapter_name", "default")]
    )
    tokenizer.save_pretrained(save_dir)

    print("Cleaning memory, post-training...")
    # The trainer, the model and the optimiser state reference each other, so
    # `del` only makes them collectable — the cycle collector is what actually
    # frees them. Emptying the cache first (as this did) therefore ran before
    # anything had been released, and the blocks were never returned to the
    # driver: the next model load saw a full card and offloaded to CPU.
    del model, tokenizer, trainer
    config.free_vram()

    print("KTO Training done.")


if __name__ == "__main__":
    train_kto(num_epochs=3)
