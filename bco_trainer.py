import inspect

import pandas as pd
import torch
import gc
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import BCOConfig, BCOTrainer
from datasets import Dataset

from metrics import config

torch.cuda.empty_cache()
gc.collect()


def resolve_ref_model(ref_mode: str, base_model: str, sft_path: str):
    """Reference model for the KL term.

    "base"     -> None. With a PEFT checkpoint TRL computes reference logprobs
                  by disabling the adapters, so the anchor is the raw base
                  model, not SFT. This is what the notebook did implicitly.
    "sft"      -> the SFT checkpoint, pinned for every round, so drift is always
                  measured from the tuned baseline.
    "previous" -> the checkpoint this round trains from, so the anchor moves and
                  divergence from SFT compounds across rounds.
    """
    if ref_mode == "base":
        return None
    if ref_mode == "sft":
        return sft_path
    if ref_mode == "previous":
        return base_model
    raise ValueError(f"unknown ref_mode: {ref_mode!r} (base | sft | previous)")



def train_bco(
    num_epochs=3,
    base_model=None,
    dataset_path=None,
    output_dir=None,
    ref_mode="base",
    sft_path=None,
):
    print("\n" + "=" * 50)
    print(f"BCO Training (Epochs: {num_epochs})")
    print("=" * 50)

    sft_path = sft_path or config.PATH_SFT
    path_sft = base_model or sft_path
    dataset_path = dataset_path or config.MASTER_TRAINING_DATASET
    save_dir = output_dir or f"{config.MODELS_DIR}/bco_model_ep{num_epochs}"

    print("Uploading training dataset...")
    df = pd.read_json(dataset_path, lines=True)

    formatted_data = []
    for index, row in df.iterrows():
        formatted_data.append(
            {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "label": row["label"],  # True if ScamLLM returns Safe
            }
        )

    dataset = Dataset.from_list(formatted_data)
    print(f" Read {len(dataset)} samples.")

    print("\nUpload starting model for BCO training...")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path_sft,
            max_seq_length=512,
            load_in_4bit=True,
        )

    except Exception as e:
        print(f"training model uploading exception: {e}")
        return

    ref_model = resolve_ref_model(ref_mode, path_sft, sft_path)
    print(f"\nTrainer configuration (ref_mode={ref_mode})...")

    training_args = BCOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
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
        output_dir=save_dir,  # <--- MODIFICA 1: Ora salva direttamente su Drive
        save_strategy="epoch",  # <--- MODIFICA 2: Salva un checkpoint alla fine di ogni epoca
        save_total_limit=3,  # <--- MODIFICA 3 (Opzionale ma consigliata): Evita di riempire il Drive, tiene solo gli ultimi 3 checkpoint
        gradient_checkpointing=True,  # <--- AGGIUNGI QUESTA RIGA! FONDAMENTALE!
    )

    trainer = BCOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if "dataset" not in inspect.signature(trainer._get_train_sampler).parameters:
        original_sampler = trainer._get_train_sampler
        trainer._get_train_sampler = lambda dataset=None: original_sampler()

    print("\nStarting BCO training...")
    trainer.train()

    print(f"\nSaving FINAL BCO model in: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Cleaning memory, post-training...")
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("BCO Training done.")


if __name__ == "__main__":
    train_bco(num_epochs=3)
