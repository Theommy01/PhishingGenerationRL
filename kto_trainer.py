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

torch.cuda.empty_cache()
gc.collect()


def train_kto(num_epochs=3):
    print("\n" + "=" * 50)
    print(f"KTO Training (Epochs: {num_epochs})")
    print("=" * 50)

    # Nota: partiamo sempre dal modello di base SFT per l'addestramento KTO
    path_sft = config.PATH_SFT
    dataset_path = config.MASTER_TRAINING_DATASET
    save_dir = f"{config.MODELS_DIR}/kto_model_ep{num_epochs}"

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
    print(f" Read {len(dataset)} samples.")

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

    print("\nTrainer configuration...")

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
        desirable_weight=1.0,
        undesirable_weight=1.0,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Patch per Unsloth se necessario
    if "dataset" not in inspect.signature(trainer._get_train_sampler).parameters:
        original_sampler = trainer._get_train_sampler
        trainer._get_train_sampler = lambda dataset=None: original_sampler()

    print("\nStarting KTO training...")
    trainer.train()

    print(f"\nSaving FINAL KTO model in: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Cleaning memory, post-training...")
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("KTO Training done.")


if __name__ == "__main__":
    train_kto(num_epochs=3)
