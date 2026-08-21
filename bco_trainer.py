import inspect

import pandas as pd
import torch
import gc
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import BCOConfig, BCOTrainer
from datasets import Dataset

from metrics import config
from reference_model import attach_reference, training_stats

torch.cuda.empty_cache()
gc.collect()




def train_bco(
    num_epochs=3,
    base_model=None,
    dataset_path=None,
    output_dir=None,
    ref_mode="sft",
    sft_path=None,
    seed=config.DEFAULT_TRAINING_SEED,
):
    print("\n" + "=" * 50)
    print(f"BCO Training (Epochs: {num_epochs}, seed: {seed})")
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

    print(f"\nTrainer configuration (ref_mode={ref_mode})...")
    ref_kwargs = attach_reference(model, ref_mode, path_sft, sft_path)

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
        # TrainingArguments falls back to `seed` for data_seed, so this covers
        # the sampler's shuffle as well as init and dropout
        seed=seed,
        output_dir=save_dir,  # <--- MODIFICA 1: Ora salva direttamente su Drive
        save_strategy="epoch",  # <--- MODIFICA 2: Salva un checkpoint alla fine di ogni epoca
        save_total_limit=3,  # <--- MODIFICA 3 (Opzionale ma consigliata): Evita di riempire il Drive, tiene solo gli ultimi 3 checkpoint
        gradient_checkpointing=True,  # <--- AGGIUNGI QUESTA RIGA! FONDAMENTALE!
    )

    trainer = BCOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        **ref_kwargs,
    )

    if "dataset" not in inspect.signature(trainer._get_train_sampler).parameters:
        original_sampler = trainer._get_train_sampler
        trainer._get_train_sampler = lambda dataset=None: original_sampler()

    print("\nStarting BCO training...")
    trainer.train()

    # BCO's loss has no per-step KL for TRL to log — its formulation matches an
    # underlying distribution rather than penalising a divergence — so these
    # stats carry the losses only. `policy_kl.py` measures the divergence for
    # BCO rounds instead, on the text the round generated.
    stats = training_stats(trainer.state.log_history)

    print(f"\nSaving FINAL BCO model in: {save_dir}")
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

    print("BCO Training done.")
    return stats


if __name__ == "__main__":
    train_bco(num_epochs=3)
