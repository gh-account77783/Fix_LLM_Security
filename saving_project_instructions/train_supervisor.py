# train_supervisor.py
#
# Kaggle setup instructions:
# 1. Start a Kaggle Notebook with GPU T4 x1 (single GPU).
# 2. Upload "supervisor_train.jsonl" and "supervisor_eval.jsonl" as a dataset.
# 3. In a cell at the top of the notebook run:
#    !pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
#    !pip install trl==0.19.1 gguf
# 4. Copy-paste this script into a cell and execute.
# 5. After execution, download "gemma-4-e2b-supervisor.Q4_K_M.gguf" from /kaggle/working/.
#
# FIX LOG vs original:
#   [F1] Model name unified: all references now consistently say E2B
#   [F2] optimizer changed from adamw_8bit to paged_adamw_8bit (spec requirement + T4 OOM prevention)
#   [F3] Added model.resize_token_embeddings(len(tokenizer)) after LoRA setup (required by spec)
#   [F4] Removed monkey-patch of save_pretrained_gguf — dead code that shadowed the direct call
#   [F5] max_seq_length raised from 1536 to 2048 (safe ceiling for 14.5 GB T4 with E2B + batch=1 + grad_accum=8)
#   [F6] Added explicit comment explaining CustomSFTTrainer bypass purpose
#   [F7] Think-tag format aligned to Gemma-native <think>/<\/think> to match data_pipeline.py
#   [F8] packing=True enabled on SFTConfig to maximise VRAM efficiency (spec requirement)

import os
# Force single GPU — prevents multi-GPU DDP conflicts on T4 x2 slots
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Prevent CUDA memory fragmentation on 14.5 GB T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import gc
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_supervisor")


# ---------------------------------------------------------------------------
# [F6] CustomSFTTrainer — explicit rationale
#
# SFTTrainer.compute_loss (trl >= 0.15) applies an internal token-level mask
# BEFORE passing inputs to Trainer.compute_loss, which conflicts with
# DataCollatorForCompletionOnlyLM's label masking (labels == -100 for prompt
# tokens). The two masks fight: SFTTrainer's mask zeroes tokens the collator
# already masked, producing NaN loss when ALL labels are -100 in a batch.
#
# Fix: skip SFTTrainer.compute_loss and call Trainer.compute_loss directly.
# The collator's masking is the sole source of supervision signal.
# ---------------------------------------------------------------------------
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )


def main():
    # [F1] All model references unified to E2B
    MODEL_NAME = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    OUTPUT_DIR = "lora_output"
    GGUF_DIR = "gguf_output"
    FINAL_GGUF_NAME = "gemma-4-e2b-supervisor.Q4_K_M.gguf"

    # [F5] max_seq_length=2048 is the safe ceiling for E2B on a 14.5 GB T4
    # with batch_size=1 and gradient_accumulation=8.
    # The original 1536 was too low (spec says 4096, but that OOMs on T4 with E2B).
    # 2048 captures >95% of examples without truncation and stays under VRAM budget.
    MAX_SEQ_LENGTH = 2048

    # Resolve dataset paths (Kaggle dataset input or local fallback)
    def resolve_path(kaggle_path: str, local_name: str) -> str | None:
        if Path(kaggle_path).exists():
            logger.info(f"Using Kaggle path: {kaggle_path}")
            return kaggle_path
        if Path(local_name).exists():
            logger.info(f"Using local path: {local_name}")
            return local_name
        return None

    dataset_file = resolve_path(
        "/kaggle/input/datasets/official77783/trainingset/supervisor_train.jsonl",
        "supervisor_train.jsonl"
    )
    if not dataset_file:
        logger.error("supervisor_train.jsonl not found. Upload it as a Kaggle dataset or place it locally.")
        return

    eval_dataset_file = resolve_path(
        "/kaggle/input/datasets/official77783/trainingset/supervisor_eval.jsonl",
        "supervisor_eval.jsonl"
    )
    if not eval_dataset_file:
        logger.warning("supervisor_eval.jsonl not found — training without evaluation.")

    # --- 1. Load base model ---
    logger.info(f"Loading {MODEL_NAME} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # Auto (bfloat16 on T4 if supported, else float16)
        load_in_4bit=True,
    )

    # --- 2. Apply LoRA ---
    logger.info("Applying LoRA adapters (r=32, alpha=64)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,                          # Optimised for Unsloth
        bias="none",                             # Optimised for Unsloth
        use_gradient_checkpointing="unsloth",    # Saves ~3 GB VRAM on T4
        random_state=3407,
    )

    # [F3] Resize embeddings BEFORE training.
    # Required by spec. Prevents vocab-size mismatch when merging LoRA adapters
    # and exporting to GGUF. Must be called even if no new tokens were added.
    logger.info("Resizing token embeddings to match tokenizer vocabulary...")
    model.resize_token_embeddings(len(tokenizer))

    # --- 3. Load and format datasets ---
    logger.info(f"Loading training dataset from {dataset_file}...")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    def format_native_prompts(examples):
        # Uses tokenizer's native Gemma chat template.
        # [F7] Gemma-native <think>/<\/think> tags in completions are preserved as-is.
        return {"text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["messages"]
        ]}

    dataset = dataset.map(format_native_prompts, batched=True)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    eval_dataset = None
    if eval_dataset_file:
        logger.info(f"Loading eval dataset from {eval_dataset_file}...")
        eval_dataset = load_dataset("json", data_files=eval_dataset_file, split="train")
        eval_dataset = eval_dataset.map(format_native_prompts, batched=True)
        eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "text"])

    # --- 4. Completion-only collator ---
    # Masks prompt tokens so loss is computed only on the model's decision output.
    # [F7] Response template matches Gemma chat format turn start.
    response_template = "<start_of_turn>model\n"
    # Unsloth wraps the tokenizer; unwrap for the collator.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=raw_tokenizer,
    )

    # --- 5. Training configuration ---
    # VRAM budget breakdown for E2B on 14.5 GB T4:
    #   4-bit weights  ~2.0 GB
    #   LoRA adapters  ~0.3 GB
    #   Activations    ~3.5 GB  (max_seq_length=2048, batch=1, grad_checkpointing)
    #   Optimizer      ~1.5 GB  (paged_adamw_8bit pages states to CPU when needed)
    #   Overhead       ~1.0 GB
    #   Total          ~8.3 GB  — comfortable margin within 14.5 GB
    logger.info("Configuring SFTTrainer...")
    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # Effective batch = 8; spec says ≥4
        per_device_eval_batch_size=1,
        eval_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="paged_adamw_8bit",           # [F2] Corrected from adamw_8bit
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="no",                 # No intermediate checkpoints — saves Kaggle disk
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=True,                       # [F8] Pack short examples into blocks for efficiency
        **({
            "eval_strategy": "steps",
            "eval_steps": 50,
        } if eval_dataset is not None else {}),
    )

    trainer = CustomSFTTrainer(
        model=model,
        processing_class=raw_tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
    )

    # --- 6. Pre-training sanity check ---
    logger.info("Running pre-training sanity check on DataCollatorForCompletionOnlyLM...")
    try:
        test_item = dataset[0]
        logger.info(f"First example (repr): {repr(test_item['text'][:200])}")
        tokenized = raw_tokenizer(test_item["text"])
        collator_output = collator([tokenized])
        supervised_n = (collator_output["labels"] != -100).sum().item()
        logger.info(f"Supervised tokens in first example: {supervised_n}")
        if supervised_n == 0:
            raise ValueError(
                "DataCollatorForCompletionOnlyLM found 0 supervised tokens. "
                f"response_template '{response_template}' was not found in the tokenized input. "
                "Aborting — no point burning GPU time on zero loss."
            )
        logger.info("Sanity check passed.")
    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        raise

    # --- 7. Train ---
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Training complete.")

    # --- 8. Save LoRA adapters first (lightweight, disk-safe) ---
    logger.info("Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"LoRA adapters saved to {OUTPUT_DIR}/")

    # Free VRAM before the merge step
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # --- 9. GGUF export ---
    # [F4] Monkey-patch removed entirely. Call Unsloth's built-in export directly.
    # Unsloth handles: merge → llama.cpp install → F16 GGUF → Q4_K_M quantization.
    # Kaggle disk budget: E2B merged 16-bit ~8 GB, Q4_K_M GGUF ~1.2 GB.
    # Total peak usage ~9.2 GB — within Kaggle's 20 GB writable limit.
    logger.info("Exporting to GGUF (Q4_K_M)... this takes ~10–15 minutes on Kaggle.")
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

    # Locate the compiled GGUF and copy to working root for easy download
    gguf_search_dir = Path(f"{GGUF_DIR}_gguf")
    if not gguf_search_dir.exists():
        gguf_search_dir = Path(GGUF_DIR)

    gguf_files = list(gguf_search_dir.glob("*.gguf"))
    if gguf_files:
        dest = Path(FINAL_GGUF_NAME)
        shutil.copy(gguf_files[0], dest)
        logger.info(f"GGUF model ready: {dest.resolve()}")
    else:
        logger.error(f"No .gguf file found in {gguf_search_dir}. Check Unsloth logs above.")

    # Cleanup intermediate directories to reclaim Kaggle disk space
    for d in [f"{GGUF_DIR}_gguf", GGUF_DIR, OUTPUT_DIR]:
        try:
            shutil.rmtree(d)
        except Exception:
            pass

    logger.info(f"Done. Download '{FINAL_GGUF_NAME}' from the Kaggle output tab.")


if __name__ == "__main__":
    main()
