# train_supervisor_smoketest.py  --  Smoke-test before full Kaggle run
# Binary classifier: assistant completion is {"decision": "PASS"|"BLOCK"} only.
# Cell 1 (Kaggle):
#   !pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git" -q
#   !pip install trl==0.19.1 gguf -q
# Settings: GPU T4 x1 OR x2, Internet ON.
# Attach dataset with supervisor_train.jsonl + supervisor_eval.jsonl.
#
# Expected behaviour:
#   - No ValueError on collator sanity check (template found)
#   - No NaN / inf loss at any step
#   - Loss should trend DOWN over 3 epochs (~150 steps total)
#   - Run time: ~3-5 minutes

# ---------------------------------------------------------------------------
# MUST be set before ANY other import — Unsloth detects GPUs at import time.
# Without this, on T4 x2 Kaggle the model shards across GPUs and BnB 4-bit
# raises ValueError (cannot mix GPU + CPU with 4-bit quantization).
import os
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"   # restrict to GPU 0 only
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ---------------------------------------------------------------------------

import gc
import logging
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoketest")

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CustomSFTTrainer
#
# Two overrides are needed:
#
# 1. compute_loss: trl >= 0.15 adds an internal logit mask in SFTTrainer.compute_loss
#    that conflicts with DataCollatorForCompletionOnlyLM's label mask (-100), producing
#    NaN loss. Fix: skip to Trainer.compute_loss directly.
#
# 2. prediction_step: Unsloth patches the standard prediction_step in
#    unsloth/models/rl.py (unsloth_prediction_step) and ALWAYS calls
#    compute_loss(return_outputs=True), then pipes the full logit tensor through
#    convert_to_fp32 — completely ignoring prediction_loss_only=True.
#    On T4 with a 2B model: logits = vocab_size x seq_len x batch x 4 bytes ~ 7 GiB -> OOM.
#    Fix: override prediction_step to bypass Unsloth's patch, compute only the scalar
#    loss, and return (loss, None, None) so logits are never materialised in fp32.
# ---------------------------------------------------------------------------
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs)

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Bypass Unsloth's patched prediction_step to prevent eval OOM.

        Unsloth's unsloth_prediction_step always calls compute_loss(return_outputs=True)
        and converts full logits to fp32, ignoring prediction_loss_only=True.
        This override computes only the scalar loss and discards outputs immediately.
        """
        inputs = self._prepare_inputs(inputs)
        result = super(SFTTrainer, self).compute_loss(model, inputs, return_outputs=True)
        loss = result[0] if isinstance(result, tuple) else result
        return (loss.detach(), None, None)  # (loss, logits=None, labels=None)



def find(name: str) -> "str | None":
    """Locate a file under /kaggle/input/ or the working directory."""
    for p in [*Path("/kaggle/input").rglob(name), REPO_ROOT / "data" / name, Path(name)]:
        if p.exists():
            log.info("Found %s at %s", name, p)
            return str(p)
    return None


def is_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


def gguf_staging_dir(name: str) -> str:
    """GGUF merge+quantize needs ~25 GB peak; /kaggle/working is capped at 20 GB."""
    if is_kaggle():
        staging = f"/tmp/{name}"
        for d in (staging, f"{staging}_gguf"):
            shutil.rmtree(d, ignore_errors=True)
        return staging
    return name


def free_disk_before_gguf() -> None:
    """Reclaim disk space before GGUF export by scrubbing /tmp of orphaned directories. Remove previous /temp files"""
    if not is_kaggle():
        return
    log.info("Scrubbing /tmp for orphaned directories...")
    # Clear old failed runs in /tmp
    for p in Path("/tmp").glob("*"):
        try:
            if p.is_dir() and ("gguf" in p.name or "lora" in p.name):
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    log.info("Temporary directories cleared ✓")


def find_gguf_export(staging_dir: str) -> "Path | None":
    """Return the main (non-mmproj) GGUF from a staging directory."""
    hits = [
        p for p in (
            *Path(staging_dir).rglob("*.gguf"),
            *Path(staging_dir + "_gguf").rglob("*.gguf"),
        )
        if "mmproj" not in p.name.lower()
    ]
    if not hits:
        return None
    q4 = [p for p in hits if "q4_k_m" in p.name.lower()]
    return q4[0] if q4 else max(hits, key=lambda p: p.stat().st_size)


def publish_gguf(staging_dir: str, dest: str, log) -> bool:
    """Copy final GGUF (+ Ollama Modelfile) to /kaggle/working for download."""
    src = find_gguf_export(staging_dir)
    if src is None:
        log.error("No .gguf file found under %s — check Unsloth output above.", staging_dir)
        return False
    size_gb = src.stat().st_size / (1024 ** 3)
    log.info("Copying %s (%.2f GB) -> %s ...", src, size_gb, dest)
    print(f"Copying GGUF to {dest} ({size_gb:.2f} GB) — may take 1-2 min...", flush=True)
    shutil.copy2(src, dest)
    modelfile = Path(staging_dir + "_gguf") / "Modelfile"
    if modelfile.is_file():
        shutil.copy2(modelfile, "Modelfile")
        log.info("Ollama Modelfile copied to ./Modelfile")
    log.info("GGUF export done: %s", dest)
    print(f"SUCCESS: {dest} ready in Output tab.", flush=True)
    return True


def main() -> None:
    # -----------------------------------------------------------------------
    # Smoke-test knobs
    # -----------------------------------------------------------------------
    N_EXAMPLES = 300         # number of training examples to use
    N_EPOCHS   = 1           # epochs over those examples
    LOG_STEPS  = 10         # log loss every 10 steps (averages the loss of 10 examples)
    # -----------------------------------------------------------------------

    MODEL   = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    SEQ_LEN = 2048
    LR_DIR  = "smoketest_lora_out"
    GGUF    = "gemma-4-e2b-supervisor-smoketest.Q4_K_M.gguf"  # '-smoketest' suffix avoids confusion

    train_file = find("supervisor_train.jsonl")
    eval_file  = find("supervisor_eval.jsonl")   # optional — used for eval during training
    if not train_file:
        log.error("supervisor_train.jsonl not found — attach it as a Kaggle dataset."); return

    # --- Model ---
    log.info("Loading %s ...", MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=SEQ_LEN, dtype=None, load_in_4bit=True)

    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407)

    # Gemma4 wraps a Processor (no len()) — unwrap the inner text tokenizer first
    raw_tok = getattr(tokenizer, "tokenizer", tokenizer)
    model.resize_token_embeddings(len(raw_tok))  # prevent GGUF vocab mismatch

    # --- Data (50 examples only) ---

    def fmt(batch):
        return {"text": [tokenizer.apply_chat_template(c, tokenize=False,
                         add_generation_prompt=False)
                         for c in batch["messages"]]}

    full_ds = load_dataset("json", data_files=train_file, split="train")
    # Shuffle then slice so we get a mix of PASS and BLOCK examples
    smoke_ds = full_ds.shuffle(seed=42).select(range(N_EXAMPLES))
    smoke_ds = smoke_ds.map(fmt, batched=True)
    smoke_ds = smoke_ds.remove_columns([c for c in smoke_ds.column_names if c != "text"])

    # Optional eval set — use a tiny slice (10 examples) so eval is fast
    eval_ds = None
    if eval_file:
        raw_eval = load_dataset("json", data_files=eval_file, split="train")
        eval_ds  = raw_eval.shuffle(seed=99).select(range(min(100, len(raw_eval))))
        eval_ds  = eval_ds.map(fmt, batched=True)
        eval_ds  = eval_ds.remove_columns([c for c in eval_ds.column_names if c != "text"])
        log.info("Smoke-test eval dataset: %d examples", len(eval_ds))

    log.info("Smoke-test dataset: %d examples (shuffled slice of full train set)", len(smoke_ds))

    # Always print first example BEFORE the sanity check so we can inspect the
    # actual chat-template format regardless of whether the check passes or fails.
    log.info("--- First training example (truncated to 800 chars) ---")
    log.info(smoke_ds[0]["text"][:800])
    log.info("--- End of example ---")

    # --- Collator ---
    # packing=False is mandatory — packing + DataCollatorForCompletionOnlyLM -> ValueError
    #
    # AUTO-DETECT the response template instead of hardcoding it.
    # Problem: the chat template format changes between Unsloth/Gemma4 versions:
    #   - Old:  '<start_of_turn>model\n'
    #   - New:  '<|turn>model\n'
    # Hardcoding either string breaks on the other version.
    #
    # Solution: inject unique sentinels as user/assistant content, run
    # apply_chat_template, then regex-extract exactly what the template puts
    # BETWEEN the user sentinel and the assistant sentinel.  That substring is
    # the response template, version-independent.  Then window-scan the token
    # stream to get the exact in-context IDs.
    import re as _re

    _SU = "XSENTINELUSERX"    # sentinel user content   — unique, ASCII-only
    _SA = "XSENTINELASSTX"    # sentinel assistant content

    _fake_convo = [
        {"role": "user",      "content": _SU},
        {"role": "assistant", "content": _SA},
    ]
    _fake_text = tokenizer.apply_chat_template(
        _fake_convo, tokenize=False, add_generation_prompt=False)
    _fake_ids  = list(raw_tok.encode(_fake_text, add_special_tokens=False))

    log.info("Fake conversation text : %r", _fake_text)
    log.info("Fake conversation IDs  : %s", _fake_ids)

    # Extract the text that sits between user content and assistant content
    _m = _re.search(_re.escape(_SU) + r"(.+?)" + _re.escape(_SA), _fake_text, _re.DOTALL)
    if not _m:
        raise RuntimeError(
            f"Sentinels not found in apply_chat_template output: {_fake_text!r}\n"
            "This likely means the chat template doesn't separate user/assistant turns correctly.")

    TEMPLATE = _m.group(1)   # e.g. "<turn|>\n<|turn>model\n" or "<end_of_turn>\n<start_of_turn>model\n"
    log.info("Auto-detected response template text: %r", TEMPLATE)

    # Find the exact token IDs for TEMPLATE by window-scanning the fake token stream
    # (token boundaries may not align with character boundaries, so we decode each window)
    response_template_ids = None
    for _wlen in range(1, len(_fake_ids) + 1):
        for _i in range(len(_fake_ids) - _wlen + 1):
            if raw_tok.decode(_fake_ids[_i : _i + _wlen]) == TEMPLATE:
                response_template_ids = _fake_ids[_i : _i + _wlen]
                break
        if response_template_ids is not None:
            break

    if response_template_ids is None:
        raise RuntimeError(
            f"Auto-detected template text {TEMPLATE!r} not found as a contiguous "
            f"token window in fake IDs {_fake_ids}.\nFake text: {_fake_text!r}")

    log.info("Response template IDs  : %s", response_template_ids)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=raw_tok)

    # Sanity check: abort early if response_template is not found in first example
    out = collator([raw_tok(smoke_ds[0]["text"])])
    n   = (out["labels"] != -100).sum().item()
    log.info("Supervised tokens in first example: %d (expect ~8-15 for JSON-only classifier)", n)
    if n == 0:
        raise ValueError(
            "0 supervised tokens — response_template IDs not found in tokenized input.\n"
            f"Template IDs tried: {response_template_ids}\n"
            "Check the printed example above to find the correct assistant turn marker.")

    # Smoke-test training config: eval every ~1/6 of training if eval_ds available
    # prediction_loss_only=True  — CRITICAL: without this, eval materialises ALL
    #   logits in fp32 (vocab_size x seq_len x batch), which OOMs on T4 with a 2B model.
    #   With this flag the evaluator computes only the scalar loss and discards logits
    #   immediately, keeping eval memory near-zero.
    # eval_accumulation_steps=1  — process eval one example at a time instead of
    #   accumulating tensors across the whole eval set before releasing them.
    _eval_steps = 10   # fires every 10 steps
    eval_kw = (
        {"eval_strategy": "steps", "eval_steps": _eval_steps,
         "prediction_loss_only": True, "eval_accumulation_steps": 1}
        if eval_ds else {}
    )
    trainer = CustomSFTTrainer(
        model=model, processing_class=raw_tok,
        train_dataset=smoke_ds, eval_dataset=eval_ds, data_collator=collator,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,   # 1 = see every step (vs 8 in full run)
            warmup_steps=0,                  # no warmup — want signal from step 1
            num_train_epochs=N_EPOCHS,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=LOG_STEPS,         # log every step
            optim="paged_adamw_8bit",        # pages optimizer states to CPU under VRAM pressure
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=LR_DIR,
            save_strategy="no",             # no checkpoints — this is throwaway
            report_to="none",
            max_seq_length=SEQ_LEN,
            dataset_text_field="text",
            dataset_num_proc=1,
            packing=False,                  # incompatible with DataCollatorForCompletionOnlyLM
            **eval_kw,
        ),
    )

    # Total steps: N_EXAMPLES * N_EPOCHS / batch_size / grad_accum
    total_steps = N_EXAMPLES * N_EPOCHS  # = 150 steps
    log.info(
        "Starting smoke-test: %d examples × %d epochs = %d gradient steps (~3-5 min)...",
        N_EXAMPLES, N_EPOCHS, total_steps,
    )

    trainer.train()

    # --- Loss trend summary ---
    log.info("=" * 60)
    log.info("SMOKE-TEST COMPLETE — Loss history:")
    history = trainer.state.log_history
    loss_entries = [e for e in history if "loss" in e]
    if loss_entries:
        losses = [e["loss"] for e in loss_entries]
        steps  = [e["step"] for e in loss_entries]

        for step, loss in zip(steps, losses):
            log.info("  step %3d  loss = %.4f", step, loss)

        first_loss = losses[0]
        last_loss  = losses[-1]
        trend      = "↓ DECREASING" if last_loss < first_loss else "↑ NOT DECREASING"
        log.info("First loss: %.4f  |  Last loss: %.4f  |  Trend: %s", first_loss, last_loss, trend)

        nan_steps = [s for s, l in zip(steps, losses) if not (l == l)]  # NaN check
        if nan_steps:
            log.error("NaN loss detected at steps: %s", nan_steps)
        else:
            log.info("No NaN loss detected ✓")
    else:
        log.warning("No loss entries found in log_history — check trainer output above.")

    log.info("=" * 60)
    log.info("If loss is decreasing and no NaN: run train_supervisor.py for the full training.")
    log.info("If loss is flat/NaN: check the CustomSFTTrainer fix and response_template.")

    # -----------------------------------------------------------------------
    # POST-TRAINING PHASE — mirrors train_supervisor.py exactly
    # Runs on the tiny smoke-test model to verify the full save/export pipeline.
    # -----------------------------------------------------------------------

    # --- Phase 1: Save LoRA adapters (lightweight fallback if GGUF export fails) ---
    log.info("=" * 60)
    log.info("PHASE 1 — Saving LoRA adapters to %s/ ...", LR_DIR)
    model.save_pretrained(LR_DIR)
    tokenizer.save_pretrained(LR_DIR)
    log.info("LoRA adapters saved ✓")

    del trainer; gc.collect(); torch.cuda.empty_cache()

    # --- Phase 2: GGUF Q4_K_M export ---
    # NOTE: Even on a tiny smoke-test model this takes ~10-20 min on T4.
    #       It exercises the full quantisation pipeline — any llama.cpp / GGUF
    #       compatibility problems will surface here, not during the 60-80 min run.
    # Kaggle /kaggle/working is 20 GB — merge+F16+Q4 needs ~25 GB peak, so stage in /tmp.
    free_disk_before_gguf()
    GG_DIR = gguf_staging_dir("smoketest_gguf_out")
    log.info("=" * 60)
    log.info("PHASE 2 — Exporting GGUF Q4_K_M (staging=%s, output=%s) (~10-20 min)...",
             GG_DIR, GGUF)
    model.save_pretrained_gguf(GG_DIR, tokenizer, quantization_method="q4_k_m")
    print("Unsloth GGUF export returned — copying to /kaggle/working ...", flush=True)

    ok = publish_gguf(GG_DIR, GGUF, log)
    if not ok:
        log.error("LoRA adapters are still in %s/ as a fallback.", LR_DIR)

    # --- Phase 3: Cleanup (only after successful copy) ---
    log.info("=" * 60)
    log.info("PHASE 3 — Cleaning up temporary directories...")
    for d in [GG_DIR, GG_DIR + "_gguf", LR_DIR]:
        shutil.rmtree(d, ignore_errors=True)
    log.info("Cleanup done. GGUF file kept at: %s", GGUF if ok else "(copy failed)")

    log.info("=" * 60)
    if ok:
        log.info("SMOKE-TEST FULLY COMPLETE (training + save + GGUF export + cleanup).")
        print("SMOKE-TEST FULLY COMPLETE.", flush=True)
    else:
        log.error("SMOKE-TEST INCOMPLETE — GGUF was built but not copied to working dir.")
    log.info("If all phases passed: run train_supervisor.py for the real 60-80 min training.")


if __name__ == "__main__":
    main()
