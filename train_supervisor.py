# train_supervisor.py  --  Gemma 4 E2B Security Supervisor Fine-Tuning
#
# Cell 1 (Kaggle):
#   !pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git" -q
#   !pip install trl==0.19.1 gguf -q
# Settings: GPU T4 x1 OR x2, Internet ON.
# Attach dataset with supervisor_train.jsonl + supervisor_eval.jsonl.
# Output: gemma-4-e2b-supervisor.Q4_K_M.gguf  (download from Output tab)

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
log = logging.getLogger("train")

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
#    unsloth/models/rl.py and ALWAYS calls compute_loss(return_outputs=True), then
#    converts full logits to fp32 — ignoring prediction_loss_only=True completely.
#    On T4 with a 2B model: logits ~ 7 GiB -> OOM.
#    Fix: override to bypass Unsloth's patch and return (loss, None, None).
# ---------------------------------------------------------------------------
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs)

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Bypass Unsloth's patched prediction_step to prevent eval OOM."""
        inputs = self._prepare_inputs(inputs)
        result = super(SFTTrainer, self).compute_loss(model, inputs, return_outputs=True)
        loss = result[0] if isinstance(result, tuple) else result
        return (loss.detach(), None, None)  # (loss, logits=None, labels=None)



def find(name: str) -> "str | None":
    """Locate a file under /kaggle/input/ or the working directory."""
    for p in [*Path("/kaggle/input").rglob(name), Path(name)]:
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
    """Reclaim /kaggle/working space before GGUF export."""
    if not is_kaggle():
        return
    for d in ("gguf_out", "gguf_out_gguf", "smoketest_gguf_out", "smoketest_gguf_out_gguf"):
        shutil.rmtree(d, ignore_errors=True)
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        for entry in hf_hub.iterdir():
            if entry.name.startswith("models--"):
                shutil.rmtree(entry, ignore_errors=True)
        log.info("Cleared HuggingFace hub cache to free /kaggle/working disk.")


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
        log.error("No .gguf file found under %s.", staging_dir)
        return False
    size_gb = src.stat().st_size / (1024 ** 3)
    log.info("Copying %s (%.2f GB) -> %s ...", src, size_gb, dest)
    print(f"Copying GGUF to {dest} ({size_gb:.2f} GB) ...", flush=True)
    shutil.copy2(src, dest)
    modelfile = Path(staging_dir + "_gguf") / "Modelfile"
    if modelfile.is_file():
        shutil.copy2(modelfile, "Modelfile")
    log.info("Done: %s  -- download from the Kaggle Output tab.", dest)
    print(f"SUCCESS: {dest} ready in Output tab.", flush=True)
    return True


def main() -> None:
    MODEL   = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
    SEQ_LEN = 2048       # safe ceiling for E2B on 14.5 GB T4
    LR_DIR  = "lora_out"
    GGUF    = "gemma-4-e2b-supervisor.Q4_K_M.gguf"

    train_file = find("supervisor_train.jsonl")
    eval_file  = find("supervisor_eval.jsonl")
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

    # --- Data ---

    def fmt(batch):
        return {"text": [tokenizer.apply_chat_template(c, tokenize=False,
                         add_generation_prompt=False)
                         for c in batch["messages"]]}

    ds   = load_dataset("json", data_files=train_file, split="train").map(fmt, batched=True)
    ds   = ds.remove_columns([c for c in ds.column_names if c != "text"])
    eval_ds = None
    if eval_file:
        eval_ds = load_dataset("json", data_files=eval_file, split="train").map(fmt, batched=True)
        eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c != "text"])

    # --- Collator ---
    # packing=False is mandatory — packing + DataCollatorForCompletionOnlyLM -> ValueError
    #
    # AUTO-DETECT the response template instead of hardcoding it.
    # Problem: chat template format varies by Unsloth/Gemma4 version:
    #   - Old: '<start_of_turn>model\n'
    #   - New: '<|turn>model\n'
    # Solution: inject unique sentinels, run apply_chat_template, regex-extract
    # what sits between them — that IS the response template, version-independent.
    import re as _re

    _SU = "XSENTINELUSERX"
    _SA = "XSENTINELASSTX"

    _fake_convo = [
        {"role": "user",      "content": _SU},
        {"role": "assistant", "content": _SA},
    ]
    _fake_text = tokenizer.apply_chat_template(
        _fake_convo, tokenize=False, add_generation_prompt=False)
    _fake_ids  = list(raw_tok.encode(_fake_text, add_special_tokens=False))

    log.info("Fake conversation text : %r", _fake_text)
    log.info("Fake conversation IDs  : %s", _fake_ids)

    _m = _re.search(_re.escape(_SU) + r"(.+?)" + _re.escape(_SA), _fake_text, _re.DOTALL)
    if not _m:
        raise RuntimeError(
            f"Sentinels not found in apply_chat_template output: {_fake_text!r}\n"
            "The chat template doesn't separate user/assistant turns as expected.")

    TEMPLATE = _m.group(1)
    log.info("Auto-detected response template text: %r", TEMPLATE)

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
    out = collator([raw_tok(ds[0]["text"])])
    n   = (out["labels"] != -100).sum().item()
    log.info("Supervised tokens in first example: %d", n)
    if n == 0:
        raise ValueError(
            "0 supervised tokens — response_template IDs not found in tokenized input.\n"
            f"Template IDs tried: {response_template_ids}\n"
            "Inspect ds[0]['text'] and update the template.")

    # --- Train ---
    # prediction_loss_only=True  — CRITICAL: prevents eval from materialising ALL
    # logits in fp32 (vocab_size x seq_len x batch), which OOMs on a 14 GB T4.
    # eval_accumulation_steps=1  — releases eval tensors after each example.
    eval_kw = (
        {"eval_strategy": "steps", "eval_steps": 50,
         "prediction_loss_only": True, "eval_accumulation_steps": 1}
        if eval_ds else {}
    )
    trainer = CustomSFTTrainer(
        model=model, processing_class=raw_tok,
        train_dataset=ds, eval_dataset=eval_ds, data_collator=collator,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10, num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="paged_adamw_8bit",   # pages optimizer states to CPU under VRAM pressure
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=LR_DIR,
            save_strategy="no",         # no checkpoints -- saves Kaggle disk
            report_to="none",
            max_seq_length=SEQ_LEN,
            dataset_text_field="text",
            dataset_num_proc=1,
            packing=False,              # incompatible with DataCollatorForCompletionOnlyLM
            **eval_kw,
        ),
    )

    log.info("Training (~60-80 min on T4)...")
    trainer.train()

    # --- Save LoRA adapters first (lightweight fallback if GGUF export fails) ---
    model.save_pretrained(LR_DIR); tokenizer.save_pretrained(LR_DIR)
    del trainer; gc.collect(); torch.cuda.empty_cache()

    # --- GGUF export (stage in /tmp on Kaggle — /kaggle/working is only 20 GB) ---
    free_disk_before_gguf()
    GG_DIR = gguf_staging_dir("gguf_out")
    log.info("Exporting GGUF Q4_K_M (staging=%s, output=%s) (~10-20 min)...", GG_DIR, GGUF)
    model.save_pretrained_gguf(GG_DIR, tokenizer, quantization_method="q4_k_m")
    print("Unsloth GGUF export returned — copying to /kaggle/working ...", flush=True)

    ok = publish_gguf(GG_DIR, GGUF, log)
    if not ok:
        log.error("Adapters kept in %s/", LR_DIR)

    for d in [GG_DIR, GG_DIR + "_gguf", LR_DIR]:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
