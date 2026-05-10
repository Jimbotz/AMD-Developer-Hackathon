"""
Fine-tuning Script for Qwen3-8B
Optimized for AMD MI300X with 192GB HBM3

Run with:
    python fine_tune_qwen3_8b.py

Expected time on MI300X: ~1.5 hours for 3 epochs
"""

import os
import sys
from pathlib import Path

# === ROCm + Python 3.12 fixes (must be before any torch/transformers imports) ===
import torch
if not hasattr(torch, "int1"):
    torch.int1 = torch.int8
if not hasattr(torch, "int2"):
    torch.int2 = torch.int8
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

try:
    import torch._library.infer_schema
    _original_infer_schema = torch._library.infer_schema.infer_schema

    def _patched_infer_schema(fn, mutates_args, error_fn=None):
        try:
            return _original_infer_schema(fn, mutates_args, error_fn)
        except ValueError as e:
            if "unsupported type torch.Tensor" in str(e):
                if "grouped_mm_fallback" in str(fn):
                    return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e

    torch._library.infer_schema.infer_schema = _patched_infer_schema
    print("🛠️  Applied ROCm + Python 3.12 fixes for torch/transformers")
except Exception:
    pass
# ========================================================

# Check AMD ROCm first — needed to configure everything else
def check_amd_rocm():
    if os.path.exists("/dev/kfd"):
        print("✅ AMD ROCm detected")
        return True
    print("⚠️  ROCm not detected, will use CUDA if available")
    return False

is_rocm = check_amd_rocm()

# Detect broken transformers custom-op on ROCm (grouped_mm_fallback type error)
_ROCM_BROKEN_TRANSFORMERS = False
def _check_transformers_works():
    global _ROCM_BROKEN_TRANSFORMERS
    try:
        import torch
        # Workaround for Python 3.12 + Transformers 4.45+ type hint bug in custom_op
        # This sometimes pre-registers the needed types
        from typing import Union, List, Optional, Sequence
        
        from transformers import AutoModelForCausalLM
        return True
    except ValueError as e:
        if "grouped_mm_fallback" in str(e) or "unsupported type torch.Tensor" in str(e):
            _ROCM_BROKEN_TRANSFORMERS = True
            return False
    except Exception:
        pass
    return True

# Probe transformers import on ROCm early, before any standard transformers code runs
if is_rocm:
    if not _check_transformers_works():
        print("⚠️  ROCm environment has a known incompatibility in 'transformers' (grouped_mm_fallback)")
        print("   This is often caused by Transformers 4.45.0 on Python 3.12.")
        print("   Attempting to use Unsloth as a mandatory workaround...")

# Import Unsloth FIRST — must be before any transformers imports per Unsloth docs
UNSLOTH_AVAILABLE = False
try:
    import torch
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available and loaded")
except Exception as e:
    print(f"⚠️  Unsloth not available: {e}")
    if is_rocm:
        print("   CRITICAL: On AMD ROCm, Unsloth is highly recommended to avoid transformers bugs.")
        print("   Install: pip install \"unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git\"")

import json
from datasets import load_dataset, Dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/qwen-adr-lora"
DATASET_PATH = "./fine-tuning/training-data.jsonl"

# ... (keep LoRA and Training configs)

def load_model():
    """Load Qwen3-8B with QLoRA configuration."""
    print(f"\n🔄 Loading model: {MODEL_NAME}")

    if _ROCM_BROKEN_TRANSFORMERS and not UNSLOTH_AVAILABLE:
        print("\n❌ ERROR: Your environment has a broken Transformers/Torch integration on ROCm.")
        print("   The 'grouped_mm_fallback' custom op is failing due to type hint issues.")
        print("\n🛠️  HOW TO FIX:")
        print("   1. Install Unsloth (Recommended): pip install \"unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git\"")
        print("   2. OR Update Transformers: pip install --upgrade transformers")
        print("   3. OR Downgrade Transformers: pip install transformers==4.44.2")
        raise RuntimeError("Incompatible environment for fine-tuning on ROCm.")

    # Native BF16 for AMD Instinct
    dtype = "bfloat16" if is_rocm else "float16"

    if UNSLOTH_AVAILABLE:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False, # Use native BF16 for MI300X
                dtype=dtype,
                trust_remote_code=True,
            )
            model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
            print(f"✅ Model loaded with Unsloth optimizations ({dtype})")
            return model, tokenizer
        except Exception as e:
            print(f"⚠️  Unsloth loading failed: {e}. Falling back to standard transformers...")

    # Standard transformers path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Fix: Use 'dtype' instead of deprecated 'torch_dtype'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype if is_rocm else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    print("✅ Model loaded with standard PEFT")
    model.print_trainable_parameters()

    return model, tokenizer


def train_model(model, tokenizer, dataset):
    """Train the model."""
    print("\n🚀 Starting training...")

    if UNSLOTH_AVAILABLE:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        
        training_args = UnslothTrainingArguments(
            **TRAINING_ARGS,
            bf16=is_rocm,
            fp16=not is_rocm,
            report_to="none",
        )

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
    else:
        from trl import SFTTrainer, SFTConfig

        sft_config = SFTConfig(
            **TRAINING_ARGS,
            bf16=is_rocm,
            fp16=not is_rocm,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            dataset_text_field="text",
        )

    print("   Training started (this may take 1-2 hours on MI300X)...")
    trainer.train()

    return trainer

def save_model(model, tokenizer):
    """Save the fine-tuned model."""
    print(f"\n💾 Saving model to {OUTPUT_DIR}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Model saved successfully!")

def main():
    """Main fine-tuning pipeline."""
    print("="*60)
    print("ADR Validator - Qwen3-8B Fine-tuning")
    print("="*60)

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"\n❌ Dataset not found at {DATASET_PATH}")
        print("   Run generate_dataset.py first to create the training data")
        sys.exit(1)

    dataset = load_and_prepare_dataset(DATASET_PATH)

    model, tokenizer = load_model()

    trainer = train_model(model, tokenizer, dataset)

    save_model(model, tokenizer)

    print("\n" + "="*60)
    print("✅ Fine-tuning complete!")
    print("="*60)
    print(f"\n📁 Model saved to: {OUTPUT_DIR}")
    print("\n💡 To use the fine-tuned model:")
    print(f"   python backend/main.py")
    print(f"   (The API will automatically use models/qwen-adr-lora)")

if __name__ == "__main__":
    main()