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

# --- GLOBAL ROCM & PY3.12 WORKAROUND ---
# This must run before ANY transformers/torch imports
import torch

# 1. FIX: module 'torch' has no attribute 'int1'
for _int_type in range(1, 9):
    _attr = f"int{_int_type}"
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.int8)
print("🛠️  Patched missing 'torch.intX' attributes")

# 2. FIX: torch.utils._pytree has no attribute 'register_constant'
# This happens in some ROCm torch builds where torchao/transformers expect a newer pytree API
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def _mock_register_constant(cls):
        # Mocking the registration — doesn't affect standard causal LM training
        return cls
    torch.utils._pytree.register_constant = _mock_register_constant
    print("🛠️  Patched missing 'torch.utils._pytree.register_constant' attribute")

# 3. Disable torchao integration in transformers to avoid further issues
# We set this as early as possible
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
# Also disable bitsandbytes/flash_attn if they cause import issues in specific environments
# os.environ["BNB_CUDA_VERSION"] = "0" # Force bitsandbytes to not look for CUDA

# 4. Patch the specific bug in Torch 2.4/2.5 + Python 3.12 type hint registration
try:
    import torch._library.infer_schema
    _original_infer_schema = torch._library.infer_schema.infer_schema

    # Signature varies across torch versions, use *args, **kwargs for robustness
    def _patched_infer_schema(*args, **kwargs):
        try:
            return _original_infer_schema(*args, **kwargs)
        except ValueError as e:
            if "unsupported type torch.Tensor" in str(e):
                # Try to extract the function from args
                fn = args[0] if args else kwargs.get('fn')
                if fn and "grouped_mm_fallback" in str(fn):
                    return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e

    torch._library.infer_schema.infer_schema = _patched_infer_schema
    print("🛠️  Applied Torch type-hint patch for ROCm + Python 3.12")
except Exception as e:
    print(f"⚠️  Failed to apply torch patch: {e}")

# Import Unsloth FIRST — must be before any transformers imports per Unsloth docs
UNSLOTH_AVAILABLE = False
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available and loaded")
except Exception as e:
    print(f"⚠️  Unsloth not available: {e}")
# ------------------------------

import json
from datasets import load_dataset, Dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/qwen-adr-lora"
DATASET_PATH = "training-data.jsonl"

# Check AMD ROCm
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
        from transformers import AutoModelForCausalLM
        return True
    except ValueError as e:
        if "grouped_mm_fallback" in str(e) or "unsupported type torch.Tensor" in str(e):
            _ROCM_BROKEN_TRANSFORMERS = True
            return False
    except Exception:
        pass
    return True

if is_rocm:
    if not _check_transformers_works():
        print("⚠️  ROCm environment has a known incompatibility in 'transformers' (grouped_mm_fallback)")
        print("   Attempting to use Unsloth as a mandatory workaround...")

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# Training arguments
TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 10,
    "logging_steps": 10,
    "save_steps": 100,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "seed": 42,
    "output_dir": OUTPUT_DIR,
    "max_grad_norm": 0.3,
}

def load_and_prepare_dataset(dataset_path: str):
    """Load and prepare the dataset for training."""
    print(f"📂 Loading dataset from {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    training_examples = []
    for line in lines:
        item = json.loads(line.strip())
        messages = item.get("messages", [])

        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"
        text += "<|assistant|>\n"

        training_examples.append({"text": text})

    dataset = Dataset.from_list(training_examples)
    print(f"✅ Loaded {len(dataset)} training examples")
    return dataset

def load_model():
    """Load Qwen3-8B with QLoRA configuration."""
    print(f"\n🔄 Loading model: {MODEL_NAME}")

    if _ROCM_BROKEN_TRANSFORMERS and not UNSLOTH_AVAILABLE:
        print("\n❌ ERROR: Your environment has a broken Transformers/Torch integration on ROCm.")
        print("   The 'grouped_mm_fallback' custom op is failing due to type hint issues.")
        print("\n🛠️  HOW TO FIX:")
        print("   1. Install Unsloth (Recommended): pip install \"unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git\"")
        raise RuntimeError("Incompatible environment for fine-tuning on ROCm.")

    # Native BF16 for AMD Instinct
    dtype = torch.bfloat16 if is_rocm else torch.float16

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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
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
        sys.exit(1)

    dataset = load_and_prepare_dataset(DATASET_PATH)

    model, tokenizer = load_model()

    trainer = train_model(model, tokenizer, dataset)

    save_model(model, tokenizer)

    print("\n" + "="*60)
    print("✅ Fine-tuning complete!")
    print("="*60)

if __name__ == "__main__":
    main()
