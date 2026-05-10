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
        import importlib
        importlib.import_module("transformers.modeling_utils")
        return True
    except ValueError as e:
        if "grouped_mm_fallback" in str(e):
            _ROCM_BROKEN_TRANSFORMERS = True
            return False
    except Exception:
        pass
    return True

# Probe transformers import on ROCm early, before any standard transformers code runs
if is_rocm:
    if not _check_transformers_works():
        print("⚠️  ROCm PyTorch build has incompatible transformers custom op (grouped_mm_fallback)")
        print("   Standard transformers path is disabled on this hardware")
        print("   Fine-tuning will use Unsloth only")

# Import Unsloth FIRST — must be before any transformers imports per Unsloth docs
# Unsloth patches torch before transformers loads to apply its optimizations
UNSLOTH_AVAILABLE = False
try:
    import torch
    import unsloth
    from unsloth import FastLanguageModel
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available")
except ImportError as e:
    print(f"⚠️  Unsloth not available: {e}")
    print("   Install: pip install \"unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git\"")

import json
from datasets import load_dataset, Dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/qwen-adr-lora"
DATASET_PATH = "./fine-tuning/training-data.jsonl"

# LoRA Configuration - optimized for MI300X
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
}

# Training Configuration - optimized for MI300X
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "save_steps": 100,
    "logging_steps": 10,
    "max_seq_length": MAX_SEQ_LENGTH,
    "seed": 3407,
}

def format_chat_template(example):
    """Format dataset for Qwen3 chat template."""
    messages = example.get("messages", [])

    conversation = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            conversation += f"<|system|>\n{content}\n"
        elif role == "user":
            conversation += f"<|user|>\n{content}\n"
        elif role == "assistant":
            conversation += f"<|assistant|>\n{content}\n"

    conversation += "<|assistant|>\n"

    return {"text": conversation}

def load_and_prepare_dataset(dataset_path: str):
    """Load and prepare the training dataset."""
    print(f"\n📂 Loading dataset from {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"   Loaded {len(dataset)} examples")

    dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)

    print(f"   Dataset prepared for training")
    print(f"   Example: {dataset[0]['text'][:200]}...")

    return dataset

def load_model():
    """Load Qwen3-8B with QLoRA configuration."""
    print(f"\n🔄 Loading model: {MODEL_NAME}")

    if _ROCM_BROKEN_TRANSFORMERS and not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "ROCm transformers build is broken (grouped_mm_fallback custom op) and Unsloth is not installed.\n"
            "Fix: pip install \"unsloth[rocm] @ git+https://github.com/unslothai/unsloth.git\""
        )

    dtype = "bfloat16" if is_rocm else "float16"

    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
        print("✅ Model loaded with Unsloth optimizations (Native BF16)")
    else:
        # Standard transformers path — only reached on non-ROCm or if ROCm transformers works
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