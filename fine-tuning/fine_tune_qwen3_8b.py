"""
Fine-tuning Script for Qwen3-8B - STABLE VERSION
Optimized for AMD MI300X - Force FP16 to avoid NaNs
"""

import os
import sys
from pathlib import Path
import torch

# --- GLOBAL ROCM & PY3.12 WORKAROUND ---
for _int_type in range(1, 9):
    _attr = f"int{_int_type}"
    if not hasattr(torch, _attr):
        setattr(torch, _attr, torch.int8)

import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def _mock_register_constant(cls):
        return cls
    torch.utils._pytree.register_constant = _mock_register_constant

os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

try:
    import torch._library.infer_schema
    _original_infer_schema = torch._library.infer_schema.infer_schema
    def _patched_infer_schema(*args, **kwargs):
        try:
            return _original_infer_schema(*args, **kwargs)
        except ValueError as e:
            if "unsupported type torch.Tensor" in str(e):
                fn = args[0] if args else kwargs.get('fn')
                if fn and "grouped_mm_fallback" in str(fn):
                    return "transformers::grouped_mm_fallback(Tensor input, Tensor weight, Tensor offs) -> Tensor"
            raise e
    torch._library.infer_schema.infer_schema = _patched_infer_schema
except Exception:
    pass

UNSLOTH_AVAILABLE = False
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except Exception:
    pass

import json
from datasets import load_dataset, Dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./models/qwen-adr-lora"
DATASET_PATH = "training-data.jsonl"

def check_amd_rocm():
    return os.path.exists("/dev/kfd")

is_rocm = check_amd_rocm()

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# STABLE TRAINING ARGS FOR ROCM
TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "logging_steps": 5,
    "save_steps": 50,
    "learning_rate": 1e-5, # Ultra stable LR
    "weight_decay": 0.01,
    "optim": "adamw_torch",
    "lr_scheduler_type": "linear",
    "seed": 42,
    "output_dir": OUTPUT_DIR,
    "max_grad_norm": 0.5,
}

def load_and_prepare_dataset(dataset_path: str):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    training_examples = [{"text": "".join([f"<|{m['role']}|>\n{m['content']}\n" for m in json.loads(line.strip()).get("messages", [])]) + "<|assistant|>\n"} for line in lines]
    return Dataset.from_list(training_examples)

def load_model():
    dtype = torch.float16 # FORCE FP16 FOR STABILITY
    
    if UNSLOTH_AVAILABLE:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,
                dtype=dtype,
                trust_remote_code=True,
            )
            model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
            return model, tokenizer
        except Exception:
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["lora_alpha"], target_modules=LORA_CONFIG["target_modules"], lora_dropout=LORA_CONFIG["lora_dropout"], bias=LORA_CONFIG["bias"], task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, peft_config), tokenizer

def train_model(model, tokenizer, dataset):
    if UNSLOTH_AVAILABLE:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        trainer = UnslothTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=UnslothTrainingArguments(**TRAINING_ARGS, bf16=False, fp16=True, report_to="none"))
    else:
        from trl import SFTTrainer, SFTConfig
        trainer = SFTTrainer(model=model, train_dataset=dataset, processing_class=tokenizer, args=SFTConfig(**TRAINING_ARGS, bf16=False, fp16=True, report_to="none", remove_unused_columns=False, dataset_text_field="text"))
    trainer.train()
    return trainer

def main():
    dataset = load_and_prepare_dataset(DATASET_PATH)
    model, tokenizer = load_model()
    train_model(model, tokenizer, dataset)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
