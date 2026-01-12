#!/usr/bin/env python3
"""
generate.py - (Final Fixed Version)
修复维度冲突、浮点数问题、空值问题。
"""

import os
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
)
from peft import PeftModel
from safetensors import safe_open

# 假设 utils 在当前目录下
from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir


def get_logger(filename=None):
    logger = logging.getLogger(__name__)
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    formatter = logging.Formatter(fmt=format_str, datefmt=datefmt)
    logger.setLevel(logging.INFO)
    
    if filename is not None:
        file_handler = logging.FileHandler(filename, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def calculate_data_stats(dataset):
    """
    只计算真实数据的 vocab size。
    不需要再为了 Stage 1 做 51 的兜底，因为我们会用 Key Deletion 策略。
    """
    if not hasattr(dataset, 'metadata') or dataset.metadata is None:
        return 43 # Default fallback
    
    max_cat_id = 0
    for col_name, col_meta in dataset.metadata.items():
        if col_meta.get("type") == "categorical":
            cats = col_meta.get("categories", {})
            vocab_size = cats.get("vocab_size", 0)
            if vocab_size > 0:
                max_cat_id = max(max_cat_id, vocab_size - 1)
    
    # 直接返回真实值
    return max_cat_id + 1 if max_cat_id > 0 else 43


def load_stage1_model(base_model_path, tokenizer, num_bins, cat_vocab_size, device, logger):
    """
    Load Stage 1 base model using Key Deletion strategy.
    自动剔除形状不匹配的权重，解决 51 vs 43 问题。
    """
    logger.info(f"Loading Stage 1 base model from {base_model_path}")
    logger.info(f"Initializing model with target cat_vocab_size: {cat_vocab_size}")
    
    config = AutoConfig.from_pretrained(base_model_path)
    
    # 1. 直接用真实尺寸 (43) 初始化模型
    model = TypeAwareGPT2(
        config=config,
        num_bins=num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=None
    )
    
    # 2. 加载权重文件
    model_path = os.path.join(base_model_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_model_path, "model.safetensors")
        if os.path.exists(model_path):
            ckpt = {}
            with safe_open(model_path, framework="pt") as f:
                for k in f.keys():
                    ckpt[k] = f.get_tensor(k)
        else:
            raise FileNotFoundError(f"Model file not found in {base_model_path}")
    else:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    
    # 3. [关键修复] 智能剔除形状不匹配的 keys
    # 比如 Stage 1 是 [51, 768]，现在模型是 [43, 768]，直接删掉这层权重
    # 这层权重会由后续加载的 Stage 2 LoRA 补上，所以删了也没事
    current_model_dict = model.state_dict()
    keys_to_remove = []
    
    for k, v in ckpt.items():
        if k in current_model_dict:
            if v.shape != current_model_dict[k].shape:
                logger.warning(f"Shape mismatch for {k}: Checkpoint {v.shape} != Model {current_model_dict[k].shape}. Dropping key.")
                keys_to_remove.append(k)
    
    for k in keys_to_remove:
        del ckpt[k]
    
    # 4. 加载剩余权重 (strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    
    # Set type token IDs
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        type_ids_map = {
            "NUM": tokenizer.convert_tokens_to_ids("[NUM]"),
            "CAT": tokenizer.convert_tokens_to_ids("[CAT]"),
            "MIX": tokenizer.convert_tokens_to_ids("[MIX]"),
        }
        if all(v is not None and v != tokenizer.unk_token_id for v in type_ids_map.values()):
            model.set_type_token_ids(type_ids_map)
    
    model = model.to(device)
    model.eval()
    return model


def load_stage2_lora(model, lora_path, logger):
    """Load Stage 2 LoRA adapter and merge."""
    logger.info(f"Loading Stage 2 LoRA adapter from {lora_path}")
    
    # Load as LoRA adapter
    # 因为 cat_expert 在 modules_to_save 里，LoRA 包含了它的完整权重 (43维)。
    # 这会覆盖掉 load_stage1 时留下的随机初始化的 cat_expert。
    model = PeftModel.from_pretrained(model, lora_path)
    logger.info("LoRA adapter loaded. Merging and unloading...")
    model = model.merge_and_unload()
    logger.info("LoRA weights merged successfully")
    return model


def sample_from_expert(expert_outputs, col_name, col_type, col_meta, device, temperature=1.0):
    """
    Added col_name argument to handle integer columns specifically.
    """
    # 定义需要强制转整数的列名（针对 Adult 数据集）
    INT_COLUMNS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    col_lower = col_name.lower()

    if col_type == "numerical":
        bin_logits = expert_outputs["num_bin_logits"]
        residual = expert_outputs["num_residual"]
        bin_probs = F.softmax(bin_logits / temperature, dim=-1)
        bin_id = torch.multinomial(bin_probs, 1).squeeze(-1)
        res = residual.squeeze(-1)
        
        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])
        bin_id_np = bin_id.cpu().numpy()
        res_np = res.cpu().numpy()
        
        values = []
        for i in range(len(bin_id_np)):
            bid = int(bin_id_np[i])
            bid = max(0, min(bid, len(bin_edges) - 2))
            lower, upper = bin_edges[bid], bin_edges[bid + 1]
            width = max(upper - lower, 1e-9)
            nv = lower + res_np[i] * width
            nv = np.clip(nv, 0.0, 1.0)
            
            norm_min, norm_max = stats["norm_min"], stats["norm_max"]
            v = nv * (norm_max - norm_min) + norm_min
            if stats.get("needs_log", False):
                shift = stats.get("log_shift", 1e-6)
                v = np.exp(v) - shift
            
            # [关键修复] 强制整数化
            if col_lower in INT_COLUMNS:
                v = int(round(v))
                
            values.append(str(v))
        return values
    
    elif col_type == "categorical":
        cat_logits = expert_outputs["cat_logits"]
        cat_probs = F.softmax(cat_logits / temperature, dim=-1)
        cat_id = torch.multinomial(cat_probs, 1).squeeze(-1)
        
        categories = col_meta["categories"]
        vocab = categories["unique"]
        unk_id = categories.get("unk_id", 0) 
        
        cat_id_np = cat_id.cpu().numpy()
        values = []
        for i in range(len(cat_id_np)):
            cid = int(cat_id_np[i])
            if 0 <= cid < len(vocab):
                values.append(str(vocab[cid]))
            else:
                # [关键修复] 之前漏了这行，导致列表长度不够
                values.append(str(vocab[unk_id]))
        
        return values
    
    elif col_type == "mixed":
        mask_logits = expert_outputs["mixed_mask_logits"]
        bin_logits = expert_outputs["mixed_bin_logits"]
        residual = expert_outputs["mixed_residual"]
        
        mask_probs = torch.sigmoid(mask_logits / temperature)
        mask = torch.bernoulli(mask_probs)
        mask_np = mask.cpu().numpy()
        
        bin_probs = F.softmax(bin_logits / temperature, dim=-1)
        bin_ids = torch.multinomial(bin_probs, 1).squeeze(-1)
        bin_ids_np = bin_ids.cpu().numpy()
        residual_np = residual.cpu().numpy()
        
        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])
        norm_min, norm_max = stats["norm_min"], stats["norm_max"]
        needs_log = stats.get("needs_log", False)
        log_shift = stats.get("log_shift", 1e-6) if needs_log else 0.0
        
        values = []
        for i in range(len(mask_np)):
            if mask_np[i] < 0.5:
                values.append("0")
            else:
                bid = int(bin_ids_np[i])
                bid = max(0, min(bid, len(bin_edges) - 2))
                lower, upper = bin_edges[bid], bin_edges[bid + 1]
                width = max(upper - lower, 1e-9)
                nv = lower + residual_np[i] * width
                nv = np.clip(nv, 0.0, 1.0)
                v = nv * (norm_max - norm_min) + norm_min
                if needs_log:
                    v = np.exp(v) - log_shift
                
                # [关键修复] Mixed 类型如果是整数列，也要取整
                if col_lower in INT_COLUMNS:
                    v = int(round(v))
                    
                values.append(str(v))
        return values


def generate_column_batch(model, tokenizer, prompts, col_name, col_type, col_meta, metadata, device, temperature=1.0, key_val_sep="is"):
    type_token_map = {"numerical": "[NUM]", "categorical": "[CAT]", "mixed": "[MIX]"}
    type_token = type_token_map.get(col_type, "[NUM]")
    prefix = f"{type_token} {col_name} {key_val_sep}"
    
    full_prompts = [(p + " " + prefix) if p and not p.endswith(" ") else (p + prefix) for p in prompts]
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    encoded = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        transformer_outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = transformer_outputs[0]
    
    seq_lengths = attention_mask.sum(dim=1) - 1
    col_hidden_states = torch.cat([last_hidden_states[i, int(l):int(l)+1] for i, l in enumerate(seq_lengths)], dim=0).unsqueeze(1)
    
    with torch.no_grad():
        if col_type == "numerical":
            bin_logits, residual = model.num_expert(col_hidden_states)
            expert_outputs = {"num_bin_logits": bin_logits.squeeze(1), "num_residual": residual.squeeze(1)}
        elif col_type == "categorical":
            cat_logits = model.cat_expert(col_hidden_states)
            expert_outputs = {"cat_logits": cat_logits.squeeze(1)}
        elif col_type == "mixed":
            mask_logits, bin_logits, residual = model.mixed_expert(col_hidden_states)
            expert_outputs = {"mixed_mask_logits": mask_logits.squeeze(1), "mixed_bin_logits": bin_logits.squeeze(1), "mixed_residual": residual.squeeze(1)}
    
    # 传递 col_name 用于判断是否取整
    return sample_from_expert(expert_outputs, col_name, col_type, col_meta, device, temperature)


def generate_synthetic_data(model, tokenizer, dataset, metadata, num_samples, device, temperature=1.0, batch_size=32, logger=None):
    column_names = dataset.column_names
    key_val_sep = "is" # Default for 'great'
    
    bos_token = "[BOS]"
    eoc_token = "[EOC]"
    prompts = [bos_token] * num_samples
    generated_data = {col: [] for col in column_names}
    
    logger.info(f"Generating {num_samples} samples...")
    
    for col_name in tqdm(column_names, desc="Generating columns"):
        col_meta = metadata.get(col_name.lower())
        if col_meta is None:
            logger.warning(f"Metadata missing for {col_name}, filling empty.")
            for _ in range(num_samples): generated_data[col_name].append("")
            continue
            
        col_type = col_meta["type"]
        all_values = []
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_prompts = prompts[batch_start:batch_end]
            
            try:
                batch_values = generate_column_batch(model, tokenizer, batch_prompts, col_name, col_type, col_meta, metadata, device, temperature, key_val_sep)
                all_values.extend(batch_values)
            except Exception as e:
                logger.error(f"Error generating {col_name} batch {batch_start}: {e}")
                # 只有真的发生 Crash 时才会走到这里，正常情况不应该发生了
                all_values.extend([""] * (batch_end - batch_start))
        
        # Update prompts
        type_token = {"numerical": "[NUM]", "categorical": "[CAT]", "mixed": "[MIX]"}.get(col_type, "[NUM]")
        for i, val in enumerate(all_values):
            prompts[i] += f" {type_token} {col_name} {key_val_sep} {val} {eoc_token}"
            generated_data[col_name].append(val)
            
    return pd.DataFrame(generated_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    # [修改] 默认 num_samples 为 None，稍后从 train set 获取
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples. Defaults to training set size.")
    
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_bins", type=int, default=100) # Ensure this matches Stage 2
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger()
    
    # Load Tokenizer & Dataset
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading training data from {args.train_data_path}")
    train_df = pd.read_csv(args.train_data_path)
    metadata = get_metadata(train_df)
    
    # [修复] 确定生成数量
    if args.num_samples is None:
        args.num_samples = len(train_df)
        logger.info(f"Num samples not specified. Defaulting to training set size: {args.num_samples}")
    else:
        logger.info(f"Generating specified number of samples: {args.num_samples}")
    
    # [修复] 确定 Vocab Size
    # 这里我们只算真实的 43，不再迁就 Stage 1
    dataset = LLMtgDataset.from_pandas(train_df)
    dataset.metadata = metadata
    real_cat_vocab_size = calculate_data_stats(dataset)
    logger.info(f"Real Cat Vocab Size: {real_cat_vocab_size}")
    
    # Load Stage 1 (with Key Deletion)
    model = load_stage1_model(
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        cat_vocab_size=real_cat_vocab_size, # Pass 43
        device=args.device,
        logger=logger
    )
    
    # Load Stage 2
    model = load_stage2_lora(model, args.lora_path, logger)
    
    # Generate
    syn_df = generate_synthetic_data(
        model=model, tokenizer=tokenizer, dataset=dataset, metadata=metadata,
        num_samples=args.num_samples, device=args.device, batch_size=args.batch_size, logger=logger
    )
    
    mkdir(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".")
    syn_df.to_csv(args.output_file, index=False)
    logger.info(f"Done. Saved to {args.output_file}")

if __name__ == "__main__":
    main()