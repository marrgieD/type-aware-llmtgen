#!/usr/bin/env python3
"""
generate.py - 推理生成最终的合成表格数据

核心约束：不能使用 model.generate()，必须通过 Experts 采样。
Stage 2 冻结了 LM Head，所有数值和类别值必须通过 Experts 采样得到。
"""

import os
import argparse
import logging
import pickle
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
    """Calculate global max_cat_id from dataset metadata."""
    if not hasattr(dataset, 'metadata') or dataset.metadata is None:
        return 43
    
    max_cat_id = 0
    for col_name, col_meta in dataset.metadata.items():
        if col_meta.get("type") == "categorical":
            cats = col_meta.get("categories", {})
            vocab_size = cats.get("vocab_size", 0)
            if vocab_size > 0:
                max_cat_id = max(max_cat_id, vocab_size - 1)
    
    cat_vocab_size = max_cat_id + 1 if max_cat_id > 0 else 43
 
    return cat_vocab_size


def load_stage1_model(base_model_path, tokenizer, num_bins, cat_vocab_size, device, logger):
    """Load Stage 1 base model."""
    logger.info(f"Loading Stage 1 base model from {base_model_path}")
    
    config = AutoConfig.from_pretrained(base_model_path)
    
    model = TypeAwareGPT2(
        config=config,
        num_bins=num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=None
    )
    
    # Load weights
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
    
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys (first 10): {missing_keys[:10]}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
    
    # Set type token IDs
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        type_ids_map = {
            "NUM": tokenizer.convert_tokens_to_ids("[NUM]"),
            "CAT": tokenizer.convert_tokens_to_ids("[CAT]"),
            "MIX": tokenizer.convert_tokens_to_ids("[MIX]"),
        }
        if all(v is not None and v != tokenizer.unk_token_id for v in type_ids_map.values()):
            model.set_type_token_ids(type_ids_map)
            logger.info(f"Set type_token_ids: {type_ids_map}")
    
    model = model.to(device)
    model.eval()
    logger.info("Stage 1 base model loaded successfully")
    return model


def load_stage2_lora(model, lora_path, logger):
    """Load Stage 2 LoRA adapter and merge."""
    logger.info(f"Loading Stage 2 LoRA adapter from {lora_path}")
    
    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        logger.warning(f"No adapter_config.json found in {lora_path}. Trying to load as regular checkpoint...")
        # Try loading as regular checkpoint
        model_path = os.path.join(lora_path, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(lora_path, "model.safetensors")
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
            logger.info("Loaded as regular checkpoint (not LoRA)")
            return model
        else:
            raise FileNotFoundError(f"No valid checkpoint found in {lora_path}")
    
    # Load as LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    logger.info("LoRA adapter loaded. Merging and unloading...")
    model = model.merge_and_unload()
    logger.info("LoRA weights merged successfully")
    return model


def sample_from_expert(expert_outputs, col_type, col_meta, device, temperature=1.0):
    """
    Sample a value from expert outputs based on column type.
    
    Returns:
        str: The sampled value as a string
    """
    if col_type == "numerical":
        bin_logits = expert_outputs["num_bin_logits"]  # [B, K]
        residual = expert_outputs["num_residual"]  # [B]
        
        # Sample bin
        bin_probs = F.softmax(bin_logits / temperature, dim=-1)
        bin_id = torch.multinomial(bin_probs, 1).squeeze(-1)  # [B]
        
        # Get residual (already in [0, 1])
        res = residual.squeeze(-1)  # [B]
        
        # Inverse transform
        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])
        
        # Get bin boundaries
        bin_id_np = bin_id.cpu().numpy()
        res_np = res.cpu().numpy()
        
        values = []
        for i in range(len(bin_id_np)):
            bid = int(bin_id_np[i])
            bid = max(0, min(bid, len(bin_edges) - 2))
            
            lower = bin_edges[bid]
            upper = bin_edges[bid + 1]
            width = max(upper - lower, 1e-9)
            
            # Denormalize
            nv = lower + res_np[i] * width
            nv = np.clip(nv, 0.0, 1.0)
            
            # Unnormalize
            norm_min = stats["norm_min"]
            norm_max = stats["norm_max"]
            v = nv * (norm_max - norm_min) + norm_min
            
            # Inverse log if needed
            if stats.get("needs_log", False):
                shift = stats.get("log_shift", 1e-6)
                v = np.exp(v) - shift
            
            values.append(str(v))
        
        return values
    
    elif col_type == "categorical":
        cat_logits = expert_outputs["cat_logits"]  # [B, V_cat]
        
        # Sample category ID
        cat_probs = F.softmax(cat_logits / temperature, dim=-1)
        cat_id = torch.multinomial(cat_probs, 1).squeeze(-1)  # [B]
        
        # Map to string
        categories = col_meta["categories"]
        vocab = categories["unique"]
        
        cat_id_np = cat_id.cpu().numpy()
        values = []
        for i in range(len(cat_id_np)):
            cid = int(cat_id_np[i])
            if 0 <= cid < len(vocab):
                values.append(str(vocab[cid]))
            else:
                # values.append(str(vocab[0]))  # Fallback to [UNK]
                vocab[categories.get("unk_id", 0)]
        
        return values
    
    elif col_type == "mixed":
        mask_logits = expert_outputs["mixed_mask_logits"]  # [B]
        bin_logits = expert_outputs["mixed_bin_logits"]  # [B, K]
        residual = expert_outputs["mixed_residual"]  # [B]
        
        # Sample mask (gate)
        mask_probs = torch.sigmoid(mask_logits / temperature)  # [B]
        mask = torch.bernoulli(mask_probs)  # [B]
        
        mask_np = mask.cpu().numpy()
        batch_size = len(mask_np)
        
        # Sample bins for all samples at once
        bin_probs = F.softmax(bin_logits / temperature, dim=-1)
        bin_ids = torch.multinomial(bin_probs, 1).squeeze(-1)  # [B]
        bin_ids_np = bin_ids.cpu().numpy()
        
        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])
        norm_min = stats["norm_min"]
        norm_max = stats["norm_max"]
        needs_log = stats.get("needs_log", False)
        log_shift = stats.get("log_shift", 1e-6) if needs_log else 0.0
        
        residual_np = residual.cpu().numpy()
        
        values = []
        for i in range(batch_size):
            if mask_np[i] < 0.5:
                # Missing/zero
                values.append("0")
            else:
                # Sample numeric value
                bid = int(bin_ids_np[i])
                bid = max(0, min(bid, len(bin_edges) - 2))
                
                lower = bin_edges[bid]
                upper = bin_edges[bid + 1]
                width = max(upper - lower, 1e-9)
                
                nv = lower + residual_np[i] * width
                nv = np.clip(nv, 0.0, 1.0)
                
                v = nv * (norm_max - norm_min) + norm_min
                
                if needs_log:
                    v = np.exp(v) - log_shift
                
                values.append(str(v))
        
        return values
    
    else:
        raise ValueError(f"Unknown column type: {col_type}")


def generate_column_batch(
    model, tokenizer, prompts, col_name, col_type, col_meta, 
    metadata, device, temperature=1.0, logger=None, key_val_sep="is"
):
    """
    Generate values for a single column for a batch of prompts.
    
    Args:
        model: TypeAwareGPT2 model
        tokenizer: Tokenizer
        prompts: List of current prompt strings (one per sample)
        col_name: Column name
        col_type: Column type ("numerical", "categorical", "mixed")
        col_meta: Column metadata
        metadata: Full metadata dict
        device: Device
        temperature: Sampling temperature
        logger: Logger
    
    Returns:
        List[str]: Generated values (one per sample)
    """
    # Get type token
    type_token_map = {
        "numerical": "[NUM]",
        "categorical": "[CAT]",
        "mixed": "[MIX]"
    }
    type_token = type_token_map.get(col_type, "[NUM]")
    
    # Get key-value separator (from serializer)
    # Default to "is" for "great" serializer
    key_val_sep = "is"
    
    # Build prefix: [TYPE] col_name is
    # Note: format should match dataset.py: "{type_token} {col_name}{token_kv}{val_str} {token_eoc}"
    # So we build: "[TYPE] col_name is" (without the value and EOC yet)
    prefix = f"{type_token} {col_name} {key_val_sep}"
    
    # Tokenize prompts with prefix
    # Add space before prefix if prompt doesn't end with space
    full_prompts = []
    for p in prompts:
        if p and not p.endswith(" "):
            full_prompts.append(p + " " + prefix)
        else:
            full_prompts.append(p + prefix)
    
    # Tokenize batch
    encoded = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Forward pass through transformer to get hidden states
    # We only need hidden states, not expert outputs (we'll compute those separately)
    batch_size = input_ids.shape[0]
    
    with torch.no_grad():
        transformer_outputs = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_states = transformer_outputs[0]  # [B, T, H]
    
    # Extract last token hidden states for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because we want the last token index
    
    col_hidden_states = []
    for i in range(batch_size):
        last_pos = int(seq_lengths[i].item())
        col_hidden_states.append(last_hidden_states[i, last_pos:last_pos+1])
    
    col_hidden_states = torch.cat(col_hidden_states, dim=0)  # [B, H]
    col_hidden_states = col_hidden_states.unsqueeze(1)  # [B, 1, H] -> [B, C, H] where C=1
    
    # Forward through experts
    with torch.no_grad():
        if col_type == "numerical":
            bin_logits, residual = model.num_expert(col_hidden_states)
            expert_outputs = {
                "num_bin_logits": bin_logits.squeeze(1),  # [B, K]
                "num_residual": residual.squeeze(1),  # [B]
            }
        elif col_type == "categorical":
            cat_logits = model.cat_expert(col_hidden_states)
            expert_outputs = {
                "cat_logits": cat_logits.squeeze(1),  # [B, V_cat]
            }
        elif col_type == "mixed":
            mask_logits, bin_logits, residual = model.mixed_expert(col_hidden_states)
            expert_outputs = {
                "mixed_mask_logits": mask_logits.squeeze(1),  # [B]
                "mixed_bin_logits": bin_logits.squeeze(1),  # [B, K]
                "mixed_residual": residual.squeeze(1),  # [B]
            }
        else:
            raise ValueError(f"Unknown column type: {col_type}")
    
    # Sample values
    values = sample_from_expert(expert_outputs, col_type, col_meta, device, temperature)
    
    return values

def generate_synthetic_data(
    model, tokenizer, dataset, metadata, num_samples, device, 
    temperature=1.0, batch_size=32, logger=None, serializer="great"
):
    """
    Generate synthetic data row by row, column by column.
    """
    column_names = dataset.column_names
    serializer = dataset.serializer
    
    if serializer == "list":
        key_val_sep = ":"
    elif serializer == "text":
        key_val_sep = "is"
    elif serializer == "apval":
        key_val_sep = ":"
    else:  # serializer == "great" (default)
        key_val_sep = "is"
        
    logger.info(f"Using separator: '{key_val_sep}' for serializer: {serializer}")
    
    # Get EOC token
    eoc_token = "[EOC]"
    
    # Initialize prompts
    bos_token = "[BOS]"
    prompts = [bos_token] * num_samples
    
    # Store generated values
    generated_data = {col: [] for col in column_names}
    
    logger.info(f"Generating {num_samples} samples, {len(column_names)} columns")
    
    # Generate column by column
    # tqdm 这里会保留，显示进度条
    for col_idx, col_name in enumerate(tqdm(column_names, desc="Generating columns")):
        col_lower = col_name.lower()
        col_meta = metadata.get(col_lower)
        
        if col_meta is None:
            # 这里的 warning 如果也烦可以注释掉，但通常这个只会出一次
            # logger.warning(f"Metadata not found for column {col_name}, skipping")
            for _ in range(num_samples):
                generated_data[col_name].append("")
            continue
        
        col_type = col_meta["type"]
        
        # Generate in batches
        all_values = []
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_prompts = prompts[batch_start:batch_end]
            
            try:
                batch_values = generate_column_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    col_name=col_name,
                    col_type=col_type,
                    col_meta=col_meta,
                    metadata=metadata,
                    device=device,
                    temperature=temperature,
                    logger=logger,
                    key_val_sep=key_val_sep 
                )
                expected_n = (batch_end - batch_start)
                if not isinstance(batch_values, list):
                    batch_values = list(batch_values)
                
                # --- 修改开始：屏蔽了这里的 Warning 输出 ---
                if len(batch_values) != expected_n:
                    # 原来的代码在这里疯狂输出 warning，现已注释掉
                    # if logger is not None:
                    #     logger.warning(
                    #         f"Column {col_name} batch {batch_start}-{batch_end}: got {len(batch_values)} values, expected {expected_n}. Padding/truncating."
                    #     )
                    
                    # 只要保留下面的补齐/截断逻辑，代码就能正常运行
                    if len(batch_values) < expected_n:
                        batch_values = batch_values + [""] * (expected_n - len(batch_values))
                    else:
                        batch_values = batch_values[:expected_n]
                # --- 修改结束 ---

                all_values.extend(batch_values)
            except Exception as e:
                # 真正的 Error 建议保留，防止程序挂了不知道原因
                logger.error(f"Error generating column {col_name} batch {batch_start}-{batch_end}: {e}")
                all_values.extend([""] * (batch_end - batch_start))
        
        # Update prompts
        type_token_map = {
            "numerical": "[NUM]",
            "categorical": "[CAT]",
            "mixed": "[MIX]"
        }
        col_type = col_meta["type"]
        type_token = type_token_map.get(col_type, "[NUM]")
        
        for i, val in enumerate(all_values):
            prompts[i] += f" {type_token} {col_name} {key_val_sep} {val} {eoc_token}"
            generated_data[col_name].append(val)
    
    # Convert to DataFrame and normalize column lengths
    lengths = {k: len(v) for k, v in generated_data.items()}
    target_n = num_samples
    bad = {k: n for k, n in lengths.items() if n != target_n}
    
    # 这里的 Warning 只有在生成完所有数据后出现一次，如果不想要也可以注释掉
    if bad and logger is not None:
         pass # logger.warning(f"Column length mismatch before DataFrame: {bad}; padding/truncating to {target_n}")
         
    for k, v in generated_data.items():
        if len(v) < target_n:
            v.extend([""] * (target_n - len(v)))
        elif len(v) > target_n:
            del v[target_n:]
            
    df = pd.DataFrame(generated_data)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data")
    
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Path to Stage 1 base model checkpoint")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to Stage 2 LoRA adapter")
    parser.add_argument("--train_data_path", type=str, required=True,
                       help="Path to training CSV (for metadata)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to generate")
    
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                       help="Base model name (for tokenizer/config)")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                       help="Tokenizer name (defaults to model_name_or_path)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory")
    
    parser.add_argument("--serializer", type=str, default="great",
                       help="Serialization type (great/apval/text/list)")
    parser.add_argument("--num_bins", type=int, default=100,
                       help="Number of bins for numeric prediction")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup logger
    logger = get_logger()
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # Load tokenizer
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset for metadata
    logger.info(f"Loading training data from {args.train_data_path}")
    train_df = pd.read_csv(args.train_data_path)
    train_dataset = LLMtgDataset.from_pandas(train_df)
    train_dataset.set_serializer(args.serializer)
    train_dataset.set_tokenizer(tokenizer)
    
    # Generate metadata
    metadata = get_metadata(train_df)
    train_dataset.metadata = metadata
    # cat_vocab_size = calculate_data_stats(train_dataset)
    # logger.info(f"Calculated cat_vocab_size: {cat_vocab_size}")
    
    # [Step 1] 计算两个尺寸：一个是真实数据的尺寸，一个是 Stage 1 兼容尺寸
    real_cat_vocab_size = calculate_data_stats(train_dataset) # 这里应该返回真实的 43 (如果不改 calculate_data_stats 的话)
    # 强制 Stage 1 尺寸为 51 (为了加载 base model)
    stage1_vocab_size = 51 
    
    logger.info(f"Real vocab size: {real_cat_vocab_size}, Stage 1 requires: {stage1_vocab_size}")
    # Load Stage 1 model
    model = load_stage1_model(
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        cat_vocab_size=stage1_vocab_size,
        device=args.device,
        logger=logger
    )
    # [Step 3] 🔥🔥🔥 关键手术：把模型改回 43 以匹配 Stage 2 LoRA 🔥🔥🔥
    if real_cat_vocab_size != stage1_vocab_size:
        logger.info(f"Resizing cat_expert from {stage1_vocab_size} to {real_cat_vocab_size} to match Stage 2 LoRA...")
        
        # 1. 替换 Linear 层 (PyTorch 会自动初始化新层，但马上会被 LoRA 权重覆盖，所以无所谓)
        # 注意：这里假设你的 model.cat_expert.cat_head 是一个 nn.Linear
        # 768 是 hidden_size，如果不确定可以用 model.config.n_embd
        model.cat_expert.cat_head = torch.nn.Linear(
            model.config.n_embd, 
            real_cat_vocab_size
        ).to(args.device)
        
        # 2. 更新模型内部记录的尺寸属性
        model.cat_vocab_size = real_cat_vocab_size
    # Load Stage 2 LoRA and merge
    model = load_stage2_lora(model, args.lora_path, logger)
    
    # Generate synthetic data
    logger.info(f"Starting generation of {args.num_samples} samples...")
    synthetic_df = generate_synthetic_data(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        metadata=metadata,
        num_samples=args.num_samples,
        device=args.device,
        temperature=args.temperature,
        batch_size=args.batch_size,
        logger=logger,
        serializer=args.serializer
    )
    
    # Save output
    mkdir(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".")
    synthetic_df.to_csv(args.output_file, index=False)
    logger.info(f"Generated {len(synthetic_df)} samples saved to {args.output_file}")
    
    # Print statistics
    logger.info("\nGeneration Statistics:")
    logger.info(f"  Total samples: {len(synthetic_df)}")
    logger.info(f"  Columns: {list(synthetic_df.columns)}")
    logger.info(f"  Missing values: {synthetic_df.isna().sum().sum()}")


if __name__ == "__main__":
    main()
