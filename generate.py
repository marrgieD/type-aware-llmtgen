#!/usr/bin/env python3
"""
generate.py - (DDP Version)
支持多卡并行生成，解决超时问题。
"""
import sys
import os

# 获取当前用户的 site-packages 路径 (根据你的报错路径推断)
# 这一步是为了让 Python 优先找你自己装的包，而不是系统的包
user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)
else:
    # 如果已经在里面，把它挪到第一个
    sys.path.remove(user_site)
    sys.path.insert(0, user_site)

import os
import argparse
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
)
from peft import PeftModel
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence
# 假设 utils 在当前目录下
from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir


def get_logger(filename=None, verbosity=1):
    logger = logging.getLogger(__name__)
    logger.propagate = False  # 防止重复打印
    
    # 只有主进程打印日志
    if verbosity > 0:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        datefmt = "%m/%d/%Y %I:%M:%S %p"
        formatter = logging.Formatter(fmt=format_str, datefmt=datefmt)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            if filename is not None:
                file_handler = logging.FileHandler(filename, encoding="utf-8")
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.ERROR) # 非主进程只报严重错误
        
    return logger


def calculate_data_stats(dataset):
    if not hasattr(dataset, 'metadata') or dataset.metadata is None:
        return 43
    
    max_cat_id = 0
    for col_name, col_meta in dataset.metadata.items():
        if col_meta.get("type") == "categorical":
            cats = col_meta.get("categories", {})
            vocab_size = cats.get("vocab_size", 0)
            if vocab_size > 0:
                max_cat_id = max(max_cat_id, vocab_size - 1)
    return max_cat_id + 1 if max_cat_id > 0 else 43


def load_stage1_model(base_model_path, tokenizer, num_bins, cat_vocab_size, device, logger):
    logger.info(f"Loading Stage 1 base model from {base_model_path}")
    
    config = AutoConfig.from_pretrained(base_model_path)
    model = TypeAwareGPT2(
        config=config,
        num_bins=num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=None
    )
    
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
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    
    current_model_dict = model.state_dict()
    keys_to_remove = []
    
    for k, v in ckpt.items():
        if k in current_model_dict:
            if v.shape != current_model_dict[k].shape:
                if logger.isEnabledFor(logging.INFO): # 只在主进程警告
                    logger.warning(f"Shape mismatch for {k}: {v.shape} vs {current_model_dict[k].shape}. Dropping.")
                keys_to_remove.append(k)
    
    for k in keys_to_remove:
        del ckpt[k]
    
    model.load_state_dict(ckpt, strict=False)
    
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
    logger.info(f"Loading Stage 2 LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    return model


def sample_from_expert(expert_outputs, col_name, col_type, col_meta, device, temperature=1.0):
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
            
            if col_lower in INT_COLUMNS:
                v = int(round(v))
                
            values.append(str(v))
        return values
    
    # elif col_type == "categorical":
    #     cat_logits = expert_outputs["cat_logits"]
    #     cat_probs = F.softmax(cat_logits / temperature, dim=-1)
    #     cat_id = torch.multinomial(cat_probs, 1).squeeze(-1)
        
    #     categories = col_meta["categories"]
    #     vocab = categories["unique"]
    #     unk_id = categories.get("unk_id", 0) 
        
    #     cat_id_np = cat_id.cpu().numpy()
    #     values = []
    #     for i in range(len(cat_id_np)):
    #         cid = int(cat_id_np[i])
    #         if 0 <= cid < len(vocab):
    #             values.append(str(vocab[cid]))
    #         else:
    #             values.append(str(vocab[unk_id]))
    #     return values
    elif col_type == "categorical":
        cat_logits = expert_outputs["cat_logits"]
        
        # 获取当前列的类别信息
        categories = col_meta["categories"]
        vocab = categories["unique"]
        vocab_len = len(vocab) # 比如 Income=2

        # ================= 🔥🔥🔥 Logit Masking 修复 🔥🔥🔥 =================
        # 强制屏蔽非法类别索引。如果模型输出维度 (43) 大于当前列实际类别数 (2)，
        # 将 index >= 2 的位置设为 -inf，防止采样到非法值导致 [UNK]/NaN
        if vocab_len < cat_logits.size(-1):
            cat_logits[:, vocab_len:] = -float('inf')
        # ====================================================================

        cat_probs = F.softmax(cat_logits / temperature, dim=-1)
        cat_id = torch.multinomial(cat_probs, 1).squeeze(-1)
        
        unk_id = categories.get("unk_id", 0) 
        
        cat_id_np = cat_id.cpu().numpy()
        values = []
        for i in range(len(cat_id_np)):
            cid = int(cat_id_np[i])
            if 0 <= cid < len(vocab):
                values.append(str(vocab[cid]))
            else:
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
                
                if col_lower in INT_COLUMNS:
                    v = int(round(v))
                    
                values.append(str(v))
        return values

# def generate_column_batch(model, tokenizer, prompts, col_name, col_type, col_meta, metadata, device, temperature=1.0, key_val_sep="is"):
#     type_token_map = {"numerical": "[NUM]", "categorical": "[CAT]", "mixed": "[MIX]"}
#     type_token = type_token_map.get(col_type, "[NUM]")
#     prefix = f"{type_token} {col_name} {key_val_sep}"
    
#     # 构造 Prompt
#     full_prompts = [(p + " " + prefix) if p and not p.endswith(" ") else (p + prefix) for p in prompts]
    
#     encoded = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     input_ids = encoded["input_ids"].to(device)
#     attention_mask = encoded["attention_mask"].to(device)
    
#     # ================= 🚀 核心修复：把 no_grad 范围扩大到最后！ =================
#     with torch.no_grad():
#         # 1. Backbone 计算
#         transformer_outputs = model.transformer(
#             input_ids=input_ids, 
#             attention_mask=attention_mask
#         )
#         last_hidden_states = transformer_outputs[0]

#         # 2. 提取 Hidden State (保留高级索引优化)
#         batch_size = last_hidden_states.size(0)
#         last_token_indices = attention_mask.sum(dim=1) - 1
#         col_hidden_states = last_hidden_states[torch.arange(batch_size, device=device), last_token_indices]
#         col_hidden_states = col_hidden_states.unsqueeze(1) # [Batch, 1, Hidden]

#         # 3. Expert 计算 (必须在 no_grad 里面！)
#         # 上次就是这里漏在外面了，导致 output 带上了梯度
#         expert_outputs = {}
        
#         if col_type == "numerical":
#             bin_logits, residual = model.num_expert(col_hidden_states)
#             expert_outputs = {"num_bin_logits": bin_logits.squeeze(1), "num_residual": residual.squeeze(1)}
#         elif col_type == "categorical":
#             cat_logits = model.cat_expert(col_hidden_states)
#             expert_outputs = {"cat_logits": cat_logits.squeeze(1)}
#         elif col_type == "mixed":
#             mask_logits, bin_logits, residual = model.mixed_expert(col_hidden_states)
#             expert_outputs = {"mixed_mask_logits": mask_logits.squeeze(1), "mixed_bin_logits": bin_logits.squeeze(1), "mixed_residual": residual.squeeze(1)}
    
#     # 离开 no_grad 块时，expert_outputs 里的 tensor 已经是 clean 的（无梯度），可以放心转 numpy
#     return sample_from_expert(expert_outputs, col_name, col_type, col_meta, device, temperature)


def generate_column_batch(
    model, tokenizer, prompts, col_name, col_type, col_meta, metadata, 
    device, temperature=1.0, key_val_sep="is",
    max_context_tokens=512 # 加上这个防止显存爆炸
):
    model.eval()
    
    # 1. 准备前缀: [CAT] col_name is
    type_token_map = {"numerical": "[NUM]", "categorical": "[CAT]", "mixed": "[MIX]"}
    type_token = type_token_map.get(col_type, "[NUM]")
    prefix = f"{type_token} {col_name} {key_val_sep}"
    
    # 2. 构造 Prompt 并 Tokenize
    # 这里我们只做单纯的 Expert 推理，所以只需要把 prompt + prefix 喂进去拿到 hidden state 即可
    seqs = []
    for p in prompts:
        p = p or ""
        # 拼接逻辑：保证 prompt 和 prefix 之间有空格
        full_text = (p + " " + prefix) if (p and not p.endswith(" ")) else (p + prefix)
        
        ids = tokenizer.encode(full_text, add_special_tokens=False)
        # 简单的左侧截断，防止超长
        if len(ids) > max_context_tokens:
            ids = ids[-max_context_tokens:]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    
    # Padding
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # 3. 核心计算 (全程 no_grad，速度最快)
    with torch.no_grad():
        # A. 跑一遍模型拿到 Hidden States
        # 假设你的 model forward 返回 dict 包含 "hidden_states"
        # 如果是 HuggingFace 原生模型，通常是 outputs.last_hidden_state
        out = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_expert_logits=False 
        )
        
        # 兼容一下：有的模型实现返回是 tuple，有的是 dict
        if isinstance(out, dict):
            hidden_states = out.get("hidden_states", out.get("last_hidden_state"))
        else:
            hidden_states = out[0] # tuple

        # B. 提取最后一个 Token 的向量 (Expert Head 只需要看最后一个 token)
        batch_size = hidden_states.size(0)
        # 找到每个样本非 padding 的最后一个位置
        last_token_indices = attention_mask.sum(dim=1) - 1
        col_hidden_states = hidden_states[torch.arange(batch_size, device=device), last_token_indices]
        col_hidden_states = col_hidden_states.unsqueeze(1) # [Batch, 1, Hidden]

        # C. 丢给对应的 Expert Head
        expert_outputs = {}
        
        if col_type == "numerical":
            bin_logits, residual = model.num_expert(col_hidden_states)
            expert_outputs = {
                "num_bin_logits": bin_logits.squeeze(1),
                "num_residual": residual.squeeze(1)
            }
            
        elif col_type == "categorical":
            # 直接算 logits，不管它原本是不是想用 LM，现在强制走分类头
            cat_logits = model.cat_expert(col_hidden_states)
            expert_outputs = {
                "cat_logits": cat_logits.squeeze(1)
            }
            
        elif col_type == "mixed":
            mask_logits, bin_logits, residual = model.mixed_expert(col_hidden_states)
            expert_outputs = {
                "mixed_mask_logits": mask_logits.squeeze(1),
                "mixed_bin_logits": bin_logits.squeeze(1),
                "mixed_residual": residual.squeeze(1)
            }

    # 4. 采样并返回
    # 离开 no_grad 块，数据已经是 clean 的 tensor
    return sample_from_expert(expert_outputs, col_name, col_type, col_meta, device, temperature)

def generate_synthetic_data(model, tokenizer, dataset, metadata, num_samples, device, temperature=0.7, batch_size=32, logger=None, show_progress=True):
    column_names = dataset.column_names
    key_val_sep = "is" 
    
    bos_token = "[BOS]"
    eoc_token = "[EOC]"
    prompts = [bos_token] * num_samples
    generated_data = {col: [] for col in column_names}
    
    if show_progress:
        logger.info(f"Starting generation for {num_samples} samples...")
        iterator = tqdm(column_names, desc="Generating columns")
    else:
        iterator = column_names
    
    for col_name in iterator:
        col_meta = metadata.get(col_name.lower())
        if col_meta is None:
            if show_progress: logger.warning(f"Metadata missing for {col_name}, filling empty.")
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
                if show_progress: logger.error(f"Error generating {col_name} batch {batch_start}: {e}")
                all_values.extend([""] * (batch_end - batch_start))
        
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
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--target_col", type=str, default=None, 
                       help="The target column name to move to the first position.")
    # DDP args
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # --- DDP Setup ---
    # 检查环境变量（torchrun 会自动设置 LOCAL_RANK 等）
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        # 使用 torchrun 启动
        args.local_rank = env_local_rank
        
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        is_main_process = (rank == 0)
    else:
        # 单卡模式兜底
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
    
    set_seed(args.seed + rank) # 确保每个进程种子略有不同（虽然生成时主要是模型决定，但安全起见）
    logger = get_logger(verbosity=1 if is_main_process else 0)
    
    # --- Data Loading ---
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {
        "additional_special_tokens": [
            "[NUM]", "[CAT]", "[MIX]", "[BOS]", "[EOS]", "[EOC]", "[UNK]",
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    if is_main_process:
        logger.info(f"Loading training data from {args.train_data_path}")
    train_df = pd.read_csv(args.train_data_path)
    # ================= 🚀 新增：Target-First 重排逻辑 =================
    if args.target_col:
        if args.target_col in train_df.columns:
            if is_main_process:
                logger.info(f"Applying Target-First strategy: Moving '{args.target_col}' to the first column.")
            cols = [args.target_col] + [c for c in train_df.columns if c != args.target_col]
            train_df = train_df[cols]
        elif is_main_process:
            logger.warning(f"Target column '{args.target_col}' not found in metadata file! Skipping reordering.")
    metadata = get_metadata(train_df)
    
    if args.num_samples is None:
        args.num_samples = len(train_df)
    
    if is_main_process:
        logger.info(f"Total target samples: {args.num_samples}")
        logger.info(f"World Size: {world_size}")

    # --- Split Workload ---
    # 计算当前进程需要生成的样本数
    samples_per_gpu = args.num_samples // world_size
    remainder = args.num_samples % world_size
    
    if rank < remainder:
        my_num_samples = samples_per_gpu + 1
    else:
        my_num_samples = samples_per_gpu
        
    logger.info(f"Rank {rank} generating {my_num_samples} samples...")

    # --- Model Loading ---
    dataset = LLMtgDataset.from_pandas(train_df)
    dataset.metadata = metadata
    real_cat_vocab_size = calculate_data_stats(dataset)
    
    model = load_stage1_model(
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        cat_vocab_size=real_cat_vocab_size,
        device=device,
        logger=logger
    )
    model = load_stage2_lora(model, args.lora_path, logger)
    
    # --- Generation ---
    syn_df = generate_synthetic_data(
        model=model, tokenizer=tokenizer, dataset=dataset, metadata=metadata,
        num_samples=my_num_samples, device=device, batch_size=args.batch_size, 
        logger=logger, show_progress=is_main_process
    )
    
    # --- Save Partial Results ---
    output_dir = os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else "."
    mkdir(output_dir)
    
    partial_output_file = f"{args.output_file}.part{rank}"
    syn_df.to_csv(partial_output_file, index=False)
    logger.info(f"Rank {rank} saved to {partial_output_file}")
    
    # 等待所有进程保存完毕
    if world_size > 1:
        dist.barrier()
    
    # --- Merge Results (Only Rank 0) ---
    if is_main_process:
        logger.info("Merging partial files...")
        all_dfs = []
        # 按顺序读取 part0, part1...
        for r in range(world_size):
            fname = f"{args.output_file}.part{r}"
            if os.path.exists(fname):
                all_dfs.append(pd.read_csv(fname))
            else:
                logger.error(f"Missing file: {fname}")
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(args.output_file, index=False)
            logger.info(f"Successfully merged {len(final_df)} samples to {args.output_file}")
            
            # 清理临时文件
            for r in range(world_size):
                try:
                    os.remove(f"{args.output_file}.part{r}")
                except:
                    pass
        else:
            logger.error("No data generated!")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()