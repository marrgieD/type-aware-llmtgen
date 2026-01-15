#!/usr/bin/env python3
"""
generate_stage2_only.py
DDP 多卡生成：Stage2-only（无 Stage1、无 LoRA）
- 从 --model_path 目录加载完整模型（pytorch_model.bin 或 model.safetensors）
- 自动推断 checkpoint 里的 vocab size / cat head size，避免 shape mismatch
- 按 train_data_path 的列顺序 + metadata 类型，用 expert 逐列采样
"""

import os
import argparse
import logging
import glob

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig, set_seed
from safetensors import safe_open

from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir


# ----------------- logger -----------------
def get_logger(filename=None, verbosity=1):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    if verbosity > 0:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt=fmt)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            if filename is not None:
                fh = logging.FileHandler(filename, encoding="utf-8")
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
    else:
        logger.setLevel(logging.ERROR)
    return logger


# ----------------- checkpoint loading utils -----------------
def _load_state_dict_from_dir(model_dir: str):
    """
    支持：
      - model_dir/pytorch_model.bin
      - model_dir/model.safetensors
      - model_dir/*.bin (兜底)
    """
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    st_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(st_path):
        ckpt = {}
        with safe_open(st_path, framework="pt") as f:
            for k in f.keys():
                ckpt[k] = f.get_tensor(k)
        return ckpt

    if os.path.exists(bin_path):
        ckpt = torch.load(bin_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        return ckpt

    # 兜底：找任意 .bin
    candidates = glob.glob(os.path.join(model_dir, "*.bin"))
    if candidates:
        ckpt = torch.load(candidates[0], map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        return ckpt

    raise FileNotFoundError(f"No pytorch_model.bin / model.safetensors found in: {model_dir}")


def _infer_vocab_and_cat_size(ckpt: dict, fallback_cat_size: int):
    """
    从 ckpt 推断 embedding vocab size 和 cat head size，避免 DDP/加载 shape mismatch。
    """
    vocab_sz = None
    cat_sz = None

    # GPT-2 embedding 常见 key
    for k in ["transformer.wte.weight", "wte.weight"]:
        if k in ckpt:
            vocab_sz = ckpt[k].shape[0]
            break

    # 你的 TypeAwareGPT2 分类头
    for k in ["cat_expert.cat_head.weight", "cat_head.weight"]:
        if k in ckpt:
            cat_sz = ckpt[k].shape[0]
            break

    if cat_sz is None:
        cat_sz = fallback_cat_size

    return vocab_sz, cat_sz


def load_stage2_only_model(model_path: str, tokenizer, num_bins: int, fallback_cat_vocab_size: int, device, logger):
    logger.info(f"Loading Stage2-only model from: {model_path}")

    ckpt = _load_state_dict_from_dir(model_path)
    vocab_ckpt, cat_sz = _infer_vocab_and_cat_size(ckpt, fallback_cat_vocab_size)

    # config：优先用 model_path 里的（如果你 save_pretrained 过会有 config.json）
    try:
        config = AutoConfig.from_pretrained(model_path)
        logger.info("Loaded config from model_path.")
    except Exception:
        config = AutoConfig.from_pretrained("distilgpt2")
        logger.info("Loaded config from distilgpt2 (fallback).")

    # 用 ckpt 推断出来的 cat_sz 来建模，避免 cat head mismatch
    model = TypeAwareGPT2(
        config=config,
        num_bins=num_bins,
        cat_vocab_size=cat_sz,
        type_token_ids=None,
    )

    # vocab mismatch：以 ckpt 为准（最稳）
    if vocab_ckpt is not None:
        emb_sz = model.get_input_embeddings().weight.shape[0]
        if emb_sz != vocab_ckpt:
            logger.info(f"Resizing embeddings {emb_sz} -> {vocab_ckpt} (match checkpoint)")
            model.resize_token_embeddings(vocab_ckpt)

    # 去掉 shape 不一致的 key（如果你又改过结构）
    cur_sd = model.state_dict()
    drop = []
    for k, v in ckpt.items():
        if k in cur_sd and v.shape != cur_sd[k].shape:
            drop.append(k)
    if drop:
        logger.warning(f"Dropping {len(drop)} mismatched keys. Examples: {drop[:6]}")
        for k in drop:
            ckpt.pop(k, None)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        logger.warning(f"Missing keys (first 10): {missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected keys (first 10): {unexpected[:10]}")

    # type token ids
    type_ids_map = {
        "NUM": tokenizer.convert_tokens_to_ids("[NUM]"),
        "CAT": tokenizer.convert_tokens_to_ids("[CAT]"),
        "MIX": tokenizer.convert_tokens_to_ids("[MIX]"),
    }
    if all(v is not None and v != tokenizer.unk_token_id for v in type_ids_map.values()):
        model.set_type_token_ids(type_ids_map)

    model.to(device)
    model.eval()
    return model


# ----------------- sampling -----------------
def sample_from_expert(expert_outputs, col_name, col_type, col_meta, temperature=1.0):
    INT_COLUMNS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    col_lower = col_name.lower()

    if col_type == "numerical":
        bin_logits = expert_outputs["num_bin_logits"]
        residual = expert_outputs["num_residual"]

        bin_probs = F.softmax(bin_logits / max(temperature, 1e-6), dim=-1)
        bin_id = torch.multinomial(bin_probs, 1).squeeze(-1)
        res = residual.squeeze(-1)

        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])

        bin_id_np = bin_id.detach().cpu().numpy()
        res_np = res.detach().cpu().numpy()

        values = []
        for i in range(len(bin_id_np)):
            bid = int(bin_id_np[i])
            bid = max(0, min(bid, len(bin_edges) - 2))
            lower, upper = bin_edges[bid], bin_edges[bid + 1]
            width = max(float(upper - lower), 1e-9)

            nv = float(lower) + float(res_np[i]) * width
            nv = float(np.clip(nv, 0.0, 1.0))

            norm_min, norm_max = stats["norm_min"], stats["norm_max"]
            v = nv * (norm_max - norm_min) + norm_min

            if stats.get("needs_log", False):
                shift = stats.get("log_shift", 1e-6)
                v = float(np.exp(v) - shift)

            if col_lower in INT_COLUMNS:
                v = int(round(v))

            values.append(str(v))
        return values

    if col_type == "categorical":
        cat_logits = expert_outputs["cat_logits"]
        categories = col_meta["categories"]
        vocab = categories["unique"]
        vocab_len = len(vocab)

        cat_probs = F.softmax(cat_logits / max(temperature, 1e-6), dim=-1)
        cat_id = torch.multinomial(cat_probs, 1).squeeze(-1).detach().cpu().numpy()

        values = []
        for cid in cat_id:
            cid = int(cid)
            if 0 <= cid < vocab_len:
                values.append(str(vocab[cid]))
            else:
                values.append(str(vocab[0]))
        return values

    if col_type == "mixed":
        mask_logits = expert_outputs["mixed_mask_logits"]
        bin_logits = expert_outputs["mixed_bin_logits"]
        residual = expert_outputs["mixed_residual"]

        mask_prob = torch.sigmoid(mask_logits / max(temperature, 1e-6))
        mask = (torch.rand_like(mask_prob) < mask_prob).long()

        bin_probs = F.softmax(bin_logits / max(temperature, 1e-6), dim=-1)
        bin_id = torch.multinomial(bin_probs, 1).squeeze(-1)
        res = residual.squeeze(-1)

        stats = col_meta["stats"]
        bin_edges = np.array(stats["bin_edges"])

        mask_np = mask.detach().cpu().numpy()
        bin_np = bin_id.detach().cpu().numpy()
        res_np = res.detach().cpu().numpy()

        values = []
        for i in range(len(mask_np)):
            if int(mask_np[i]) == 0:
                values.append("0")
                continue
            bid = int(bin_np[i])
            bid = max(0, min(bid, len(bin_edges) - 2))
            lower, upper = bin_edges[bid], bin_edges[bid + 1]
            width = max(float(upper - lower), 1e-9)

            nv = float(lower) + float(res_np[i]) * width
            nv = float(np.clip(nv, 0.0, 1.0))

            norm_min, norm_max = stats["norm_min"], stats["norm_max"]
            v = nv * (norm_max - norm_min) + norm_min

            if stats.get("needs_log", False):
                shift = stats.get("log_shift", 1e-6)
                v = float(np.exp(v) - shift)

            values.append(str(v))
        return values

    return [""]

@torch.no_grad()
def generate_column_batch(model, tokenizer, batch_prompts, col_name, col_type, col_meta, metadata, device, temperature, key_val_sep):
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # ✅ 关键：不要调用 model(...)，直接走 transformer backbone
    # GPT2Model 的返回一般是 BaseModelOutputWithPastAndCrossAttentions
    backbone_out = model.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )

    hidden = backbone_out.last_hidden_state[:, -1:, :]  # (B,1,H)

    if col_type == "numerical":
        bin_logits, residual = model.num_expert(hidden)
        expert_outputs = {"num_bin_logits": bin_logits.squeeze(1), "num_residual": residual.squeeze(1)}

    elif col_type == "categorical":
        cat_logits = model.cat_expert(hidden)
        expert_outputs = {"cat_logits": cat_logits.squeeze(1)}

    elif col_type == "mixed":
        mask_logits, bin_logits, residual = model.mixed_expert(hidden)
        expert_outputs = {
            "mixed_mask_logits": mask_logits.squeeze(1),
            "mixed_bin_logits": bin_logits.squeeze(1),
            "mixed_residual": residual.squeeze(1),
        }
    else:
        expert_outputs = {}

    return sample_from_expert(expert_outputs, col_name, col_type, col_meta, temperature=temperature)


def generate_synthetic_data(model, tokenizer, dataset, metadata, num_samples, device, temperature=1.0, batch_size=32, logger=None, show_progress=True):
    column_names = dataset.column_names
    key_val_sep = "is"
    bos_token = "[BOS]"
    eoc_token = "[EOC]"

    prompts = [bos_token] * num_samples
    generated_data = {col: [] for col in column_names}

    iterator = tqdm(column_names, desc="Generating columns") if show_progress else column_names

    for col_name in iterator:
        col_meta = metadata.get(col_name.lower())
        if col_meta is None:
            if show_progress and logger:
                logger.warning(f"Metadata missing for {col_name}, filling empty.")
            generated_data[col_name].extend([""] * num_samples)
            continue

        col_type = col_meta["type"]
        all_values = []

        for st in range(0, num_samples, batch_size):
            ed = min(st + batch_size, num_samples)
            batch_prompts = prompts[st:ed]
            vals = generate_column_batch(
                model, tokenizer, batch_prompts,
                col_name, col_type, col_meta, metadata,
                device, temperature, key_val_sep
            )
            all_values.extend(vals)

        type_token = {"numerical": "[NUM]", "categorical": "[CAT]", "mixed": "[MIX]"}.get(col_type, "[NUM]")
        for i, val in enumerate(all_values):
            prompts[i] += f" {type_token} {col_name} {key_val_sep} {val} {eoc_token}"
            generated_data[col_name].append(val)

    return pd.DataFrame(generated_data)


# ----------------- args/main -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Stage2-only checkpoint dir (contains pytorch_model.bin/model.safetensors)")
    p.add_argument("--train_data_path", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--model_name_or_path", type=str, default="distilgpt2", help="tokenizer/config fallback")
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_bins", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    # DDP setup
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        is_main = (rank == 0)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    set_seed(args.seed + rank)
    logger = get_logger(verbosity=1 if is_main else 0)

    # tokenizer: 优先从 model_path 读（如果你保存过 tokenizer），否则 fallback
    tok_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if is_main:
            logger.info("Loaded tokenizer from model_path.")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if is_main:
            logger.info(f"Loaded tokenizer from {tok_name} (fallback).")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 保险：确保特殊 token 存在（不会破坏 ckpt，因为我们加载时以 ckpt vocab 为准）
    special = ["[BOS]", "[EOC]", "[NUM]", "[CAT]", "[MIX]"]
    add = []
    for t in special:
        if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
            add.append(t)
    if add:
        tokenizer.add_tokens(add, special_tokens=False)
        if is_main:
            logger.info(f"Added special tokens to tokenizer: {add}")

    # load real data for metadata/columns
    if is_main:
        logger.info(f"Loading training data: {args.train_data_path}")
    train_df = pd.read_csv(args.train_data_path)

    # target-first reorder (保持和训练一致)
    if args.target_col and args.target_col in train_df.columns:
        cols = [args.target_col] + [c for c in train_df.columns if c != args.target_col]
        train_df = train_df[cols]
        if is_main:
            logger.info(f"Target-First: moved '{args.target_col}' to first column.")

    metadata = get_metadata(train_df)
    dataset = LLMtgDataset.from_pandas(train_df)
    dataset.metadata = metadata

    if args.num_samples is None:
        args.num_samples = len(train_df)

    # split workload
    base = args.num_samples // world_size
    rem = args.num_samples % world_size
    my_n = base + (1 if rank < rem else 0)

    if is_main:
        logger.info(f"World size={world_size}, total={args.num_samples}")
    logger.info(f"Rank {rank} generating {my_n} samples on {device}")

    # fallback cat size（如果 ckpt 没存 cat head）
    # 这里不重要：真正建模会以 ckpt 推断出来的 cat size 为准
    fallback_cat_vocab_size = 43

    model = load_stage2_only_model(
        model_path=args.model_path,
        tokenizer=tokenizer,
        num_bins=args.num_bins,
        fallback_cat_vocab_size=fallback_cat_vocab_size,
        device=device,
        logger=logger,
    )

    syn_df = generate_synthetic_data(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        metadata=metadata,
        num_samples=my_n,
        device=device,
        temperature=args.temperature,
        batch_size=args.batch_size,
        logger=logger,
        show_progress=is_main,
    )

    # save partial
    out_dir = os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else "."
    mkdir(out_dir)
    part = f"{args.output_file}.part{rank}"
    syn_df.to_csv(part, index=False)
    logger.info(f"Rank {rank} wrote {part}")

    if world_size > 1:
        dist.barrier()

    if is_main:
        logger.info("Merging partial files...")
        all_dfs = []
        for r in range(world_size):
            fn = f"{args.output_file}.part{r}"
            if not os.path.exists(fn):
                logger.error(f"Missing part file: {fn}")
                continue
            all_dfs.append(pd.read_csv(fn))
        final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else syn_df
        final_df.to_csv(args.output_file, index=False)
        logger.info(f"Saved merged file: {args.output_file} (n={len(final_df)})")

        for r in range(world_size):
            fn = f"{args.output_file}.part{r}"
            try:
                os.remove(fn)
            except Exception:
                pass

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
