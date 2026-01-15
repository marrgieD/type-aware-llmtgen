# ft_opacus0.py
# Stage2 ablation: No DP, No LoRA, full fine-tune from distilgpt2 with DDP (torch.distributed.run)
import os
import math
import argparse
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoConfig, set_seed

from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir


# -------------------------
# logging / ddp helpers
# -------------------------
def is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    return torch.distributed.get_rank() if is_dist() else 0

def get_world_size():
    return torch.distributed.get_world_size() if is_dist() else 1

def is_main():
    return get_rank() == 0

def setup_logger(out_dir):
    logger = logging.getLogger("ft_opacus0")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file (only main rank)
    if is_main():
        fh = logging.FileHandler(os.path.join(out_dir, "train.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------------------
# collate
# -------------------------
def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = padded_input_ids.clone()
        labels[padded_attention_mask == 0] = -100

        result = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": labels,
        }

        stack_keys = [
            "expert_token_idxs", "col_type_ids",
            "num_bin", "num_res",
            "cat_id",
            "mixed_mask", "mixed_bin", "mixed_res",
        ]
        for k in stack_keys:
            if k in batch[0] and isinstance(batch[0][k], torch.Tensor):
                result[k] = torch.stack([item[k] for item in batch], dim=0)

        return result

    return collate_fn


# -------------------------
# loss (same logic as your current)
# -------------------------
def compute_loss(model_outputs, batch, expert_loss_weight=1.0, lm_loss_weight=1.0):
    lm_logits = model_outputs.get("lm_logits", None)
    if lm_logits is None:
        raise ValueError("model_outputs missing 'lm_logits'")

    labels = batch["labels"]
    device = lm_logits.device

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = lm_loss_fct(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    lm_loss_item = float(lm_loss.detach().cpu())

    expert_outputs = model_outputs.get("expert_outputs", None)
    if expert_outputs is None:
        return lm_loss * lm_loss_weight, lm_loss_item, 0.0

    valid_cols = model_outputs.get("valid_mask", None)
    if valid_cols is None:
        col_positions = model_outputs.get("col_positions", None)
        if col_positions is None:
            raise ValueError("model_outputs missing both 'valid_mask' and 'col_positions'")
        valid_cols = (col_positions >= 0)
    valid_cols = valid_cols.bool()

    col_type_ids = batch.get("col_type_ids", None)
    if col_type_ids is None:
        raise ValueError("Batch missing 'col_type_ids'")
    col_type_ids = col_type_ids.long()

    total_expert_loss_sum = torch.zeros((), device=device)
    total_denom = torch.zeros((), device=device)

    ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    mse_none = nn.MSELoss(reduction="none")
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    # numeric
    if (batch.get("num_bin") is not None) and (batch.get("num_res") is not None):
        is_num = valid_cols & (col_type_ids == 0)
        if is_num.any():
            num_bin_logits = expert_outputs["num_bin_logits"]
            num_residual = expert_outputs["num_residual"]
            num_bin_labels = batch["num_bin"]
            num_res_labels = batch["num_res"]

            mask_bin = is_num & (num_bin_labels != -100)
            if mask_bin.any():
                K = num_bin_logits.size(-1)
                raw_bin = ce_none(num_bin_logits.reshape(-1, K), num_bin_labels.reshape(-1))
                m = mask_bin.reshape(-1).float()
                total_expert_loss_sum += (raw_bin * m).sum()
                total_denom += m.sum()

            mask_res = is_num & (num_res_labels != -100)
            if mask_res.any():
                raw_res = mse_none(num_residual.reshape(-1), num_res_labels.reshape(-1).float())
                m = mask_res.reshape(-1).float()
                total_expert_loss_sum += (raw_res * m).sum()
                total_denom += m.sum()

    # categorical
    if batch.get("cat_id") is not None:
        is_cat = valid_cols & (col_type_ids == 1)
        if is_cat.any():
            cat_logits = expert_outputs["cat_logits"]
            cat_id_labels = batch["cat_id"]
            mask_cat = is_cat & (cat_id_labels != -100)
            if mask_cat.any():
                V = cat_logits.size(-1)
                raw_cat = ce_none(cat_logits.reshape(-1, V), cat_id_labels.reshape(-1))
                m = mask_cat.reshape(-1).float()
                total_expert_loss_sum += (raw_cat * m).sum()
                total_denom += m.sum()

    # mixed
    if (batch.get("mixed_mask") is not None) and (batch.get("mixed_bin") is not None) and (batch.get("mixed_res") is not None):
        is_mixed = valid_cols & (col_type_ids == 2)
        if is_mixed.any():
            mixed_mask_logits = expert_outputs["mixed_mask_logits"]
            mixed_bin_logits = expert_outputs["mixed_bin_logits"]
            mixed_residual = expert_outputs["mixed_residual"]

            mixed_mask_labels = batch["mixed_mask"]
            mixed_bin_labels = batch["mixed_bin"]
            mixed_res_labels = batch["mixed_res"]

            # mask
            raw_mask = bce_none(mixed_mask_logits.reshape(-1), mixed_mask_labels.reshape(-1).float())
            m = is_mixed.reshape(-1).float()
            total_expert_loss_sum += (raw_mask * m).sum()
            total_denom += m.sum()

            # active branch
            active = is_mixed & (mixed_mask_labels > 0.5)

            mask_active_bin = active & (mixed_bin_labels != -100)
            if mask_active_bin.any():
                K = mixed_bin_logits.size(-1)
                raw = ce_none(mixed_bin_logits.reshape(-1, K), mixed_bin_labels.reshape(-1))
                m = mask_active_bin.reshape(-1).float()
                total_expert_loss_sum += (raw * m).sum()
                total_denom += m.sum()

            mask_active_res = active
            if mask_active_res.any():
                raw = mse_none(mixed_residual.reshape(-1), mixed_res_labels.reshape(-1).float())
                m = mask_active_res.reshape(-1).float()
                total_expert_loss_sum += (raw * m).sum()
                total_denom += m.sum()

    expert_loss = total_expert_loss_sum / (total_denom + 1e-8) if total_denom.item() > 0 else torch.zeros((), device=device)
    expert_loss_item = float(expert_loss.detach().cpu())

    total_loss = (lm_loss_weight * lm_loss) + (expert_loss_weight * expert_loss)
    return total_loss, lm_loss_item, expert_loss_item


# -------------------------
# save (rank0 only)
# -------------------------
def save_checkpoint(model, tokenizer, config, out_dir, step, logger):
    if not is_main():
        return
    ckpt_dir = os.path.join(out_dir, f"checkpoint-{step}")
    mkdir(ckpt_dir)

    # unwrap ddp
    m = model.module if hasattr(model, "module") else model

    # save state_dict (safest for custom module)
    torch.save(m.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    # also save config/tokenizer for reproducibility
    try:
        config.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    except Exception as e:
        logger.info(f"Warning: failed to save config/tokenizer: {e}")

    logger.info(f"Saved checkpoint to {ckpt_dir}")


def parse_args():
    p = argparse.ArgumentParser("ft_opacus0: noDP noLoRA fullFT distilgpt2 + experts (DDP)")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, default=None)  # not used heavily, kept for parity
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name_or_path", type=str, default="distilgpt2")
    p.add_argument("--cache_dir", type=str, default="./cache")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--num_warmup_steps", type=int, default=0)

    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    p.add_argument("--expert_loss_weight", type=float, default=1.0)
    p.add_argument("--lm_loss_weight", type=float, default=1.0)
    p.add_argument("--num_bins", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--target_col", type=str, default=None, help="move target col to first (Target-First)")

    # DDP local rank is injected by torchrun/torch.distributed.run
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return p.parse_args()


def main():
    args = parse_args()

    # init DDP
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    mkdir(args.output_dir)
    logger = setup_logger(args.output_dir)

    # seed (different per rank but deterministic)
    seed = args.seed + get_rank()
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if is_main():
        logger.info("Arguments:")
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")
        logger.info(f"DDP world_size={get_world_size()}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    import pandas as pd
    train_df = pd.read_csv(args.train_file)

    if args.target_col and (args.target_col in train_df.columns):
        if is_main():
            logger.info(f"Target-First: moving '{args.target_col}' to first column")
        cols = [args.target_col] + [c for c in train_df.columns if c != args.target_col]
        train_df = train_df[cols]
    # A. 把所有“列名”加进去 (防止列名变成 UNK)
    new_tokens = set()
    metadata = get_metadata(train_df)
    for col in train_df.columns:
        if str(col) not in tokenizer.vocab:
            new_tokens.add(str(col))
            
    # B. 把所有“分类值”加进去 (这是 UNK 泛滥的源头)
    for col_name, col_meta in metadata.items():
        if col_meta["type"] == "categorical":
            cats = col_meta["categories"]["unique"]
            for cat in cats:
                cat_str = str(cat)
                # 只有当 tokenizer 真的不认识它时才加
                if cat_str not in tokenizer.vocab:
                    new_tokens.add(cat_str)
    
    # C. 执行扩充 (如果有新词)
    new_tokens_list = list(new_tokens)
    if len(new_tokens_list) > 0:
        if is_main: logger.info(f"🚀 Found {len(new_tokens_list)} new tokens (e.g., {new_tokens_list[:5]}). Adding them to vocab...")
        tokenizer.add_tokens(new_tokens_list)

    train_dataset = LLMtgDataset.from_pandas(train_df)
    train_dataset.set_serializer("great")
    train_dataset.set_tokenizer(tokenizer)

    metadata = get_metadata(train_df)
    train_dataset.metadata = metadata

    # cat vocab size from metadata (same spirit as your script)
    max_cat_id = 0
    for col_name, col_meta in metadata.items():
        if col_meta.get("type") == "categorical":
            cats = col_meta.get("categories", {})
            vocab_size = cats.get("vocab_size", 0)
            if vocab_size > 0:
                max_cat_id = max(max_cat_id, vocab_size - 1)
    cat_vocab_size = max_cat_id + 1 if max_cat_id > 0 else 43

    if is_main():
        logger.info(f"cat_vocab_size = {cat_vocab_size}")

    # dataloader (DDP sampler)
    sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    # model
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = TypeAwareGPT2(
        config=config,
        num_bins=args.num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=train_dataset.type_token_ids if hasattr(train_dataset, "type_token_ids") else None,
    )
# ⚠️ 极其重要：词表扩充了，模型的 Embedding 层必须跟着变大！
    # 这一步绝对不能省，否则报错 index out of range
    if len(tokenizer) > model.config.vocab_size:
        if is_main: logger.info(f"Resizing model embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    if hasattr(train_dataset, 'type_token_ids'): model.set_type_token_ids(train_dataset.type_token_ids)
    # resize embeddings if tokenizer expanded
    emb_sz = model.get_input_embeddings().weight.shape[0]
    new_sz = len(tokenizer)

    # ✅ 所有 rank 都要 resize，保证参数形状一致
    if new_sz != emb_sz:
        if is_main():
            logger.info(f"Resizing embeddings {emb_sz} -> {new_sz}")
        model.resize_token_embeddings(new_sz)

    # （可选）保险：等大家都 resize 完再进 DDP
    if is_dist():
        torch.distributed.barrier()

    if hasattr(train_dataset, "type_token_ids") and train_dataset.type_token_ids:
        model.set_type_token_ids(train_dataset.type_token_ids)

    model.to(device)

    # DDP wrap
    if is_dist():
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,   # ✅ 关键
        )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # steps
    steps_per_epoch = len(train_loader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    if is_main():
        logger.info(f"Training: epochs={args.num_train_epochs}, max_steps={args.max_train_steps}, steps/epoch={steps_per_epoch}")

    model.train()
    global_step = 0

    for epoch in range(args.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            if "expert_token_idxs" not in batch:
                raise ValueError("Training batch missing 'expert_token_idxs' (anchor positions).")

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                expert_token_idxs=batch["expert_token_idxs"],
                col_type_ids=batch.get("col_type_ids"),
                output_expert_logits=True,
            )

            loss, lm_item, ex_item = compute_loss(
                outputs,
                batch,
                expert_loss_weight=args.expert_loss_weight,
                lm_loss_weight=args.lm_loss_weight,
            )

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if is_main() and (global_step % args.logging_steps == 0):
                logger.info(f"Step {global_step}: loss={loss.item()*args.gradient_accumulation_steps:.4f}, lm={lm_item:.4f}, expert={ex_item:.4f}")

            if global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, config, args.output_dir, global_step, logger)

        if global_step >= args.max_train_steps:
            break

    # final
    if is_main():
        final_dir = os.path.join(args.output_dir, "final")
        mkdir(final_dir)
        m = model.module if hasattr(model, "module") else model
        torch.save(m.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
        try:
            config.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
        except Exception as e:
            logger.info(f"Warning: failed to save config/tokenizer: {e}")
        logger.info(f"Saved final model to {final_dir}")

    if is_dist():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
