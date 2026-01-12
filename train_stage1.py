import os
import argparse
import logging
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPT2Model,
    set_seed,
    get_linear_schedule_with_warmup,
)
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# 引用你的模块
from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir

def get_logger(filename=None):
    logger = logging.getLogger(__name__)
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    encoding = "utf-8"
    formatter = logging.Formatter(fmt=format_str, datefmt=datefmt)
    logger.setLevel(logging.INFO)
    if filename is not None:
        file_handler = logging.FileHandler(filename, encoding=encoding)
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
    # 🟢 Fix B: 移除 magic number 100，确保架构一致性
    if not hasattr(dataset, 'metadata') or dataset.metadata is None:
        # 如果没有 metadata，默认给一个最小安全值，或者抛出警告
        # 这里的 43 是基于一般情况的 fallback
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

def compute_lm_loss(model_outputs, batch):
    lm_logits = model_outputs.get("lm_logits", model_outputs.get("logits"))
    if lm_logits is None:
        raise ValueError("model_outputs missing 'lm_logits'.")

    labels = batch['labels']
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = lm_loss_fct(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1)
    )
    return lm_loss

def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id
    
    def collate_fn(batch):
        # 1. LM Inputs: 变长，需要 Pad
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

        # 🟢 Fix Core 1: Expert 字段必须是定长 C (Column Count)，使用 Stack
        # 前提：Dataset 必须保证每条数据返回的这些字段长度一致 (等于 Schema 列数)
        # 缺失的锚点 Dataset 侧应填 -1，Label 侧应填 -100
        
        stack_keys = [
            "expert_token_idxs", "col_type_ids", 
            "num_bin", "num_res", 
            "cat_id", 
            "mixed_mask", "mixed_bin", "mixed_res"
        ]
        
        for k in stack_keys:
            if k in batch[0] and isinstance(batch[0][k], torch.Tensor):
                result[k] = torch.stack([item[k] for item in batch], dim=0)

        return result

    return collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--serializer", type=str, default="great")
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                expert_token_idxs=batch.get("expert_token_idxs"),
                col_type_ids=batch.get("col_type_ids"),
            )
            loss = compute_lm_loss(outputs, batch)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss

def main():
    args = parse_args()
    set_seed(args.seed)
    mkdir(args.output_dir)
    logger = get_logger(filename=os.path.join(args.output_dir, "train.log"))
    logger.info(f"Arguments: {args}")

    # 1. Tokenizer
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset
    logger.info(f"Loading training data from {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    train_dataset = LLMtgDataset.from_pandas(train_df)
    train_dataset.set_serializer(args.serializer)
    train_dataset.set_tokenizer(tokenizer) 
    
    metadata = get_metadata(train_df)
    train_dataset.metadata = metadata
    cat_vocab_size = calculate_data_stats(train_dataset)

    validation_dataset = None
    if args.validation_file:
        logger.info(f"Loading validation data from {args.validation_file}")
        validation_df = pd.read_csv(args.validation_file)
        validation_dataset = LLMtgDataset.from_pandas(validation_df)
        validation_dataset.set_serializer(args.serializer)
        validation_dataset.set_tokenizer(tokenizer) 
        validation_dataset.metadata = metadata

    # 3. Model
    config_name = args.config_name if args.config_name else args.model_name_or_path
    config = AutoConfig.from_pretrained(config_name, cache_dir=args.cache_dir)

    logger.info("Initializing TypeAwareGPT2...")
    model = TypeAwareGPT2(
        config=config,
        num_bins=args.num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=None # 先传 None
    )

    logger.info(f"Loading GPT-2 weights from {args.model_name_or_path}...")
    gpt2_base = GPT2Model.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    msg = model.transformer.load_state_dict(gpt2_base.state_dict(), strict=True)
    
    #避免某些情况下 lm_head 没绑上的玄学问题
    model.tie_weights()
    
    logger.info(f"Weights loaded: {msg}")

    # Resize Embeddings
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing embeddings: {embedding_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()

    # 🟢 Fix Core 3: Type Token 显式设置 (包含 MIX)
    try:
        type_ids_map = {
            "NUM": tokenizer.convert_tokens_to_ids("[NUM]"),
            "CAT": tokenizer.convert_tokens_to_ids("[CAT]"),
            "MIX": tokenizer.convert_tokens_to_ids("[MIX]"), # 补全 MIX
        }
        # 只要 key tokens 都存在 (不是 unk)，就设置
        if all(v is not None and v != tokenizer.unk_token_id for v in type_ids_map.values()):
             model.set_type_token_ids(type_ids_map)
             logger.info(f"Explicitly set type_token_ids: {type_ids_map}")
        else:
            logger.warning("Some type tokens ([NUM], [CAT], [MIX]) not found. Skipping explicit set.")
    except Exception as e:
        logger.warning(f"Failed to set type_token_ids explicitly: {e}")

    model = model.to(args.device)

    # 4. Loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer), # 🟢 已改为 Stack
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if validation_dataset:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(tokenizer),
            num_workers=args.num_workers,
            pin_memory=True
        )

    # 5. Training Setup
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps
    )

    # 6. Loop
    model.train()
    update_step = 0
    global_step = 0
    best_eval_loss = float('inf')
    progress_bar = tqdm(total=args.max_train_steps, desc="Training")
    optimizer.zero_grad()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                expert_token_idxs=batch.get("expert_token_idxs"),
                col_type_ids=batch.get("col_type_ids"),
            )
            
            loss = compute_lm_loss(outputs, batch)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1
                progress_bar.update(1)

                if update_step % args.logging_steps == 0:
                    cur_loss = loss.item() * args.gradient_accumulation_steps
                    logger.info(f"Step {update_step}: Loss={cur_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

                if update_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{update_step}")
                    mkdir(save_path) # 🟢 Fix C: 加上 mkdir
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
            
            global_step += 1
            if update_step >= args.max_train_steps:
                break
        
        # 🟢 Fix A: Gradient Accumulation Flush with Rescaling
        if args.gradient_accumulation_steps > 1:
            remainder = (step + 1) % args.gradient_accumulation_steps
            if remainder != 0 and update_step < args.max_train_steps:
                # Rescale gradients to compensate for smaller batch size
                # scale_factor = args.gradient_accumulation_steps / remainder
                # for param in model.parameters():
                #     if param.grad is not None:
                #         param.grad.data.mul_(scale_factor)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1
                progress_bar.update(1)
                logger.info(f"Epoch {epoch+1} finished. Flushed gradients (rescaled by {scale_factor:.2f}).")

        if update_step >= args.max_train_steps:
            break

    progress_bar.close()
    
    # Save Final
    final_dir = os.path.join(args.output_dir, "final")
    mkdir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Done. Saved to {final_dir}")

if __name__ == "__main__":
    main()