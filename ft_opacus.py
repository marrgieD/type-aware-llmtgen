import os
import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GPT2Config,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from safetensors import safe_open
from utils.models import TypeAwareGPT2
from utils.dataset import LLMtgDataset, get_metadata
from utils.misc import mkdir
# from utils.utils import str2bool
from torch.nn.utils.rnn import pad_sequence

def get_logger(filename=None):
    logger = logging.getLogger(__name__)
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    encoding = "utf-8"

    formatter = logging.Formatter(fmt=format_str, datefmt=datefmt)
    logger.setLevel(logging.DEBUG)

    if filename is not None:
        file_handler = logging.FileHandler(filename, encoding=encoding)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def calculate_data_stats(dataset):
    """Calculate global max_cat_id from dataset metadata to get cat_vocab_size."""
    if not hasattr(dataset, 'metadata') or dataset.metadata is None:
        raise ValueError("Dataset must have metadata attribute set.")
    
    max_cat_id = 0
    for col_name, col_meta in dataset.metadata.items():
        if col_meta.get("type") == "categorical":
            cats = col_meta.get("categories", {})
            vocab_size = cats.get("vocab_size", 0)
            if vocab_size > 0:
                # vocab_size is the total number of categories (including [UNK])
                # max_cat_id would be vocab_size - 1 (0-indexed)
                max_cat_id = max(max_cat_id, vocab_size - 1)
    
    cat_vocab_size = max_cat_id + 1 if max_cat_id > 0 else 43  # default fallback
    return cat_vocab_size


def compute_loss(model_outputs, batch, expert_loss_weight=1.0, lm_loss_weight=0.5, num_bins=None):
    """
    Compute hybrid loss: LM Loss + expert_loss_weight * Expert Loss.
    Hardened version: Uses reshape(), dynamic shape inference, and strict null checks.
    """
    # 1. LM Loss (Strict Check)
    lm_logits = model_outputs.get("lm_logits", None)
    if lm_logits is None:
        raise ValueError("model_outputs missing 'lm_logits'. Check model return dict.")

    labels = batch['labels']
    device = lm_logits.device
    
    # Shift predictions
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 动态获取 Vocab Size，不依赖配置
    vocab_size = shift_logits.size(-1)

    lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    # 🛡️ 改进：使用 reshape 替代 view，处理非 contiguous 情况
    lm_loss = lm_loss_fct(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1)
    )
    lm_loss_item = lm_loss.item()
    
    # 2. Expert Loss (Dynamic Denominator)
    expert_outputs = model_outputs.get('expert_outputs')
    if expert_outputs is None:
        return lm_loss * lm_loss_weight, lm_loss_item, 0.0

    # 🛡️ 改进：更严格的 Valid Mask / Position 检查
    valid_cols = model_outputs.get("valid_mask")
    if valid_cols is None:
        col_positions = model_outputs.get('col_positions')
        if col_positions is None:
            raise ValueError("model_outputs missing both 'valid_mask' and 'col_positions'.")
        valid_cols = (col_positions >= 0)

    col_type_ids = batch.get('col_type_ids') # 0=Num, 1=Cat, 2=Mixed
    if col_type_ids is None:
        raise ValueError("Batch missing 'col_type_ids'.")

    valid_cols = valid_cols.bool()
    col_type_ids = col_type_ids.long()

    # 初始化累加器
    total_expert_loss_sum = torch.zeros((), device=device)
    total_denom = torch.zeros((), device=device)

    # 预定义 Loss
    ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    mse_none = nn.MSELoss(reduction="none")
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    # -----------------------------------------------------------
    # Numeric (Bin + Residual)
    # -----------------------------------------------------------
    if (batch.get("num_bin") is not None) and (batch.get("num_res") is not None):
        is_num = valid_cols & (col_type_ids == 0)
        
        if is_num.any():
            num_bin_logits = expert_outputs["num_bin_logits"]
            num_residual = expert_outputs["num_residual"]
            num_bin_labels = batch["num_bin"]
            num_res_labels = batch["num_res"]

            # 1. Bin Loss
            mask_bin = is_num & (num_bin_labels != -100)
            if mask_bin.any():
                # 🛡️ 改进：动态获取 K，不依赖参数 num_bins
                K = num_bin_logits.size(-1)
                # 🛡️ 改进：reshape 替代 view
                raw_bin = ce_none(num_bin_logits.reshape(-1, K), num_bin_labels.reshape(-1))
                mask_bin_flat = mask_bin.reshape(-1).float()
                
                total_expert_loss_sum += (raw_bin * mask_bin_flat).sum()
                total_denom += mask_bin_flat.sum()

            # 2. Residual Loss
            mask_res = is_num & (num_res_labels != -100)
            if mask_res.any():
                raw_res = mse_none(num_residual.reshape(-1), num_res_labels.reshape(-1).float())
                mask_res_flat = mask_res.reshape(-1).float()
                
                total_expert_loss_sum += (raw_res * mask_res_flat).sum()
                total_denom += mask_res_flat.sum()

    # -----------------------------------------------------------
    # Categorical
    # -----------------------------------------------------------
    if batch.get("cat_id") is not None:
        is_cat = valid_cols & (col_type_ids == 1)
        
        if is_cat.any():
            cat_logits = expert_outputs["cat_logits"]
            cat_id_labels = batch["cat_id"]
            
            mask_cat = is_cat & (cat_id_labels != -100)
            if mask_cat.any():
                # 🛡️ 改进：动态获取 V_cat
                V_cat = cat_logits.size(-1)
                raw_cat = ce_none(cat_logits.reshape(-1, V_cat), cat_id_labels.reshape(-1))
                mask_cat_flat = mask_cat.reshape(-1).float()
                
                total_expert_loss_sum += (raw_cat * mask_cat_flat).sum()
                total_denom += mask_cat_flat.sum()

    # -----------------------------------------------------------
    # Mixed
    # -----------------------------------------------------------
    if (batch.get("mixed_mask") is not None) and \
       (batch.get("mixed_bin") is not None) and \
       (batch.get("mixed_res") is not None):
       
        is_mixed = valid_cols & (col_type_ids == 2)
        
        if is_mixed.any():
            mixed_mask_logits = expert_outputs["mixed_mask_logits"]
            mixed_bin_logits = expert_outputs["mixed_bin_logits"]
            mixed_residual = expert_outputs["mixed_residual"]
            
            mixed_mask_labels = batch["mixed_mask"]
            mixed_bin_labels = batch["mixed_bin"]
            mixed_res_labels = batch["mixed_res"]

            # 1. Mask Prediction
            mask_m = is_mixed 
            if mask_m.any():
                raw_mask = bce_none(mixed_mask_logits.reshape(-1), mixed_mask_labels.reshape(-1).float())
                mask_m_flat = mask_m.reshape(-1).float()
                
                total_expert_loss_sum += (raw_mask * mask_m_flat).sum()
                total_denom += mask_m_flat.sum()

            # 2. Value Prediction
            # 🛡️ 改进：显式防御，确保 mixed_mask_labels 是 0/1 且不是 -100
            # 假设 dataset 里 1.0 是 active, 0.0 是 missing/NaN
            active = is_mixed & (mixed_mask_labels > 0.5)
            
            # Mixed Bin
            mask_active_bin = active & (mixed_bin_labels != -100)
            if mask_active_bin.any():
                K_mix = mixed_bin_logits.size(-1)
                raw_mix_bin = ce_none(mixed_bin_logits.reshape(-1, K_mix), mixed_bin_labels.reshape(-1))
                mask_ab_flat = mask_active_bin.reshape(-1).float()
                
                total_expert_loss_sum += (raw_mix_bin * mask_ab_flat).sum()
                total_denom += mask_ab_flat.sum()
            
            # Mixed Residual
            mask_active_res = active 
            if mask_active_res.any():
                raw_mix_res = mse_none(mixed_residual.reshape(-1), mixed_res_labels.reshape(-1).float())
                mask_ar_flat = mask_active_res.reshape(-1).float()
                
                total_expert_loss_sum += (raw_mix_res * mask_ar_flat).sum()
                total_denom += mask_ar_flat.sum()

    # 3. Final Aggregation
    if total_denom.item() == 0:
        expert_loss = torch.zeros((), device=device)
    else:
        expert_loss = total_expert_loss_sum / (total_denom + 1e-8)
    
    # 🛡️ 改进：detach() 防止 graph 泄漏，虽然 item() 也会 sync，但 detach 语义更清晰
    expert_loss_item = expert_loss.detach().item()

    expert_outputs = model_outputs['expert_outputs']
    dummy_loss = 0.0
    
    # 遍历所有 expert 的输出 logits，全都乘 0 加进去
    # 这样无论 batch 里有没有 mixed/num/cat 数据，计算图永远是通的
    for key in ["num_bin_logits", "num_residual", 
                "cat_logits", 
                "mixed_mask_logits", "mixed_bin_logits", "mixed_residual"]:
        if key in expert_outputs:
            dummy_loss += expert_outputs[key].sum() * 0.0
    total_loss = (lm_loss_weight * lm_loss) + (expert_loss_weight * expert_loss)+ dummy_loss
    
    return total_loss, lm_loss_item, expert_loss_item


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        # 1) 变长序列
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # 2) pad
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # 3) labels：pad 位置置 -100
        labels = padded_input_ids.clone()
        labels[padded_attention_mask == 0] = -100

        result = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": labels,
        }

        # 4) stack 固定长度列级标签
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TypeAwareGPT2 with Hybrid Loss (LM + Expert) using Opacus"
    )
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                       help="Path to training CSV file")
    parser.add_argument("--validation_file", type=str, default=None,
                       help="Path to validation CSV file (optional)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints and logs")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                       help="Pretrained model name or path")
    parser.add_argument("--config_name", type=str, default=None,
                       help="Config name (defaults to model_name_or_path)")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                       help="Tokenizer name (defaults to model_name_or_path)")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Maximum training steps (overrides num_train_epochs)")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                       help="Number of warmup steps")
    
    # Expert loss arguments
    parser.add_argument("--expert_loss_weight", type=float, default=1.0,
                       help="Weight for expert loss in hybrid loss")
    parser.add_argument("--lm_loss_weight", type=float, default=0.1, 
                       help="Weight for LM loss (reduce to focus on experts)")
    parser.add_argument("--num_bins", type=int, default=100,
                       help="Number of bins for numeric prediction")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout")
    
    # Two-stage training
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                       help="Path to Stage 1 checkpoint (loads before LoRA/Opacus)")
    
    # Opacus/Privacy arguments
    parser.add_argument("--enable_privacy", type=bool, default=False,
                       help="Enable differential privacy")
    parser.add_argument("--target_epsilon", type=float, default=1.0,
                       help="Target epsilon for privacy budget")
    parser.add_argument("--target_delta", type=float, default=1e-5,
                       help="Target delta for privacy budget")
    parser.add_argument("--max_grad_norm", type=float, default=0.1,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                       help="Micro batch size for BatchMemoryManager")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--target_col", type=str, default=None, 
                       help="The target column name to move to the first position (Target-First Strategy).")
    return parser.parse_args()

def load_stage1_checkpoint(model, checkpoint_path, logger):
    """Load Stage 1 checkpoint weights into base model (before LoRA is applied)."""
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    
    if "safetensors" in checkpoint_path:
        ckpt = {}
        with safe_open(checkpoint_path, framework="pt") as f:
            for k in f.keys():
                ckpt[k] = f.get_tensor(k)
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu") # weights_only=False if needed
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    
    # ================= 🚀 核心修改：清洗僵尸权重 =================
    # 强制删除所有包含 'expert' 的权重，确保 Stage 2 从随机初始化的 Expert 开始训练
    # 这样可以避免 Stage 1 的伪数据分布对 Stage 2 产生“负迁移”
    keys_to_drop = []
    for k in ckpt.keys():
        if "expert" in k:
            keys_to_drop.append(k)
            
    if len(keys_to_drop) > 0:
        logger.info(f"Dropping {len(keys_to_drop)} expert keys from Stage 1 checkpoint (to avoid negative transfer).")
        # 打印几个例子确认一下
        logger.info(f"Examples: {keys_to_drop[:3]}")
        
    for k in keys_to_drop:
        del ckpt[k]
    # ==========================================================

    # Load with strict=False to handle missing expert keys
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    
    # 这里的 logging 可以稍微简化，因为我们要 drop 很多 keys，missing 是预期的
    if missing_keys:
        # 过滤掉 expert 相关的 missing 警告，只关心其他的
        real_missing = [k for k in missing_keys if "expert" not in k]
        if real_missing:
            logger.warning(f"Missing keys (excluding experts): {real_missing[:10]}")
    
    logger.info("Stage 1 checkpoint loaded successfully (Experts reset)")
    return model
# def load_stage1_checkpoint(model, checkpoint_path, logger):
#     """Load Stage 1 checkpoint weights into base model (before LoRA is applied)."""
#     logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    
#     if "safetensors" in checkpoint_path:
#         ckpt = {}
#         with safe_open(checkpoint_path, framework="pt") as f:
#             for k in f.keys():
#                 ckpt[k] = f.get_tensor(k)
#     else:
#         ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#             ckpt = ckpt["model_state_dict"]
    
#     # Handle potential vocab-size mismatch for categorical head
#     def _drop_if_shape_mismatch(key):
#         if key in ckpt:
#             saved_shape = ckpt[key].shape
#             target_shape = model.state_dict()[key].shape
#             if saved_shape != target_shape:
#                 logger.warning(
#                     f"Drop mismatched key '{key}': ckpt {saved_shape} vs model {target_shape}"
#                 )
#                 ckpt.pop(key)

#     _drop_if_shape_mismatch("cat_expert.cat_head.weight")
#     _drop_if_shape_mismatch("cat_expert.cat_head.bias")

#     # At this point, model is still the base TypeAwareGPT2 (no LoRA applied yet)
#     # Load with strict=False to handle potential key mismatches (e.g., embedding size differences)
#     missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
#     if missing_keys:
#         logger.warning(f"Missing keys (first 10): {missing_keys[:10]}")
#         if len(missing_keys) > 10:
#             logger.warning(f"... and {len(missing_keys) - 10} more missing keys")
#     if unexpected_keys:
#         logger.warning(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
#         if len(unexpected_keys) > 10:
#             logger.warning(f"... and {len(unexpected_keys) - 10} more unexpected keys")
    
#     logger.info("Stage 1 checkpoint loaded successfully")
#     return model

# --- Helper: Robust model saving for Opacus/LoRA/DDP ---
def _save_model(model, save_dir: str, logger=None):
    """
    Recursively unwrap the model to find the underlying PeftModel or Transformers model
    that has 'save_pretrained'.
    """
    # 递归解包，直到找到 save_pretrained 方法或者没有 _module 属性为止
    # Opacus 使用 ._module, DDP 使用 .module
    unwrap_model = model
    wrap_layers = []
    while hasattr(unwrap_model, "_module") or hasattr(unwrap_model, "module"):
        if hasattr(unwrap_model, "_module"):
            wrap_layers.append(type(unwrap_model).__name__)
            unwrap_model = unwrap_model._module
        elif hasattr(unwrap_model, "module"):
            wrap_layers.append(type(unwrap_model).__name__)
            unwrap_model = unwrap_model.module
        else:
            break
    
    if logger:
        logger.info(f"Saving model to {save_dir}")
        if wrap_layers:
            logger.info(f"Unwrapped layers: {' -> '.join(wrap_layers)}")
        logger.info(f"Final model type: {type(unwrap_model).__name__}")
            
    # 此时 unwrap_model 应该是 PeftModel 或 TypeAwareGPT2
    if hasattr(unwrap_model, "save_pretrained"):
        # 对于 PeftModel，这只会保存 adapter_model.bin (LoRA权重)
        # 这正是我们想要的，因为 Base Model 是冻结的且已经有 Stage 1 存档
        unwrap_model.save_pretrained(save_dir)
        if logger:
            saved_files = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
            logger.info(f"Saved files: {', '.join(saved_files)}")
    else:
        # 兜底方案
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(unwrap_model.state_dict(), model_path)
        if logger:
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"Saved state_dict to {model_path} ({file_size:.2f} MB)")
import warnings

def main():
    # 过滤包含 "non-full backward hook" 关键词的警告
    warnings.filterwarnings("ignore", message=".*non-full backward hook.*", category=FutureWarning)

    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    mkdir(args.output_dir)
    
    # Setup logger
    logger = get_logger(filename=os.path.join(args.output_dir, "train.log"))
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # Load tokenizer
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    import pandas as pd
    train_df = pd.read_csv(args.train_file)
    # ================= 🚀 新增：Target-First 重排逻辑 =================
    if args.target_col:
        # 为了鲁棒性，先检查一下列名是否存在（这里不做 lower() 处理，保持原样匹配）
        if args.target_col in train_df.columns:
            logger.info(f"Applying Target-First strategy: Moving '{args.target_col}' to the first column.")
            # 将 target 移到第一列，其余列保持原序
            cols = [args.target_col] + [c for c in train_df.columns if c != args.target_col]
            train_df = train_df[cols]
        else:
            logger.warning(f"Target column '{args.target_col}' not found in training data! Skipping reordering.")
    train_dataset = LLMtgDataset.from_pandas(train_df)
    train_dataset.set_serializer("great")
    train_dataset.set_tokenizer(tokenizer)
    
    # Generate metadata and calculate cat_vocab_size
    metadata = get_metadata(train_df)
    train_dataset.metadata = metadata
    cat_vocab_size = calculate_data_stats(train_dataset)
    logger.info(f"Calculated cat_vocab_size: {cat_vocab_size}")
    
    validation_dataset = None
    if args.validation_file:
        validation_df = pd.read_csv(args.validation_file)
        validation_dataset = LLMtgDataset.from_pandas(validation_df)
        validation_dataset.set_serializer("great")
        validation_dataset.set_tokenizer(tokenizer)
        validation_dataset.metadata = metadata
    
    # Load config and initialize model
    config_name = args.config_name if args.config_name else args.model_name_or_path
    config = AutoConfig.from_pretrained(config_name, cache_dir=args.cache_dir)
    
    # Initialize TypeAwareGPT2 model
    model = TypeAwareGPT2(
        config=config,
        num_bins=args.num_bins,
        cat_vocab_size=cat_vocab_size,
        type_token_ids=train_dataset.type_token_ids if hasattr(train_dataset, 'type_token_ids') else None
    )
    
    # Resize token embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing embeddings from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        #重新绑定权重，确保 LM Head 同步更新
        # model.tie_weights() #Opacus 不支持 Tie Weights
    
    # Set type token IDs
    if hasattr(train_dataset, 'type_token_ids') and train_dataset.type_token_ids:
        model.set_type_token_ids(train_dataset.type_token_ids)
    
    # STEP 1: Load Stage 1 checkpoint (if provided) - BEFORE LoRA to avoid key mismatches
    if args.stage1_checkpoint:
        model = load_stage1_checkpoint(model, args.stage1_checkpoint, logger)
    
    # Move model to device (after checkpoint loading, before LoRA)
    model = model.to(args.device)
    
    # --- FIX 2: 先运行 ModuleValidator (Opacus Compatibility) ---
    # 必须在 LoRA 之前把 GPT2 的 Conv1D 换成 Linear，否则 LoRA 会挂在错误的层上
    logger.info("Fixing model with ModuleValidator (Before LoRA)...")
    try:
        model = ModuleValidator.fix(model)
        logger.info("Model fixed successfully")
    except Exception as e:
        logger.warning(f"ModuleValidator.fix failed: {e}")
        
    # STEP 2: Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for GPT2-based models
        bias="none",
        target_modules=["c_attn", "c_fc", "c_proj"],  # GPT2 modules
        modules_to_save=["num_expert", "mixed_expert", "cat_expert"],
        # target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # STEP 3: Apply ModuleValidator.fix (for Opacus compatibility)
    logger.info("Fixing model with ModuleValidator...")
    try:
        model = ModuleValidator.fix(model)
        logger.info("Model fixed successfully")
    except Exception as e:
        logger.warning(f"ModuleValidator.fix failed: {e}. Continuing anyway...")
    
    # STEP 4: Define optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    logger.info(f"Optimizer initialized with {len(trainable_params)} parameter groups")
    
    # STEP 5: Setup data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        drop_last=True,
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    logger.info(f"Training for {args.num_train_epochs} epochs ({args.max_train_steps} steps)")
    
    # STEP 6: Apply Opacus (make private)
    privacy_engine = None
    if args.enable_privacy:
        logger.info("Applying Opacus privacy engine...")
        privacy_engine = PrivacyEngine(accountant="rdp")
        
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            max_grad_norm=args.max_grad_norm,
            epochs=args.num_train_epochs,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
        )
        logger.info("*** Privacy Info ***")
        logger.info(f"Noise multiplier: {optimizer.noise_multiplier}")
        logger.info(f"Max grad norm: {args.max_grad_norm}")
        logger.info(f"Target epsilon: {args.target_epsilon}")
        logger.info(f"Target delta: {args.target_delta}")
        logger.info("*** ***")
    
    # Training loop
    model.train()
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Training")

    for epoch in range(args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break

        # ✅ DP + micro-batch：用 BatchMemoryManager
        if args.enable_privacy and args.micro_batch_size < args.batch_size:
            data_iter_ctx = BatchMemoryManager(
                data_loader=train_dataloader,
                max_physical_batch_size=args.micro_batch_size,
                optimizer=optimizer,
            )
        else:
            data_iter_ctx = None

        if data_iter_ctx is not None:
            ctx = data_iter_ctx
            data_iter = ctx.__enter__()
        else:
            ctx = None
            data_iter = train_dataloader

        try:
            for step, batch in enumerate(data_iter):
                if global_step >= args.max_train_steps:
                    break

                batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}

                # Opacus requires fresh per-sample grad buffers each micro/mini-batch
                optimizer.zero_grad(set_to_none=True)

                # ✅ ⑥ strict：训练必须有 anchor
                if "expert_token_idxs" not in batch:
                    raise ValueError("Training batch missing 'expert_token_idxs'. Dataset must provide anchor positions.")

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    expert_token_idxs=batch["expert_token_idxs"],
                    col_type_ids=batch.get("col_type_ids"),
                    output_expert_logits=True,
                )

                loss, lm_loss_item, expert_loss_item = compute_loss(
                    outputs, batch,
                    expert_loss_weight=args.expert_loss_weight,
                    lm_loss_weight=args.lm_loss_weight,
                    num_bins=args.num_bins,
                )

                dummy_loss = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        dummy_loss += p.sum() * 0.0
                
                loss = loss + dummy_loss

                loss.backward()
                if args.enable_privacy:
                    bad = []
                    for name, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        gs = getattr(p, "grad_sample", None)
                        # gs 可能是 None，或者是空 list（某些版本/情况）
                        if gs is None:
                            bad.append(name)

                    if bad:
                        logger.error(f"grad_sample is None for {len(bad)} params. Examples: {bad[:20]}")
                        raise ValueError("Some trainable params were not used in this forward/backward step.")


                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                if global_step % args.logging_steps == 0:
                    logger.info(
                        f"Step {global_step}: Loss={loss.item():.4f}, "
                        f"LM Loss={lm_loss_item:.4f}, Expert Loss={expert_loss_item:.4f}"
                    )

                if global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    mkdir(save_dir)
                    _save_model(model, save_dir, logger=logger)

        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    progress_bar.close()

    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    mkdir(final_dir)
    _save_model(model, final_dir, logger=logger)


if __name__ == "__main__":
    main()

