import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model

class NumericExpert(nn.Module):
    """Column-level numeric expert (Regression Version)."""
    def __init__(self, hidden_size: int, num_bins: int = 100):
        super().__init__()
        # self.bin_head = nn.Linear(hidden_size, num_bins)
        # self.residual_head = nn.Sequential(
        # [Modified] Regression Expert: Predict value directly
        # Removed bin_head, renamed residual_head to value_head for clarity
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, col_hidden_states: torch.Tensor):
        # bin_logits = self.bin_head(col_hidden_states)               # [B,C,K]
        # residual = self.residual_head(col_hidden_states).squeeze(-1)  # [B,C]
        # return bin_logits, residual
        # [Modified] No bin_logits for regression
        bin_logits = None 
        # Predict normalized value directly
        pred_value = self.value_head(col_hidden_states).squeeze(-1)  # [B,C]
        return bin_logits, pred_value


class CategoricalExpert(nn.Module):
    """
    Column-level categorical expert.
    Output dim is `cat_vocab_size` (e.g., 50 or 100), NOT tokenizer vocab size.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.cat_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, col_hidden_states: torch.Tensor):
        return self.cat_head(col_hidden_states) # [B,C,V_cat]


class MixedExpert(nn.Module):
    """Column-level mixed expert."""
    def __init__(self, hidden_size: int, num_bins: int = 100):
        super().__init__()
        # Output Logits for stability (BCEWithLogitsLoss required)
        self.mask_head = nn.Linear(hidden_size, 1)
        self.numeric_expert = NumericExpert(hidden_size, num_bins)

    def forward(self, col_hidden_states: torch.Tensor):
        mask_logits = self.mask_head(col_hidden_states).squeeze(-1)   # [B,C]
        bin_logits, residual = self.numeric_expert(col_hidden_states) # [B,C,K], [B,C]
        return mask_logits, bin_logits, residual


class TypeAwareGPT2(GPT2PreTrainedModel):
    """
    Type-aware GPT2 backbone + column-level experts + LM Head.
    """
    # 忽略加载权重时的某些无关警告，这是一个好习惯
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config, num_bins=100, cat_vocab_size=51, type_token_ids=None):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))

        # Experts
        self.num_expert = NumericExpert(hidden_size, num_bins)
        self.mixed_expert = MixedExpert(hidden_size, num_bins)
        self.cat_expert = CategoricalExpert(hidden_size, cat_vocab_size)
        
        # LM Head (Standard GPT-2 tied weights setup)
        # bias=False 是必须的，因为 wte 通常没有 bias
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        # Buffers
        self.register_buffer("num_token_id", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("cat_token_id", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("mix_token_id", torch.tensor(-1, dtype=torch.long))

        if type_token_ids is not None:
            self.set_type_token_ids(type_token_ids)

        # Init weights and Tie weights
        self.post_init() 

    # --- Hook Methods for resize_token_embeddings ---
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        # GPT2Model 的 embedding 在 self.transformer.wte
        # 如果 embedding 被 resize 了，这里会把 lm_head 指向新的 embedding
        if hasattr(self.transformer, "wte"):
            self.lm_head.weight = self.transformer.wte.weight

    def set_type_token_ids(self, type_token_ids: dict):
        if not type_token_ids: return
        
        def _to_single_id(x):
            if x is None: return None
            if isinstance(x, (list, tuple)): return int(x[0]) if len(x) > 0 else None
            return int(x)

        num_id = _to_single_id(type_token_ids.get("NUM"))
        cat_id = _to_single_id(type_token_ids.get("CAT"))
        mix_id = _to_single_id(type_token_ids.get("MIX"))

        if num_id is not None: self.num_token_id.fill_(num_id)
        if cat_id is not None: self.cat_token_id.fill_(cat_id)
        if mix_id is not None: self.mix_token_id.fill_(mix_id)

    def _gather_column_hidden_states(self, hidden_states, col_positions):
        B, T, H = hidden_states.shape
        C = col_positions.shape[1]
        valid_mask = (col_positions >= 0) & (col_positions < T)
        pos_safe = col_positions.clamp(0, T - 1)
        
        idx = pos_safe.unsqueeze(-1).expand(B, C, H)
        col_hidden = hidden_states.gather(1, idx)
        
        # Zero out invalid gathered states
        col_hidden = col_hidden * valid_mask.unsqueeze(-1).to(col_hidden.dtype)
        return col_hidden, valid_mask

    def _find_column_positions_vectorized(self, input_ids, expected_num_cols, col_type_ids=None):
        """
        STRICT fallback: find anchor token positions ([NUM]/[CAT]/[MIX]) if dataset didn't provide them.
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        num_id = int(self.num_token_id.item())
        cat_id = int(self.cat_token_id.item())
        mix_id = int(self.mix_token_id.item())
        
        if num_id < 0 and cat_id < 0 and mix_id < 0:
            raise ValueError("Type tokens not set via set_type_token_ids! Call model.set_type_token_ids({...})")

        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        if num_id >= 0: mask |= (input_ids == num_id)
        if cat_id >= 0: mask |= (input_ids == cat_id)
        if mix_id >= 0: mask |= (input_ids == mix_id)

        # ✅ REAL strict check (per-sample)
        found = mask.sum(dim=1)  # [B]
        if torch.any(found < expected_num_cols):
            bad = torch.nonzero(found < expected_num_cols, as_tuple=False).view(-1).tolist()
            # 详细报错信息，方便排查是哪个样本坏了
            raise ValueError(
                f"Found fewer type tokens than expected cols ({expected_num_cols}) for rows {bad}. "
                f"min_found={int(found.min().item())}, max_found={int(found.max().item())}. "
                "Likely truncation or serialization bug."
            )

        # Extract first expected_num_cols positions in order
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        big = torch.full_like(idxs, fill_value=T + 1)
        masked_idxs = torch.where(mask, idxs, big)
        
        sorted_idxs, _ = torch.sort(masked_idxs, dim=1)
        col_positions = sorted_idxs[:, :expected_num_cols].contiguous()
        return col_positions


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        col_positions=None,
        expert_token_idxs=None,
        col_type_ids=None,
        output_expert_logits=True,
        labels=None,
        **kwargs
    ):
        if labels is not None:
            kwargs.pop("labels", None)

        # 1) backbone
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = out[0] if isinstance(out, tuple) else out.last_hidden_state

        # 2) lm_head (始终可用)
        lm_logits = self.lm_head(hidden_states)

        # ✅ 关键：如果不需要 expert，就不要做列位置推断
        if not output_expert_logits:
            return {
                "lm_logits": lm_logits,
                "hidden_states": hidden_states,
            }

        # 3) resolve positions
        if col_positions is None and expert_token_idxs is not None:
            col_positions = expert_token_idxs

        if col_positions is None:
            if col_type_ids is None:
                raise ValueError("Training requires 'expert_token_idxs' or 'col_type_ids'. Cannot infer column count.")

            expected_num_cols = col_type_ids.shape[1]
            col_positions = self._find_column_positions_vectorized(input_ids, expected_num_cols, col_type_ids)

        # 4) gather + experts
        col_hidden, valid_mask = self._gather_column_hidden_states(hidden_states, col_positions)

        num_bin_logits, num_residual = self.num_expert(col_hidden)
        cat_logits = self.cat_expert(col_hidden)
        mixed_mask_logits, mixed_bin_logits, mixed_residual = self.mixed_expert(col_hidden)

        return {
            "hidden_states": hidden_states,
            "col_positions": col_positions,
            "valid_mask": valid_mask,
            "lm_logits": lm_logits,
            "expert_outputs": {
                "cat_logits": cat_logits,
                "num_bin_logits": num_bin_logits,
                "num_residual": num_residual,
                "mixed_mask_logits": mixed_mask_logits,
                "mixed_bin_logits": mixed_bin_logits,
                "mixed_residual": mixed_residual,
            },
        }
