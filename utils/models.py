import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model

class NumericExpert(nn.Module):
    """Column-level numeric expert."""
    def __init__(self, hidden_size: int, num_bins: int = 100):
        super().__init__()
        self.bin_head = nn.Linear(hidden_size, num_bins)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, col_hidden_states: torch.Tensor):
        bin_logits = self.bin_head(col_hidden_states)               # [B,C,K]
        residual = self.residual_head(col_hidden_states).squeeze(-1)  # [B,C]
        return bin_logits, residual

class CategoricalExpert(nn.Module):
    """[Missing Component Added] Column-level categorical expert."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.cat_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, col_hidden_states: torch.Tensor):
        return self.cat_head(col_hidden_states) # [B,C,V]

class MixedExpert(nn.Module):
    """Column-level mixed expert."""
    def __init__(self, hidden_size: int, num_bins: int = 100):
        super().__init__()
        # IMPORTANT: outputs logits (no sigmoid). Use BCEWithLogitsLoss in training.
        self.mask_head = nn.Linear(hidden_size, 1)
        self.numeric_expert = NumericExpert(hidden_size, num_bins)

    def forward(self, col_hidden_states: torch.Tensor):
        mask_logits = self.mask_head(col_hidden_states).squeeze(-1)   # [B,C]
        bin_logits, residual = self.numeric_expert(col_hidden_states) # [B,C,K], [B,C]
        return mask_logits, bin_logits, residual

class TypeAwareGPT2(GPT2PreTrainedModel):
    def __init__(self, config, num_bins=100, cat_vocab_size=51, type_token_ids=None):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))

        self.num_expert = NumericExpert(hidden_size, num_bins)
        self.mixed_expert = MixedExpert(hidden_size, num_bins)
        self.cat_expert = CategoricalExpert(hidden_size, cat_vocab_size)  # ✅ FIX

        self.register_buffer("num_token_id", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("cat_token_id", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("mix_token_id", torch.tensor(-1, dtype=torch.long))

        if type_token_ids is not None:
            self.set_type_token_ids(type_token_ids)

        self.post_init()

    @staticmethod
    def _to_single_id(x):
        if x is None: return None
        if isinstance(x, (list, tuple)): return int(x[0]) if len(x) > 0 else None
        return int(x)

    def set_type_token_ids(self, type_token_ids: dict):
        if not type_token_ids: return
        num_id = self._to_single_id(type_token_ids.get("NUM"))
        cat_id = self._to_single_id(type_token_ids.get("CAT"))
        mix_id = self._to_single_id(type_token_ids.get("MIX"))

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
        col_hidden = col_hidden * valid_mask.unsqueeze(-1).to(col_hidden.dtype)
        return col_hidden, valid_mask

    def _find_column_positions_vectorized(self, input_ids, expected_num_cols, col_type_ids=None):
        """Fallback: find anchors if dataset didn't provide them. STRICT mode."""
        B, T = input_ids.shape
        device = input_ids.device

        num_id = int(self.num_token_id.item())
        cat_id = int(self.cat_token_id.item())
        mix_id = int(self.mix_token_id.item())

        if num_id < 0 and cat_id < 0 and mix_id < 0:
            raise ValueError("Type tokens not set via set_type_token_ids! Call model.set_type_token_ids({...})")

        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        if num_id >= 0:
            mask |= (input_ids == num_id)
        if cat_id >= 0:
            mask |= (input_ids == cat_id)
        if mix_id >= 0:
            mask |= (input_ids == mix_id)

        # ✅ REAL strict check: per-sample count of found anchors
        found = mask.sum(dim=1)  # [B]
        if torch.any(found < expected_num_cols):
            bad = torch.nonzero(found < expected_num_cols, as_tuple=False).view(-1).tolist()
            raise ValueError(
                f"Found fewer type tokens than expected cols ({expected_num_cols}) for batch rows {bad}. "
                f"min_found={int(found.min().item())}, max_found={int(found.max().item())}. "
                f"Likely truncation or serialization bug."
            )

        # Extract first expected_num_cols positions in order
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        big = torch.full_like(idxs, fill_value=T + 1)
        masked_idxs = torch.where(mask, idxs, big)  # [B,T]
        sorted_idxs, _ = torch.sort(masked_idxs, dim=1)  # matches first

        col_positions = sorted_idxs[:, :expected_num_cols].contiguous()  # [B,C]
        # At this point strict check guarantees no big values in first C positions
        return col_positions


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        col_positions=None,
        expert_token_idxs=None, # Dataset returns this key
        col_type_ids=None,
        **kwargs
    ):
        # 1. Backbone
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = out[0] if isinstance(out, tuple) else out.last_hidden_state

        # 2. Resolve Positions (Architecture: Gather-Scatter)
        
        if col_positions is None and expert_token_idxs is not None:
            col_positions = expert_token_idxs

        if col_positions is None:
            if col_type_ids is None:
                # 训练时绝对不能没有锚点信息
                raise ValueError("Training requires 'expert_token_idxs' (col_positions) or 'col_type_ids' to infer count.")
            # Fallback (Slow & Risky)
            expected_num_cols = col_type_ids.shape[1] if col_type_ids is not None else 10 # Guess
            col_positions = self._find_column_positions_vectorized(input_ids, expected_num_cols, col_type_ids)

        # 3. Gather Hidden States [B, C, H]
        col_hidden, valid_mask = self._gather_column_hidden_states(hidden_states, col_positions)

        # 4. Expert Heads
        num_bin_logits, num_residual = self.num_expert(col_hidden)
        # [Fixed] Added Cat Expert
        cat_logits = self.cat_expert(col_hidden) 
        # Mixed
        mixed_mask_logits, mixed_bin_logits, mixed_residual = self.mixed_expert(col_hidden)

        return {
            "hidden_states": hidden_states,
            "col_positions": col_positions,
            "expert_outputs": {
                "cat_logits": cat_logits,             # [B, C, Vocab]
                "num_bin_logits": num_bin_logits,
                "num_residual": num_residual,
                "mixed_mask_logits": mixed_mask_logits, # Logits!
                "mixed_bin_logits": mixed_bin_logits,
                "mixed_residual": mixed_residual,
            },
        }