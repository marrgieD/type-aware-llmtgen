import random
import typing as tp

import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding

from datasets.arrow_dataset import (
    DatasetTransformationNotAllowedError,
)

import pandas as pd
import numpy as np
from scipy.stats import skew  # 新增: 用于计算偏度


class LLMtgDataset(Dataset):
    """GReaT Dataset

    The LLMtgDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_shuffler(self, shuffle=True):
        if shuffle:
            self.shuffle = True
        else:
            self.shuffle = False

    def set_serializer(self, serializer="great"):
        self.serializer = serializer
        # === 新增 Type Identifiers（所有 serializer 都应该包含） ===
        base_type_tokens = {
            "BOS": "[BOS]",
            "EOS": "[EOS]",
            "EOC": "[EOC]",
            "NUM": "[NUM]",
            "CAT": "[CAT]",
            "MIX": "[MIX]",
            "UNK": "[UNK]",
            "Missing": "Missing"
        }
        
        if serializer == "list":
            self.others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n", **base_type_tokens}
        elif serializer == "text":
            self.others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": ".", **base_type_tokens}
        elif serializer == "apval":
            self.others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ",", **base_type_tokens}
        else:  # serializer == "great" or other
            self.others_token = {
                "text_sep": ", ",       # 普通分隔符
                "key_val_sep": "is",    # 键值分隔符（"great" serializer 使用 "is"）
                "prefix": None,
                **base_type_tokens
            }

    def set_tokenizer(self, tokenizer):
        """Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

        try:
            name_or_path = self.tokenizer.init_kwargs["name_or_path"].lower()
        except:
            name_or_path = "gpt2"
        special_tokens_list = [
            self.others_token.get("BOS", "[BOS]"),
            self.others_token.get("EOS", "[EOS]"),
            self.others_token.get("EOC", "[EOC]"),
            self.others_token.get("NUM", "[NUM]"),
            self.others_token.get("CAT", "[CAT]"),
            self.others_token.get("MIX", "[MIX]"),
            self.others_token.get("UNK", "[UNK]"),
        ]
        
        # 过滤掉 None 或空字符串
        special_tokens_list = [t for t in special_tokens_list if t]

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens_list}
        )
        self.type_token_ids = {
        "NUM": self.tokenizer.convert_tokens_to_ids(self.others_token.get("NUM", "[NUM]")),
        "CAT": self.tokenizer.convert_tokens_to_ids(self.others_token.get("CAT", "[CAT]")),
        "MIX": self.tokenizer.convert_tokens_to_ids(self.others_token.get("MIX", "[MIX]")),
        "EOC": self.tokenizer.convert_tokens_to_ids(self.others_token.get("EOC", "[EOC]")),
        "BOS": self.tokenizer.convert_tokens_to_ids(self.others_token.get("BOS", "[BOS]")),
        "EOS": self.tokenizer.convert_tokens_to_ids(self.others_token.get("EOS", "[EOS]")),
        "UNK": self.tokenizer.convert_tokens_to_ids(self.others_token.get("UNK", "[UNK]")),
        }
        for k, v in self.type_token_ids.items():
            if v is None or v < 0:
                raise ValueError(f"Special token id for {k} is invalid: {v}")
        if num_added > 0:
            print(f"Added {num_added} special tokens: {special_tokens_list}")
        # this was necessary because of llama2 which by default prepends space to input.
        # couldn't figure out a way to disable this feature
        # self.tokenizer.add_tokens(self.others_token["text_sep"])

        # don't forget to
        # model.resize_token_embeddings(len(tokenizer))

        if "llama" in name_or_path:
            self.keys_token_id = {
                j: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for j in self.column_names
            }
            self.values_token_id = {
                j: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for j in list(map(str, set(self.to_pandas().values.flatten())))
            }
            self.others_token_id = {
                k: self.tokenizer.encode(f"{j}", add_special_tokens=False)
                for k, j in self.others_token.items()
            }
        else:
            self.keys_token_id = {
                j: self.tokenizer.encode(f" {j}", add_special_tokens=False)
                for j in self.column_names
            }
            self.values_token_id = {
                j: self.tokenizer.encode(f" {j}", add_special_tokens=False)
                for j in list(map(str, set(self.to_pandas().values.flatten())))
            }
            self.others_token_id = {}
            for k, v in self.others_token.items():
                if k == "text_sep":
                    self.others_token_id[k] = self.tokenizer.encode(
                        f"{v}", add_special_tokens=False
                    )
                else:
                    self.others_token_id[k] = self.tokenizer.encode(
                        f" {v}", add_special_tokens=False
                    )

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        row = self._data.fast_slice(key, 1)

        # 1) metadata
        if not hasattr(self, "metadata"):
            self.metadata = get_metadata(self.to_pandas())

        # 2) multiprocessing-safe shuffle (LOCAL, not self.xxx)
        shuffle_idx = list(range(len(self.column_names)))
        if self.shuffle:
            random.shuffle(shuffle_idx)

        # 3) build text
        if self.serializer == "apval":
            column_values = [str(j) for i in row.columns for j in i.to_pylist()]
            column_values = [column_values[i] for i in shuffle_idx]
            cat_keys = f"{self.others_token.get('text_sep', ', ')} ".join(
                [self.column_names[i] for i in shuffle_idx]
            )
            shuffled_text = " %s %s %s" % (
                cat_keys,
                self.others_token.get("key_val_sep", ": "),
                f"{self.others_token.get('text_sep', ', ')} ".join(column_values),
            )
        else:
            text_parts = [self.others_token.get("BOS", "")]

            token_sep = self.others_token.get("text_sep", ", ")
            token_kv = self.others_token.get("key_val_sep", ": ")
            token_eoc = self.others_token.get("EOC", "")
            missing_val = self.others_token.get("Missing", "Missing")

            for i in shuffle_idx:
                col_name = self.column_names[i]
                val = row.columns[i].to_pylist()[0]

                col_meta = self.metadata.get(col_name.lower())
                type_token = token_sep
                if col_meta:
                    ctype = col_meta["type"]
                    if ctype == "numerical":
                        type_token = self.others_token.get("NUM", "")
                    elif ctype == "categorical":
                        type_token = self.others_token.get("CAT", "")
                    elif ctype == "mixed":
                        type_token = self.others_token.get("MIX", "")

                # keep Missing deterministic
                if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
                    val_str = missing_val
                else:
                    val_str = str(val).strip()

                part = f"{type_token} {col_name}{token_kv}{val_str} {token_eoc}"
                text_parts.append(part)

            text_parts.append(self.others_token.get("EOS", ""))
            text_parts = [p for p in text_parts if p]
            shuffled_text = " ".join(text_parts)

        # 4) tokenize (建议 truncation=True，避免极端长行把训练直接搞炸)
        max_len = getattr(self, "max_length", None)
        if max_len is None:
            max_len = getattr(self.tokenizer, "model_max_length", None)

        tokenized_text = self.tokenizer(
            shuffled_text,
            truncation=True if max_len is not None else False,
            max_length=max_len,
        )

        # 5) expert labels (pass shuffle_idx explicitly)
        row_pandas = row.to_pandas().iloc[0]
        expert_labels = self.encode_row(row_pandas, self.metadata, shuffle_idx=shuffle_idx)

        # 6) build output tensors
        output = dict(tokenized_text)
        for k in ["input_ids", "attention_mask"]:
            if k in output and not isinstance(output[k], torch.Tensor):
                output[k] = torch.tensor(output[k], dtype=torch.long)

        # 7) build expert_token_idxs (alignment)
        # only meaningful for type-aware serialization
        if self.serializer != "apval":
            input_ids = output["input_ids"]

            ids = self.type_token_ids
            mask = (input_ids == ids["NUM"]) | (input_ids == ids["CAT"]) | (input_ids == ids["MIX"])

            idxs = torch.nonzero(mask, as_tuple=False).view(-1)  # positions in token sequence

            # enforce fixed length = number of columns
            C = len(shuffle_idx)
            if idxs.numel() >= C:
                expert_token_idxs = idxs[:C]
            else:
                pad = torch.full((C - idxs.numel(),), -1, dtype=torch.long)
                expert_token_idxs = torch.cat([idxs, pad], dim=0)

                # IMPORTANT: if truncation cut some columns away, keep label shape fixed and ignore them
                # We do that by setting ignore targets for the missing tail columns.
                missing = C - idxs.numel()
                if missing > 0:
                    for key in ["num_bin", "cat_id", "mixed_bin"]:
                        expert_labels[key][-missing:] = [-100] * missing
                    for key in ["num_res", "mixed_res", "mixed_mask"]:
                        expert_labels[key][-missing:] = [0.0] * missing
                    expert_labels["col_type_ids"][-missing:] = [-100] * missing

            output["expert_token_idxs"] = expert_token_idxs
        else:
            # legacy mode: no alignment info
            output["expert_token_idxs"] = torch.full((len(shuffle_idx),), -1, dtype=torch.long)

        # 8) inject expert labels (fixed-length vectors)
        for k, v in expert_labels.items():
            # ints vs floats
            if k in ["num_res", "mixed_res", "mixed_mask"]:
                output[k] = torch.tensor(v, dtype=torch.float)
            else:
                output[k] = torch.tensor(v, dtype=torch.long)

        return output


    def encode_row(self, row_data, metadata, shuffle_idx=None):
        expert_labels = {
            "num_bin": [], "num_res": [],
            "cat_id": [],
            "mixed_mask": [], "mixed_bin": [], "mixed_res": [],
            "col_type_ids": []
        }

        if shuffle_idx is None:
            ordered_cols = list(self.column_names)
        else:
            ordered_cols = [self.column_names[i] for i in shuffle_idx]

        for col in ordered_cols:
            col_meta = metadata[col.lower()]
            val = row_data[col]
            ctype = col_meta["type"]

            if ctype == "categorical":
                expert_labels["col_type_ids"].append(1)
                cats = col_meta["categories"]

                if pd.isna(val):
                    cat_id = cats.get("missing_id", cats.get("unk_id", 0))
                else:
                    val_str = str(val)
                    unk_id = cats.get("unk_id", 0)
                    cat_id = cats["w2i"].get(val_str)
                    if cat_id is None:
                        cat_id = cats["w2i_lower"].get(val_str.lower(), unk_id)

                expert_labels["cat_id"].append(cat_id)
                expert_labels["num_bin"].append(-100); expert_labels["num_res"].append(0.0)
                expert_labels["mixed_mask"].append(0.0); expert_labels["mixed_bin"].append(-100); expert_labels["mixed_res"].append(0.0)

            elif ctype == "numerical" or ctype == "mixed":
                def encode_numeric_val(raw_val, meta_stats):
                    v = float(raw_val)
                    if meta_stats["needs_log"]:
                        v = np.log(v + meta_stats.get("log_shift", 1e-6))

                    n_min, n_max = meta_stats["norm_min"], meta_stats["norm_max"]
                    nv = (v - n_min) / (n_max - n_min + 1e-9)
                    nv = np.clip(nv, 0.0, 1.0)

                    edges = np.array(meta_stats["bin_edges"])
                    b_id = np.searchsorted(edges, nv, side="right") - 1
                    b_id = max(0, min(b_id, len(edges) - 2))

                    lower, upper = edges[b_id], edges[b_id + 1]
                    width = max(upper - lower, 1e-9)
                    res = (nv - lower) / width
                    res = np.clip(res, 0.0, 1.0)
                    return b_id, res

                if ctype == "numerical":
                    expert_labels["col_type_ids"].append(0)
                    stats = col_meta["stats"]
                    use_val = val if not pd.isna(val) else stats["mean"]
                    b_id, res = encode_numeric_val(use_val, stats)

                    expert_labels["num_bin"].append(b_id)
                    expert_labels["num_res"].append(res)
                    expert_labels["cat_id"].append(-100)
                    expert_labels["mixed_mask"].append(0.0); expert_labels["mixed_bin"].append(-100); expert_labels["mixed_res"].append(0.0)

                else:  # mixed
                    expert_labels["col_type_ids"].append(2)
                    is_zero_like = pd.isna(val) or (val == 0)
                    mask = 0.0 if is_zero_like else 1.0
                    expert_labels["mixed_mask"].append(mask)

                    if not is_zero_like:
                        b_id, res = encode_numeric_val(val, col_meta["stats"])
                        expert_labels["mixed_bin"].append(b_id)
                        expert_labels["mixed_res"].append(res)
                    else:
                        expert_labels["mixed_bin"].append(-100)
                        expert_labels["mixed_res"].append(0.0)

                    expert_labels["num_bin"].append(-100); expert_labels["num_res"].append(0.0)
                    expert_labels["cat_id"].append(-100)

        return expert_labels


    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)

    def _select_contiguous(
        self,
        start: int,
        length: int,
        new_fingerprint=None,
    ):
        """
        Creates a new dataset with rows from a contiguous slice of data.

        Args:
            start (int): Start index of the slice.
            length (int): Length of the slice to select.

        Returns:
            LLMtgDataset: A new dataset instance from the specified slice.

        Raises:
            DatasetTransformationNotAllowedError: If the dataset has attached indexes.
            ValueError: If the start index or the length is out of range.
        """
        self._validate_selection(start, length)

        indices_table = None
        if self._indices is not None:
            indices_table = self._indices.slice(start, length)

        selected_table = (
            self.to_pandas().iloc[start : start + length]
            if indices_table is None
            else self.to_pandas()
        )
        new_dataset = LLMtgDataset.from_pandas(selected_table)
        new_dataset.set_serializer(self.serializer)
        new_dataset.set_tokenizer(self.tokenizer)
        new_dataset.set_shuffler(self.shuffle)

        return new_dataset

    def _validate_selection(self, start, length):
        """Validates the start index and length for dataset selection."""
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.select` on a dataset with attached indexes is not allowed. Run `.drop_index()` to remove your index, then re-add it."
            )
        if len(self) == 0 or length == 0:
            return
        if start < 0 or start >= len(self) or start + length > len(self):
            raise ValueError(
                "Start index and length are out of range for dataset selection."
            )



@dataclass
class TGCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class Deserializer:
    def __init__(self, serialization_type, column_names, dataset_object=None):
        self.serialization_type = serialization_type
        self.columns = column_names
        self.dataset_object = dataset_object

    def get_deserializer(self):
        if self.serialization_type == "great":
            return self.convert_great
        elif self.serialization_type == "text":
            return self.convert_text
        elif self.serialization_type == "apval":
            return self.convert_apval
        elif self.serialization_type == "list":
            return self.convert_list
        else:
            raise NotImplementedError

    def deserialize(self, text):
        generated = self.get_deserializer()(text, self.columns)
        df_gen = pd.DataFrame(generated)
        # print(df_gen)
        # df_gen.replace("None", None, inplace=True)
        return df_gen

    def convert_great(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": "is", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_text(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": "."}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = (
                    f.strip()
                    .lstrip(others_token["prefix"])
                    .strip()
                    .split(f" {others_token['key_val_sep']} ")
                )
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)
        return generated

    def convert_list(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n"}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")
            # import pdb;pdb.set_trace()

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_apval(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(f" {others_token['key_val_sep']} ")
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data

            keys = features[0].strip().split(others_token["text_sep"])
            if len(features) > 1:
                values = features[1].strip().split(others_token["text_sep"])
            else:
                values = None

            for index, k in enumerate(keys):
                k = k.strip()
                try:
                    v = values[index].strip()
                except:
                    v = None
                # if k in columns and td[k] == "placeholder":
                if k in columns:  # overrites previous values.
                    try:
                        td[k] = v
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated


class Deserializer2:
    def __init__(self, serialization_type, column_names, dataset_object=None):
        self.serialization_type = serialization_type
        self.columns = column_names
        self.dataset_object = dataset_object

    def get_deserializer(self):
        if self.serialization_type == "great":
            return self.convert_great
        elif self.serialization_type == "text":
            return self.convert_text
        elif self.serialization_type == "apval":
            return self.convert_apval
        elif self.serialization_type == "list":
            return self.convert_list
        else:
            raise NotImplementedError

    def deserialize(self, text):
        generated = self.get_deserializer()(text, self.columns)
        df_gen = pd.DataFrame(generated)
        # print(df_gen)
        # df_gen.replace("None", None, inplace=True)
        return df_gen

    def convert_great(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": "is", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                if values[0] in columns and td[values[0]] == "placeholder":
                    # if values[0] in columns: # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_text(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": "The", "key_val_sep": "is", "text_sep": "."}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data
            for f in features:
                values = (
                    f.strip()
                    .lstrip(others_token["prefix"])
                    .strip()
                    .split(f" {others_token['key_val_sep']} ")
                )
                # if values[0] in columns and td[values[0]] == "placeholder":
                if values[0] in columns:  # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)
        return generated

    def convert_list(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": "\n"}

        # Convert text to tabular data
        for t in text:
            features = t.split(others_token["text_sep"])
            td = dict.fromkeys(columns, "placeholder")
            # import pdb;pdb.set_trace()

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(f" {others_token['key_val_sep']} ")
                if values[0] in columns and td[values[0]] == "placeholder":
                    # if values[0] in columns: # overrites previous values.
                    try:
                        td[values[0]] = values[1]
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated

    def convert_apval(self, text, columns) -> pd.DataFrame:
        generated = []

        if self.dataset_object is not None:
            others_token = self.dataset_object.others_token
        else:
            others_token = {"prefix": None, "key_val_sep": ":", "text_sep": ","}

        # Convert text to tabular data
        for t in text:
            features = t.split(f" {others_token['key_val_sep']} ")
            td = dict.fromkeys(columns, "placeholder")

            # Transform all features back to tabular data

            keys = features[0].strip().split(others_token["text_sep"])
            if len(features) > 1:
                values = features[1].strip().split(others_token["text_sep"])
            else:
                values = None

            for index, k in enumerate(keys):
                k = k.strip()
                try:
                    v = values[index].strip()
                except:
                    v = None
                if k in columns and td[k] == "placeholder":
                    # if k in columns: # overrites previous values.
                    try:
                        td[k] = v
                    except IndexError:
                        # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                        pass
            generated.append(td)

        return generated


class GenerateStartTokens:
    TEMPLATES = {
        "key": {"great": "{} is", "list": "{} :", "text": "The {} is", "apval": "{} :"},
        "key_value": {
            "great": "{} is {},",
            "list": "{} : {}\n",
            "text": "The {} is {}.",
            "apval": "{} : {},",
        },
    }

    def __init__(
        self, n_samples, dataset, prompt_template=None, instruction=None, nshot=0
    ):
        self.n_samples = n_samples
        self.dataset = dataset
        self.all_columns = dataset.column_names
        self.prompt_template = prompt_template
        self.instruction = instruction
        self.nshot = nshot

    def get_template(self, tag):
        if self.prompt_template is None:
            prompt_template = self.TEMPLATES[tag].get(self.dataset.serializer, "{}")
        else:
            prompt_template = self.prompt_template
        return prompt_template

    def _pad(self, x, length, pad_value=50256):
        return [pad_value] * (length - len(x)) + x

    def _pad_tokens(self, tokens):
        max_length = len(max(tokens, key=len))
        tokens = [
            self._pad(t, max_length, self.dataset.tokenizer.pad_token_id)
            for t in tokens
        ]
        return tokens

    def get_start_tokens(self, start_prompt="random", start_col=None):
        start_text = self.get_start_text(start_prompt, start_col)

        start_tokens = self._pad_tokens(self.dataset.tokenizer(start_text)["input_ids"])
        return start_tokens

    def get_start_text(self, start_prompt="random", start_col=None):
        if start_prompt == "default":
            start_text = self.start(start_col)

        elif start_prompt == "random":
            start_text = self.random_start()

        elif start_prompt == "categorical":
            start_text = self.categorical_start(start_col)

        elif start_prompt == "continuous":
            start_text = self.continuous_start(start_col)

        elif start_prompt == "categorical_and_random":
            start_text = self.categorical_and_random_start(start_col)

        elif start_prompt == "continuous_and_random":
            start_text = self.continuous_and_random_start(start_col)
        elif start_prompt == "partial":
            start_text = self.partial_start(start_col)

        else:
            if type(start_prompt) == str:
                start_prompt = [start_prompt]

            start_prompt = [i.replace("\\n", "\n") for i in start_prompt]
            start_text = start_prompt

        start_text = self.expand_prompt(start_text)
        return start_text

    def start(self, start_col):
        n_samples = self.n_samples
        if self.dataset.serializer == "apval":
            start_words = []
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            for i in range(n_samples):
                start_word = ", ".join(columns)
                start_words.append(start_word)
        else:
            start_words = [start_col for _ in range(n_samples)]

        prompt_template = self.get_template("key")
        start_text = [prompt_template.format(s) for s in start_words]
        return start_text

    def random_start(self):
        n_samples = self.n_samples

        if self.dataset.serializer == "apval":
            shuffle_idx = list(range(len(self.all_columns)))
            start_words = []
            for i in range(n_samples):
                random.shuffle(shuffle_idx)
                start_word = ", ".join([self.all_columns[j] for j in shuffle_idx])
                start_words.append(start_word)
        else:
            start_words = random.choices(self.all_columns, k=n_samples)

        prompt_template = self.get_template("key")
        start_text = [prompt_template.format(s) for s in start_words]
        return start_text

    def categorical_start(self, start_col):
        n_samples = self.n_samples

        ds = self.dataset.to_pandas()
        
        # cast col as object in case it's not
        ds[start_col] = ds[start_col].astype("object")
        
        metadata = get_metadata(ds)
        
        assert metadata[start_col]["dtype"] == "object", print(f"Start Col: {start_col} should be an object.")
        
        population = metadata[start_col]["categories"]["unique"]
        weights = metadata[start_col]["categories"]["weights"]
        start_words = random.choices(population, weights, k=n_samples)

        if self.dataset.serializer == "apval":
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            start_col = ", ".join(columns)

        prompt_template = self.get_template("key_value")
        start_text = [prompt_template.format(start_col, s) for s in start_words]
        return start_text

    def continuous_start(self, start_col, noise=0.01, decimal_places=5):
        n_samples = self.n_samples

        metadata = get_metadata(self.dataset.to_pandas())
        assert metadata[start_col]["dtype"].startswith("i")

        values = metadata[start_col]["stats"]["list"](start_col)
        start_words = random.choices(values, k=n_samples)

        if self.dataset.serializer == "apval":
            columns = [start_col] + [
                i for i in self.all_columns if i.lower() != start_col.lower()
            ]
            start_col = ", ".join(columns)

        prompt_template = self.get_template("key_value")
        start_text = [prompt_template.format(start_col, s) for s in start_words]

        return start_text

    def categorical_and_random_start(self, start_col):
        start_text1 = self.categorical_start(start_col)

        if self.dataset.serializer == "apval":
            return start_text1

        start_text2 = self.random_start()
        start_text = [i + " " + j for i, j in zip(start_text1, start_text2)]

        return start_text

    def continuous_and_random_start(self, start_col):
        start_text1 = self.continuous_start(start_col)

        if self.dataset.serializer == "apval":
            return start_text1

        start_text2 = self.random_start()
        start_text = [i + " " + j for i, j in zip(start_text1, start_text2)]

        return start_text

    def partial_start(self, df):
        conditions = list(df.apply(self._encode_row_partial, axis=1))
        if self.dataset.serializer == "apval":
            return conditions
        else:
            start_cols = list(df.apply(self._get_random_missing, axis=1))
            self.n_samples = 1
            start_texts = [
                self.get_start_text("default", start_col)[0] for start_col in start_cols
            ]
            prompt_text = [i + " " + j for i, j in zip(conditions, start_texts)]
            return prompt_text

    def _get_random_missing(self, row):
        """Return a random missing column or None if all columns are filled."""
        nans = list(row[pd.isna(row)].index)
        return np.random.choice(nans) if len(nans) > 0 else None

    def _encode_row_partial(self, row, shuffle=True):
        prompt_template = self.get_template("key_value")

        num_cols = len(row.index)
        if not shuffle:
            idx_list = np.arange(num_cols)
        else:
            idx_list = np.random.permutation(num_cols)

        if self.dataset.serializer == "apval":
            keys_values = list(
                zip(
                    *[
                        (row.index[j], str(row[row.index[j]]))
                        for j in idx_list
                        if not pd.isna(row[row.index[j]])
                    ]
                )
            )
            nans = tuple(row[pd.isna(row)].index)
            keys = ", ".join(keys_values[0] + nans)
            values = ", ".join(keys_values[1])
            lists = prompt_template.format(keys, values)

        else:
            lists = " ".join(
                sum(
                    [
                        [prompt_template.format(row.index[i], row[row.index[i]])]
                        if not pd.isna(row[row.index[i]])
                        else []
                        for i in idx_list
                    ],
                    [],
                )
            )
        return lists

    def expand_prompt(self, start_text):
        if self.instruction is not None and self.nshot > 0:
            examples = [
                self.dataset.tokenizer.decode(i["input_ids"])
                for i in self.dataset.__getitems__(list(range(self.nshot)))
            ]
            examples = "\n".join(examples)

            start_text = [
                self.instruction.format(examples) + text for text in start_text
            ]

        elif self.instruction is not None:
            start_text = [self.instruction + text for text in start_text]

        elif self.nshot > 0:
            examples = [
                self.dataset.tokenizer.decode(i["input_ids"])
                for i in self.dataset.__getitems__(list(range(self.nshot)))
            ]
            examples = "\n".join(examples)

            start_text = [examples + "\n" + text for text in start_text]

        return start_text


def postprocess_data(table, metadata=None, dropna=False):
    """
    Post-processes a pandas DataFrame based on provided metadata.

    Parameters:
    table (pandas.DataFrame): The DataFrame to be processed.
    metadata (dict, optional): Metadata containing information about data types and categories for DataFrame columns.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    df = table.copy()

    def _numeric_check(x):
        try:
            return float(x)
        except:
            return np.nan

    if metadata:
        column_names = [col.lower() for col in df.columns]
        assert sorted(column_names) == sorted(
            metadata.keys()
        ), "DataFrame columns and metadata keys must match."
        df.columns = column_names

        for col in column_names:
            col_metadata = metadata[col]
            if col_metadata["dtype"] == "object":
                categories = col_metadata["categories"]["unique"]
                case_function = col_metadata["categories"]["case"]
                df[col] = df[col].fillna("None")
                df[col] = df[col].apply(
                    lambda x: case_function.get(str(x).lower(), str.lower)(x)
                )
                condition = df[col].isin(categories)
                df[col] = df[col].where(condition)
            else:
                df[col] = df[col].apply(_numeric_check)
                try:
                    df[col] = df[col].astype(col_metadata["dtype"])
                except:
                    continue

            if dropna:
                df = df.dropna()
                df[col] = df[col].astype(col_metadata["dtype"])

        # Update column names to match case specified in metadata
        df.columns = [metadata[col]["case"](col) for col in column_names]

    # Replace "None" with NaN
    df = df.replace("None", np.nan)

    if dropna:
        df = df.dropna()

    return df.reset_index(drop=True)


def get_word_case(word):
    """
    Determines the case of a given word and returns an appropriate string method.

    :param word: A string whose case is to be determined.
    :return: A string method corresponding to the detected case.
    """
    word = str(word)
    if word.isupper():
        return str.upper
    elif word.islower():
        return str.lower
    elif word.istitle():
        return str.title
    elif word[0].isupper() and not word.isalpha():
        return str.capitalize
    else:
        return str.lower

def get_metadata(data):
    """
    Generates metadata for Type-Aware Tabular Generation.
    - Restored 'weights' for categorical sampling (with [UNK] mass).
    - Forced inclusion of 'Missing' token if present in data.
    - Safe normalization for constant/small-sample numerical columns.
    """
    metadata = {}
    
    # 全局配置
    SKEW_THRESHOLD = 1.0     
    BIN_K = 100              
    MIXED_THRESHOLD = 0.05   
    TOP_K_CAT = 50

    for col in data.columns:
        col_lower = col.lower()
        col_data = data[col]
        
        metadata[col_lower] = {
            "dtype": str(col_data.dtypes),
            "case": get_word_case(col),
            "type": "unknown" 
        }

        # 1. Categorical
        if col_data.dtypes == object:
            metadata[col_lower]["type"] = "categorical"
            
            # 统一转字符串，填充缺失值为 "Missing"
            clean_series = col_data.fillna("Missing").astype(str)
            
            # 计算频次 (Probabilities)
            value_counts = clean_series.value_counts(normalize=True)
            all_categories = list(value_counts.index)
            
            # --- Top-K 截断策略 ---
            if len(all_categories) > TOP_K_CAT:
                valid_vocab = all_categories[:TOP_K_CAT]
            else:
                valid_vocab = all_categories
            
            # 强制保留 "Missing",防止 NaN 变成 UNK
            if "Missing" in all_categories and "Missing" not in valid_vocab:
                # 策略：直接追加。虽会导致 vocab size = K+1，但保证了语义安全
                valid_vocab.append("Missing")
            
            # 强制 [UNK] 为 ID 0
            final_vocab = ["[UNK]"] + [c for c in valid_vocab if c != "[UNK]"]
            
            # --- 计算 Weights (用于 Generation 采样) ---
            # 1. 计算 valid_vocab 覆盖的总概率质量
            valid_probs = value_counts[final_vocab[1:]] # 排除 [UNK]
            covered_mass = valid_probs.sum()
            
            # 2. [UNK] 的权重 = 1 - 覆盖质量
            unk_mass = max(0.0, 1.0 - covered_mass)
            
            # 3. 组装最终 weights
            final_weights = [unk_mass] + valid_probs.tolist()
            
            # 归一化 weights
            total_w = sum(final_weights)
            if total_w > 0:
                final_weights = [w / total_w for w in final_weights]

            # 映射表
            w2i = {cat: i for i, cat in enumerate(final_vocab)}
            w2i_lower = {str(cat).lower(): i for i, cat in enumerate(final_vocab)}

            metadata[col_lower]["categories"] = {
                "unique": final_vocab,
                "weights": final_weights, # [修复] 加回 weights
                "w2i": w2i,
                "w2i_lower": w2i_lower,
                "unk_id": w2i["[UNK]"],
                "missing_id": w2i.get("Missing", w2i["[UNK]"]), # 方便后续快速查找
                "vocab_size": len(final_vocab)
            }
            
        # 2. Numeric / Mixed
        else:
            # Mixed 判定
            zero_rate = (col_data == 0).mean()
            nan_rate = col_data.isna().mean()
            is_mixed = (zero_rate + nan_rate) > MIXED_THRESHOLD
            
            metadata[col_lower]["type"] = "mixed" if is_mixed else "numerical"
            
            # 提取有效部分 (不包含 NaN, Mixed模式下也不包含 0)
            if is_mixed:
                continuous_part = col_data[(col_data != 0) & (~col_data.isna())]
            else:
                continuous_part = col_data.dropna()
            
            vals = continuous_part.astype(float).to_numpy()

 
            if len(vals) > 0:
                current_min = float(vals.min())
                current_max = float(vals.max())
                current_mean = float(vals.mean())
                
                # Skewness 只有样本够才算，否则 0
                if len(vals) >= 3:
                    current_skew = float(skew(vals, bias=False))
                else:
                    current_skew = 0.0
                
                # Log 策略
                cond1 = abs(current_skew) > SKEW_THRESHOLD
                cond2 = (current_min > 0) and (current_max / current_min > 1000)
                needs_log = cond1 or cond2
                
                # 归一化参数计算 (基于真实数据)
                if needs_log:
                    shift = abs(min(0, current_min)) + 1e-6
                    log_vals = np.log(vals + shift)
                    norm_min = float(log_vals.min())
                    norm_max = float(log_vals.max())
                    meta_stats_shift = shift
                else:
                    norm_min = current_min
                    norm_max = current_max
                    meta_stats_shift = 0.0
                    log_vals = vals # 引用
                
                # 处理常数列 (max == min)
                if norm_max <= norm_min:
                    norm_max = norm_min + 1.0 

                # 计算分箱 (Quantile)
                # 归一化到 [0, 1] 用于计算 quantile
                norm_data = (log_vals - norm_min) / (norm_max - norm_min)
                quantiles = np.linspace(0, 1, BIN_K + 1)
                edges = np.unique(np.quantile(norm_data, quantiles))
                
                # 退化回退
                if len(edges) < (BIN_K // 2):
                    edges = np.linspace(0, 1, BIN_K + 1)
                elif len(edges) < BIN_K + 1:
                    edges = np.linspace(0, 1, BIN_K + 1)
                
                meta_stats = {
                    "min": current_min, "max": current_max, "mean": current_mean,
                    "skewness": current_skew, "needs_log": needs_log,
                    "log_shift": meta_stats_shift if needs_log else 1e-6,
                    "norm_min": norm_min, "norm_max": norm_max,
                    "bin_edges": edges.tolist()
                }

            else:
                #只有真·空列才用默认 0~1
                meta_stats = {
                    "min": 0.0, "max": 1.0, "mean": 0.0,
                    "skewness": 0.0, "needs_log": False,
                    "norm_min": 0.0, "norm_max": 1.0,
                    "bin_edges": np.linspace(0, 1, BIN_K + 1).tolist()
                }

            metadata[col_lower]["stats"] = meta_stats

    return metadata



def convert_tokens_to_text(tokens, tokenizer):
    """Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t, skip_special_tokens=True) for t in tokens]

    # import pdb; pdb.set_trace()
    # print(text_data)
    # Clean text
    # text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    # text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.lstrip("\n") for d in text_data]
    # text_data = [d.lstrip("\\n") for d in text_data]
    text_data = [d.lstrip("\r") for d in text_data]
    # text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def test_dataset():
    from transformers import AutoTokenizer
    import pandas as pd
    import random

    random.seed(42)

    llm = "gpt2"
    # llm = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_prefix_space = False
    # print(tokenizer.add_prefix_space)

    df = pd.read_csv("./data/adult/train.csv")
    great_ds = LLMtgDataset.from_pandas(df)

    great_ds.set_serializer()
    great_ds.set_tokenizer(tokenizer)
    great_ds.set_shuffler(shuffle=True)

    print(great_ds.column_names)

    tokens = [great_ds[i]["input_ids"] for i in range(1)]
    print(tokens)

    decoded_tokens = [
        great_ds.tokenizer.decode(j, skip_special_tokens=True) for j in tokens
    ]
    print(decoded_tokens)

    value = (
        decoded_tokens[0]
        .split(great_ds.others_token["text_sep"])[0]
        .split(great_ds.others_token["key_val_sep"])[1]
        .strip()
    )
    ###
    # This block validates that the tokenized key, values, others
    # matches how it's tokenized as part of a sentence.
    for k, v in great_ds.keys_token_id.items():
        assert v[0] in tokens[0]

    great_ds.values_token_id[value] in tokens[0]
    great_ds.others_token_id["key_val_sep"] in tokens[0]

    ####

    print(great_ds.keys_token_id)
    print(great_ds.tokenizer.encode("isis"))
    print(great_ds.tokenizer.encode("is is"))
    print(great_ds.tokenizer.encode("isis terrorist"))
    print(great_ds.tokenizer.encode("ageage"))
    print(great_ds.tokenizer.encode("age age"))
    print(
        great_ds.tokenizer.decode(
            tokenizer.encode("age is 32, age is 32"), skip_special_tokens=True
        )
    )
    # to ensure that tokens are the same, prepend with a space.
    tokens = great_ds.tokenizer.encode(" sex is Male, sex is Male. sex is ?, sex is NA")
    print(tokens)
    print(
        tokenizer.decode(
            [
                1714,
                318,
                12674,
                11,
                1714,
                318,
                12674,
                13,
                1714,
                318,
                5633,
                11,
                1714,
                318,
                11746,
            ]
        )
    )
    # "sex is Male, sex is Male. sex is?, sex is NA"


def test_data_collator():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader

    llm = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    great_ds = LLMtgDataset.from_pandas(df)
    great_ds.set_serializer()
    great_ds.set_tokenizer(tokenizer)
    great_ds.set_shuffler(shuffle=True)
    dataloader = DataLoader(
        great_ds,
        shuffle=True,
        collate_fn=DataCollator(tokenizer),
        batch_size=5,
    )
    for i in dataloader:
        print(i["input_ids"])
        print(len(i["input_ids"]))
        break


def test_apval_serialize():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader

    llm = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer("apval")
    # ds.set_serializer("list")
    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    print(ds.others_token)
    print(ds.others_token_id)
    out = ds.__getitems__([0, 1])
    print(out)
    for i in out:
        print(ds.tokenizer.decode(i["input_ids"]))
    # print(ds.tokenizer.decode([2479, 11, 670, 4871, 11, 277, 21283, 86, 13655, 11, 3707, 11, 3707, 12, 22510, 11, 29555, 12, 13376, 11, 13755, 11, 2776, 11, 3234, 11, 1714, 11, 3139, 12, 48544, 11, 3139, 12, 22462, 11, 2250, 12, 525, 12, 10464, 11, 6868, 12, 19315, 11, 3739, 1058, 5014, 11, 1812, 12, 9567, 11, 767, 2425, 1433, 11, 347, 9636, 669, 11, 1511, 11, 7236, 12, 30526, 11, 1215, 76, 12, 22902, 605, 11, 1892, 12, 259, 12, 17989, 11, 2635, 11, 12674, 11, 362, 22985, 11, 657, 11, 2319, 11, 1578, 12, 27219, 11, 19841, 1120, 74]))


def test_tabular_data_conversion():
    from transformers import AutoTokenizer
    import pandas as pd
    from torch.utils.data import DataLoader
    import pandas as pd

    # from misc import get_metadata

    llm = "gpt2"
    serialization_type = "apval"
    tokenizer = AutoTokenizer.from_pretrained(llm)
    tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv("./data/adult/train.csv")
    ds = LLMtgDataset.from_pandas(df)
    ds.set_serializer(serialization_type)

    ds.set_tokenizer(tokenizer)
    ds.set_shuffler(shuffle=False)

    out = ds.__getitems__([0, 1])
    output_text = [ds.tokenizer.decode(i["input_ids"]) for i in out]
    print(output_text[0])
    # output_text = ["income is related to income, education, occupation, marital status, education-level, occupation, capital-gain, education-num, marital-status, occupation, capital-loss, capital-gain, education-state, capital-loss, marital-status, capital-loss, capital-gain, education-num, capital-loss, capital-gain, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"]
    # output_text = ["education, occupation, marital-status, education-level, occupation, capital-gain, education-num, marital-status, occupation, capital-loss, capital-gain, education-state, capital-loss, marital-status, capital-loss, capital-gain, education-num, capital-loss, capital-gain, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-gain, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, capital-loss, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"]
    # output_text = ['education, occupation, marital-status, education-num : bachelors, adm-clerical, Never-Married, 13']
    print(output_text)

    deserializer = Deserializer(serialization_type, ds.column_names, ds)
    output_table = deserializer.deserialize(output_text)

    metadata = get_metadata(ds.to_pandas())
    if metadata is not None:
        output_table = postprocess_data(output_table, metadata)

    print(output_table)
    print(ds.to_pandas().loc[[0, 1], :])
    assert output_table.equals(ds.to_pandas().loc[[0, 1], :])

# def test_type_aware_logic_full():
#     print("\n=== Type-Aware Dataset 全功能覆盖测试 (Label + Text Serialization) ===")
#     from transformers import AutoTokenizer
#     import pandas as pd
#     import numpy as np
#     import torch

#     # 1. 构造数据
#     data = {
#         "age": [25.0, 30.0, 35.0, 40.0, 28.0, 32.0],        # Num
#         "income": [1.0, 1.0, 1.0, 1.0, 1.0, 1000000.0],     # Long-tail Num
#         "education": ["PhD", "Bachelors", "Masters", "PhD", "Bachelors", "HS-grad"], # Cat
#         "debt": [0.0, 0.0, 1000.0, np.nan, 500.0, 0.0],     # Mixed (含 NaN)
#         "constant_num": [100.0] * 6,                        # Constant
#     }
#     df = pd.DataFrame(data)

#     # Init Tokenizer & Dataset
#     try:
#         tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
#         if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
#     except:
#         print("Warning: No local GPT2 found, skipping tokenizer dependent checks.")
#         return

#     ds = LLMtgDataset.from_pandas(df)
    
#     # 【关键步骤】必须先 set_serializer 再 set_tokenizer，顺序不能乱
#     ds.set_serializer("great") 
#     ds.set_tokenizer(tokenizer) # 这里会发生 add_special_tokens
#     ds.set_shuffler(shuffle=False)
    
#     # 生成元数据
#     metadata = get_metadata(df)
#     ds.metadata = metadata
    
#     print("\n--- 1. 元数据检查 (Metadata) ---")
#     print(f"[Income] Log: {metadata['income']['stats']['needs_log']} (Expect True)")
#     print(f"[Debt] Type: {metadata['debt']['type']} (Expect mixed)")

#     print("\n--- 2. 行编码检查 (Encoding Labels) ---")
#     item = ds[2] # Row 2: Age=35, Edu=Masters, Debt=1000
#     print(f"[Age=35] Bin: {item['num_bin'][0].item()}")
#     print(f"[Edu=Masters] ID: {item['cat_id'][2].item()}")
    
#     # 检查 Unknown Fallback
#     fake_row = df.iloc[0].copy()
#     fake_row["education"] = "Alien"
#     encoded_alien = ds.encode_row(fake_row, metadata)
#     print(f"[Edu=Alien] ID: {encoded_alien['cat_id'][2]} (Expect 0)")
#     assert encoded_alien['cat_id'][2] == 0

#     print("\n--- 3. 文本序列化检查 (Text Serialization) [重点!] ---")
#     # 我们解码第一行看看长什么样
#     # Row 0: Age=25, Income=1, Edu=PhD, Debt=0, Const=100
#     input_ids = item['input_ids'] # 注意：item 是 ds[2]，不是 Row 0，不过没关系
#     decoded_text = tokenizer.decode(input_ids)
    
#     print(f"\n[Raw Decoded Text]:\n{decoded_text}")
    
#     # 验证关键 Token 是否存在
#     # 注意：Tokenizer decode 可能会在特殊符号前加空格，这取决于 BPE
#     # 我们主要检查核心字符串是否存在
#     assert "age" in decoded_text
#     assert "Masters" in decoded_text # Row 2 是 Masters
    
#     # 【核心检查】 检查 _getitem 是否加了 Type Token 和 EOC
#     # 如果 add_special_tokens 成功，decode 出来应该是 [NUM], [EOC]
#     # 如果没成功，可能会是类似 unk 或者被切碎
#     has_type_token = "[NUM]" in decoded_text or "[CAT]" in decoded_text
#     has_eoc = "[EOC]" in decoded_text
    
#     print(f"\n[Check] Has Type Tokens ([NUM]/[CAT]): {has_type_token}")
#     print(f"[Check] Has EOC Token ([EOC]): {has_eoc}")
    
#     if not has_type_token:
#         print("❌ 警告: 解码文本中没看到 [NUM]/[CAT]，可能是 Tokenizer 没加进去或者 _getitem 逻辑没跑通！")
#     if not has_eoc:
#         print("❌ 警告: 解码文本中没看到 [EOC]，dataset.py 的 _getitem 可能没更新！")

#     # 【核心检查】 检查 NaN 是否变成了 "Missing"
#     # Row 3 (Index 3) 的 Debt 是 NaN
#     item_nan = ds[3]
#     decoded_nan = tokenizer.decode(item_nan['input_ids'])
#     print(f"\n[NaN Row Decoded]:\n{decoded_nan}")
    
#     if "Missing" in decoded_nan:
#         print("✅ Check: NaN -> 'Missing' conversion successful.")
#     else:
#         print("❌ Check: NaN did NOT become 'Missing'. Check _getitem logic.")
#         # 如果变成了 "nan" 说明 _getitem 处理不对
#         if "nan" in decoded_nan.lower():
#             print("   (Found 'nan' instead, which is bad for vocab alignment)")

#     print("\n✅ 测试结束。请人工核对上面的 [Raw Decoded Text] 是否符合预期格式：")
#     print("   Expect: [BOS] [NUM] col: val [EOC] ... [EOS]")

# if __name__ == "__main__":
#     test_type_aware_logic_full()
# if __name__ == "__main__":
#     # test_dataset()
#     # test_data_collator()
#     test_append_serialize()
# test_tabular_data_conversion()
def test_type_aware_logic_full():
    """
    Type-Aware Dataset 完整单元测试
    
    测试内容：
    1. 元数据生成（metadata）的正确性
    2. 行编码（encode_row）的正确性
    3. 文本序列化（_getitem）的正确性，包括 Type Tokens 和 NaN 处理
    4. 各种数据类型的处理（numerical, categorical, mixed）
    """
    import unittest
    from transformers import AutoTokenizer
    import pandas as pd
    import numpy as np
    import torch

    class TestTypeAwareDataset(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            """设置测试数据"""
            # 构造测试数据：包含各种数据类型
            cls.data = {
                "age": [25.0, 30.0, 35.0, 40.0, 28.0, 32.0],        # Num
                "income": [1.0, 1.0, 1.0, 1.0, 1.0, 1000000.0],     # Long-tail Num (需要 log)
                "education": ["PhD", "Bachelors", "Masters", "PhD", "Bachelors", "HS-grad"], # Cat
                "debt": [0.0, 0.0, 1000.0, np.nan, 500.0, 0.0],     # Mixed (含 NaN)
                "constant_num": [100.0] * 6,                        # Constant
            }
            cls.df = pd.DataFrame(cls.data)
            
            # 初始化 Tokenizer
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if cls.tokenizer.pad_token is None:
                    cls.tokenizer.pad_token = cls.tokenizer.eos_token
            except Exception as e:
                raise unittest.SkipTest(f"无法加载 GPT2 tokenizer: {e}")
            
            # 初始化 Dataset
            cls.ds = LLMtgDataset.from_pandas(cls.df)
            cls.ds.set_serializer("great")
            cls.ds.set_tokenizer(cls.tokenizer)
            cls.ds.set_shuffler(shuffle=False)  # 关闭 shuffle 以确保列顺序一致
            
            # 生成元数据
            cls.metadata = get_metadata(cls.df)
            cls.ds.metadata = cls.metadata

        def test_01_metadata_generation(self):
            """测试元数据生成"""
            # 检查 income 列需要 log 变换（长尾分布）
            self.assertTrue(
                self.metadata['income']['stats']['needs_log'],
                "income 列应该需要 log 变换（长尾分布）"
            )
            
            # 检查 debt 列被识别为 mixed 类型
            self.assertEqual(
                self.metadata['debt']['type'],
                "mixed",
                "debt 列应该被识别为 mixed 类型（包含 NaN 和 0）"
            )
            
            # 检查 age 列被识别为 numerical
            self.assertEqual(
                self.metadata['age']['type'],
                "numerical",
                "age 列应该被识别为 numerical 类型"
            )
            
            # 检查 education 列被识别为 categorical
            self.assertEqual(
                self.metadata['education']['type'],
                "categorical",
                "education 列应该被识别为 categorical 类型"
            )
            
            # 检查 categorical 列有 categories 信息
            self.assertIn('categories', self.metadata['education'])
            self.assertIn('vocab_size', self.metadata['education']['categories'])
            self.assertIn('missing_id', self.metadata['education']['categories'])

        def test_02_encode_row_numerical(self):
            """测试数值型列的编码"""
            row_idx = 2  # Row 2: age=35.0
            row_data = self.df.iloc[row_idx]
            
            # 设置 shuffle_idx（因为 encode_row 需要它）
            if not hasattr(self.ds, 'shuffle_idx'):
                self.ds.shuffle_idx = list(range(len(self.ds.column_names)))
            
            # 获取列索引（因为 shuffle=False，列顺序应该不变）
            age_col_idx = self.ds.column_names.index('age')
            
            # 编码该行
            encoded = self.ds.encode_row(row_data, self.metadata)
            
            # 检查数据结构
            self.assertIn('num_bin', encoded)
            self.assertIn('num_res', encoded)
            self.assertIn('col_type_ids', encoded)
            
            # 检查 age 列的编码（在 shuffle_idx 中的位置）
            age_pos = self.ds.shuffle_idx.index(age_col_idx)
            self.assertEqual(
                encoded['col_type_ids'][age_pos],
                0,
                "数值型列的 col_type_ids 应该为 0"
            )
            
            # 检查 num_bin 和 num_res 在合理范围内
            age_bin = encoded['num_bin'][age_pos]
            age_res = encoded['num_res'][age_pos]
            self.assertGreaterEqual(age_bin, 0, "bin index 应该 >= 0")
            self.assertLess(age_bin, 100, "bin index 应该 < 100 (默认 BIN_K)")
            self.assertGreaterEqual(age_res, 0.0, "residual 应该 >= 0.0")
            self.assertLessEqual(age_res, 1.0, "residual 应该 <= 1.0")

        def test_03_encode_row_categorical(self):
            """测试分类型列的编码"""
            row_idx = 2  # Row 2: education="Masters"
            row_data = self.df.iloc[row_idx]
            
            # 设置 shuffle_idx（因为 encode_row 需要它）
            if not hasattr(self.ds, 'shuffle_idx'):
                self.ds.shuffle_idx = list(range(len(self.ds.column_names)))
            
            # 获取列索引
            edu_col_idx = self.ds.column_names.index('education')
            edu_pos = self.ds.shuffle_idx.index(edu_col_idx)
            
            # 编码该行
            encoded = self.ds.encode_row(row_data, self.metadata)
            
            # 检查 education 列的编码
            self.assertEqual(
                encoded['col_type_ids'][edu_pos],
                1,
                "分类型列的 col_type_ids 应该为 1"
            )
            
            # 检查 cat_id 在词汇表范围内
            edu_cat_id = encoded['cat_id'][edu_pos]
            vocab_size = self.metadata['education']['categories']['vocab_size']
            self.assertGreaterEqual(edu_cat_id, 0, "cat_id 应该 >= 0")
            self.assertLess(edu_cat_id, vocab_size, f"cat_id 应该 < vocab_size ({vocab_size})")
            
            # 验证 "Masters" 应该能找到对应的 ID（不是 UNK）
            cats = self.metadata['education']['categories']
            masters_in_vocab = "Masters" in cats['unique']
            if masters_in_vocab:
                self.assertNotEqual(
                    edu_cat_id,
                    cats['unk_id'],
                    "Masters 应该在词汇表中，不应该被编码为 UNK"
                )

        def test_04_encode_row_unknown_category(self):
            """测试未知分类值的回退逻辑"""
            fake_row = self.df.iloc[0].copy()
            fake_row["education"] = "Alien"  # 不在词汇表中的值
            
            # 设置 shuffle_idx（因为 encode_row 需要它）
            if not hasattr(self.ds, 'shuffle_idx'):
                self.ds.shuffle_idx = list(range(len(self.ds.column_names)))
            
            encoded = self.ds.encode_row(fake_row, self.metadata)
            
            # 获取 education 列的位置
            edu_col_idx = self.ds.column_names.index('education')
            edu_pos = self.ds.shuffle_idx.index(edu_col_idx)
            
            # 应该被编码为 UNK (ID=0)
            unk_id = self.metadata['education']['categories']['unk_id']
            self.assertEqual(
                encoded['cat_id'][edu_pos],
                unk_id,
                "未知的分类值应该被编码为 UNK"
            )

        def test_05_encode_row_mixed(self):
            """测试混合型列的编码"""
            row_idx = 2  # Row 2: debt=1000.0 (非零，应该被编码)
            row_data = self.df.iloc[row_idx]
            
            # 设置 shuffle_idx（因为 encode_row 需要它）
            if not hasattr(self.ds, 'shuffle_idx'):
                self.ds.shuffle_idx = list(range(len(self.ds.column_names)))
            
            debt_col_idx = self.ds.column_names.index('debt')
            debt_pos = self.ds.shuffle_idx.index(debt_col_idx)
            
            encoded = self.ds.encode_row(row_data, self.metadata)
            
            # 检查 mixed 类型的标记
            self.assertEqual(
                encoded['col_type_ids'][debt_pos],
                2,
                "混合型列的 col_type_ids 应该为 2"
            )
            
            # 非零值应该有 mask=1.0
            self.assertEqual(
                encoded['mixed_mask'][debt_pos],
                1.0,
                "非零的混合型值应该有 mask=1.0"
            )
            
            # 应该有有效的 bin 和 res
            self.assertNotEqual(
                encoded['mixed_bin'][debt_pos],
                -100,
                "非零的混合型值应该有有效的 bin"
            )

        def test_06_encode_row_nan_handling(self):
            """测试 NaN 值的处理"""
            row_idx = 3  # Row 3: debt=np.nan
            row_data = self.df.iloc[row_idx]
            
            # 设置 shuffle_idx（因为 encode_row 需要它）
            if not hasattr(self.ds, 'shuffle_idx'):
                self.ds.shuffle_idx = list(range(len(self.ds.column_names)))
            
            debt_col_idx = self.ds.column_names.index('debt')
            debt_pos = self.ds.shuffle_idx.index(debt_col_idx)
            
            encoded = self.ds.encode_row(row_data, self.metadata)
            
            # NaN 在 mixed 类型中应该有 mask=0.0
            self.assertEqual(
                encoded['mixed_mask'][debt_pos],
                0.0,
                "NaN 的混合型值应该有 mask=0.0"
            )
            
            # 应该有填充值
            self.assertEqual(
                encoded['mixed_bin'][debt_pos],
                -100,
                "NaN 的混合型值应该有 bin=-100"
            )

        def test_07_text_serialization_structure(self):
            """测试文本序列化的结构"""
            item = self.ds[2]  # 获取第 3 行
            
            # 检查返回的数据结构
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            self.assertIn('num_bin', item)
            self.assertIn('cat_id', item)
            self.assertIn('col_type_ids', item)
            
            # 检查类型（应该是 tensor）
            self.assertIsInstance(item['input_ids'], torch.Tensor)
            self.assertIsInstance(item['num_bin'], torch.Tensor)
            self.assertIsInstance(item['cat_id'], torch.Tensor)

        def test_08_text_serialization_type_tokens(self):
            """A/B/C: 强验证 Type Tokens + EOC + col_type_ids 对齐（shuffle=False 更可靠）"""
            item = self.ds[2]
            input_ids = item["input_ids"].tolist()

            # ---- 基础：包含列名 ----
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            decoded_lower = decoded_text.lower()
            self.assertTrue(
                any(col.lower() in decoded_lower for col in self.ds.column_names),
                "解码文本应该包含至少一个列名"
            )

            # ---- A) 强制 special tokens 必须是单 token ----
            # 这些 token 用于对齐/路由，必须可定位
            for tok in ["[BOS]", "[EOS]", "[NUM]", "[CAT]", "[MIX]", "[EOC]", "[UNK]"]:
                ids = self.tokenizer.encode(tok, add_special_tokens=False)
                self.assertEqual(
                    len(ids), 1,
                    f"{tok} 必须被 tokenizer 作为单独 token 编码，否则无法稳定对齐。got={ids}"
                )

            # ---- B) 强制每列都有一个 type token + 一个 EOC ----
            # 先拿 token id（必须在 vocab 里）
            num_id = self.tokenizer.convert_tokens_to_ids("[NUM]")
            cat_id = self.tokenizer.convert_tokens_to_ids("[CAT]")
            mix_id = self.tokenizer.convert_tokens_to_ids("[MIX]")
            eoc_id = self.tokenizer.convert_tokens_to_ids("[EOC]")

            # sanity: token id 不应为 unk 或 -1
            self.assertIsInstance(num_id, int)
            self.assertIsInstance(cat_id, int)
            self.assertIsInstance(mix_id, int)
            self.assertIsInstance(eoc_id, int)
            self.assertGreaterEqual(num_id, 0)
            self.assertGreaterEqual(cat_id, 0)
            self.assertGreaterEqual(mix_id, 0)
            self.assertGreaterEqual(eoc_id, 0)

            C = len(self.ds.column_names)
            type_count = sum(1 for x in input_ids if x in (num_id, cat_id, mix_id))
            eoc_count = sum(1 for x in input_ids if x == eoc_id)

            self.assertEqual(
                type_count, C,
                f"每列应当对应一个 type token（[NUM]/[CAT]/[MIX]）。expected={C}, got={type_count}. text={decoded_text}"
            )
            self.assertEqual(
                eoc_count, C,
                f"每列应当以 [EOC] 结束。expected={C}, got={eoc_count}. text={decoded_text}"
            )

            # ---- C) 强制 type token 的数量与 col_type_ids 对齐 ----
            # col_type_ids: 0 num / 1 cat / 2 mixed
            col_type_ids = item["col_type_ids"].tolist()
            self.assertEqual(len(col_type_ids), C, "col_type_ids 长度必须等于列数")

            exp_num = sum(1 for t in col_type_ids if t == 0)
            exp_cat = sum(1 for t in col_type_ids if t == 1)
            exp_mix = sum(1 for t in col_type_ids if t == 2)

            got_num = sum(1 for x in input_ids if x == num_id)
            got_cat = sum(1 for x in input_ids if x == cat_id)
            got_mix = sum(1 for x in input_ids if x == mix_id)

            self.assertEqual(got_num, exp_num, f"[NUM] 数量应等于 numerical 列数，expected={exp_num}, got={got_num}")
            self.assertEqual(got_cat, exp_cat, f"[CAT] 数量应等于 categorical 列数，expected={exp_cat}, got={got_cat}")
            self.assertEqual(got_mix, exp_mix, f"[MIX] 数量应等于 mixed 列数，expected={exp_mix}, got={got_mix}")

            # ---- （可选）额外：检查 expert_token_idxs 的长度与 type token 位置一致 ----
            # 如果你已经在 dataset 返回了 expert_token_idxs，则可以强校验
            if "expert_token_idxs" in item:
                idxs = item["expert_token_idxs"].tolist()
                self.assertEqual(len(idxs), C, "expert_token_idxs 长度必须等于列数")
                # 非 -1 的位置应该落在 type token 上
                for pos in idxs:
                    if pos == -1:
                        continue
                    self.assertTrue(
                        input_ids[pos] in (num_id, cat_id, mix_id),
                        f"expert_token_idxs 指向的位置必须是 type token。pos={pos}, token_id={input_ids[pos]}, text={decoded_text}"
                    )

            # ---- D) embeddings resize 的检查不应放在 dataset 单测里 ----
            # 这一步要在训练脚本中做：
            # assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)


        def test_09_text_serialization_nan_to_missing(self):
            """测试 NaN 值是否被转换为 "Missing" """
            item_nan = self.ds[3]  # Row 3 的 debt 是 NaN
            decoded_text = self.tokenizer.decode(item_nan['input_ids'])
            
            # 检查是否包含 "Missing"（或者 tokenizer 的特殊形式）
            # 由于 tokenizer 可能将 "Missing" 切分，我们检查相关关键词
            has_missing = (
                "Missing" in decoded_text or 
                "missing" in decoded_text.lower() or
                "nan" not in decoded_text.lower()  # 至少不应该直接出现 "nan"
            )
            
            self.assertTrue(
                has_missing,
                f"NaN 值应该被转换为 'Missing'。实际文本: {decoded_text}"
            )

        def test_10_column_order_consistency(self):
            """测试列顺序的一致性"""
            # 获取同一行的编码（shuffle=False 应该保持一致）
            item1 = self.ds[0]
            item2 = self.ds[0]  # 再次获取同一行
            
            # 列顺序应该一致
            self.assertTrue(
                torch.equal(item1['col_type_ids'], item2['col_type_ids']),
                "同一行的编码应该一致（shuffle=False）"
            )

        def test_11_expert_labels_length(self):
            """测试专家标签的长度与列数一致"""
            item = self.ds[0]
            num_columns = len(self.ds.column_names)
            
            self.assertEqual(
                len(item['col_type_ids']),
                num_columns,
                f"col_type_ids 的长度应该等于列数 ({num_columns})"
            )
            self.assertEqual(
                len(item['num_bin']),
                num_columns,
                f"num_bin 的长度应该等于列数 ({num_columns})"
            )
            self.assertEqual(
                len(item['cat_id']),
                num_columns,
                f"cat_id 的长度应该等于列数 ({num_columns})"
            )

    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTypeAwareDataset)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

if __name__ == "__main__":
    test_type_aware_logic_full()