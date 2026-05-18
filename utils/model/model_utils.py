# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from transformers import (
    AutoConfig,
)
import os
from huggingface_hub import snapshot_download
from transformers.integrations import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0

    name = os.path.basename(model_name_or_path.rstrip("/"))
    low = name.lower()
    trust_remote = low.startswith("llada") or low.startswith("dream")

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=trust_remote,  # LLaDA 必须项
        torch_dtype=torch.bfloat16  # <--- 新增建议：显式指定精度，避免默认 float32 爆显存
    )

    if tokenizer.eos_token_id is not None:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    # Resize embedding 也是为了适配可能增加的 token，保留即可
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    return model
