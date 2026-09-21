# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import json
import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str = "auto",
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        from transformers import AutoConfig, AutoTokenizer
        from peft import PeftModel
        from huggingface_hub import snapshot_download

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")
        dl_kwargs = {}
        if cache_dir:
            dl_kwargs["cache_dir"] = cache_dir

        if "TEXT_ENCODERS_DIR" in os.environ:
            mntp_local = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            supervised_local = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)
        else:
            mntp_local = snapshot_download(base_model_name_or_path, **dl_kwargs)
            supervised_local = snapshot_download(peft_model_name_or_path, **dl_kwargs)

        # Find the true LLaMA base from the MNTP adapter config
        with open(os.path.join(mntp_local, "adapter_config.json")) as f:
            adapter_cfg = json.load(f)
        llama_model_id = adapter_cfg["base_model_name_or_path"]  # meta-llama/Meta-Llama-3-8B-Instruct
        llama_local = snapshot_download(llama_model_id, **dl_kwargs)

        # Get the bidirectional LLaMA model class
        llama_config = AutoConfig.from_pretrained(llama_local)
        model_class = LLM2Vec._get_model_class(
            llama_config.__class__.__name__, enable_bidirectional=True
        )

        # Step 1: Load the bare LLaMA base to CPU in bfloat16.
        # bitsandbytes NF4/FP4/INT8 CUDA kernels crash (0xC0000005) on SM 12.0 (RTX 5080, Blackwell).
        # device_map=anything also crashes (accelerate queries CUDA even for CPU target on SM 12.0).
        # No device_map + low_cpu_mem_usage=True → pure PyTorch CPU load, no accelerate, no CUDA init.
        # Text encoding runs once per generation; CPU speed is acceptable.
        base_model = model_class.from_pretrained(
            llama_local,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # Step 2: Apply MNTP LoRA adapter via PeftModel (not model.load_adapter)
        model = PeftModel.from_pretrained(base_model, mntp_local)

        # Step 3: Apply supervised LoRA adapter on top
        model = PeftModel.from_pretrained(model, supervised_local)

        # Load tokenizer from MNTP (has correct padding config for LLM2Vec)
        tokenizer = AutoTokenizer.from_pretrained(mntp_local)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Read llm2vec_config.json if present
        llm2vec_config: dict = {}
        for cfg_dir in (supervised_local, mntp_local):
            cfg_path = os.path.join(cfg_dir, "llm2vec_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    llm2vec_config = json.load(f)
                break

        self.model = LLM2Vec(
            model=model,
            tokenizer=tokenizer,
            pooling_mode=llm2vec_config.get("pooling_mode", "mean"),
            max_length=llm2vec_config.get("max_length", 512),
            doc_max_length=llm2vec_config.get("doc_max_length", 400),
            skip_instruction=llm2vec_config.get("skip_instruction", True),
        )

        # Model lives on CPU; device_map="cuda:*" crashes on SM 12.0 (Blackwell bitsandbytes issue)
        self._device = "cpu"

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self._device = str(device) if not isinstance(device, str) else device
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(
                text,
                batch_size=1,
                show_progress_bar=False,
                device=self._device,
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self._device)
        return encoded_text, lengths
