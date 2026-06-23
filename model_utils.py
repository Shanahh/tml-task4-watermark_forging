from __future__ import annotations

from pathlib import Path

import torch
from diffusers import (
    DDPMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def resolve_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type != "cuda" or precision == "no":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    return torch.float16


def load_training_stack(
    model_id: str,
    device: torch.device,
    precision: str,
    rank: int,
):
    dtype = resolve_dtype(device, precision)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    adapter_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGETS,
    )
    unet.add_adapter(adapter_config, adapter_name="wm")

    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No LoRA parameters were made trainable.")

    return pipe, vae, unet, text_encoder, tokenizer, scheduler, trainable


@torch.no_grad()
def empty_prompt_embeddings(tokenizer, text_encoder, device, batch_size: int):
    tokens = tokenizer(
        [""] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


def save_lora(unet, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(unet, adapter_name="wm")
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(directory),
        unet_lora_layers=state,
        safe_serialization=True,
    )


def load_img2img(
    model_id: str,
    adapter_dir: Path,
    device: torch.device,
    precision: str,
    adapter_scale: float,
):
    dtype = resolve_dtype(device, precision)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(adapter_dir), adapter_name="wm")
    pipe.set_adapters("wm", adapter_weights=adapter_scale)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
