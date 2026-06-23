#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CATEGORY_RANGES, CATEGORY_RESOLUTIONS
from data import WatermarkedDataset, verify_dataset_root
from model_utils import empty_prompt_embeddings, load_training_stack, save_lora


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=Path("."))
    p.add_argument("--run-dir", type=Path, default=Path("wm_lora_runs"))
    p.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    p.add_argument("--category", choices=[*CATEGORY_RANGES, "all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--max-timestep",
        type=int,
        default=150,
        help="Train only on low-noise diffusion timesteps [0, max-timestep).",
    )
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(args: argparse.Namespace, category: str) -> None:
    device = torch.device(args.device)
    resolution = CATEGORY_RESOLUTIONS[category]
    source_dir = args.dataset / "watermarked_sources" / category
    output_dir = args.run_dir / "adapters" / category
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WatermarkedDataset(source_dir, resolution, repeats=args.repeats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    (
        pipe,
        vae,
        unet,
        text_encoder,
        tokenizer,
        scheduler,
        trainable,
    ) = load_training_stack(
        args.model_id, device, args.precision, args.rank
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_amp = device.type == "cuda" and args.precision != "no"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and args.precision == "fp16")
    )

    metadata = {
        "category": category,
        "resolution": resolution,
        "model_id": args.model_id,
        "rank": args.rank,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "max_timestep": args.max_timestep,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))

    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    losses = []

    progress = tqdm(range(1, args.steps + 1), desc=f"train {category}")
    for step in progress:
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)

        images = images.to(device, non_blocking=True)

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            prompt_embeds = empty_prompt_embeddings(
                tokenizer, text_encoder, device, images.shape[0]
            )

        noise = torch.randn_like(latents)
        max_t = min(args.max_timestep, scheduler.config.num_train_timesteps)
        timesteps = torch.randint(
            0, max_t, (latents.shape[0],), device=device, dtype=torch.long
        )
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            prediction = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
            loss = F.mse_loss(prediction.float(), noise.float())
            scaled_loss = loss / args.grad_accum

        scaler.scale(scaled_loss).backward()

        if step % args.grad_accum == 0 or step == args.steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.5f}")

        if step % args.save_every == 0:
            save_lora(unet, output_dir / f"checkpoint-{step}")

    save_lora(unet, output_dir / "final")
    np.save(output_dir / "losses.npy", np.asarray(losses, dtype=np.float32))
    print(f"Saved {category} adapter to {output_dir / 'final'}")

    del pipe, vae, unet, text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()
    verify_dataset_root(args.dataset)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    categories = list(CATEGORY_RANGES) if args.category == "all" else [args.category]
    for category in categories:
        train_one(args, category)


if __name__ == "__main__":
    main()
