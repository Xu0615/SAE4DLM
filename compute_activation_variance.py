#!/usr/bin/env python3
"""
Issue #25: Compute variance of SAE feature activations for masked tokens
across diffusion timesteps.

Produces a heatmap per layer (L1, L10, L23) with:
  - x-axis: denoising timestep (mask probability p)
  - y-axis: SAE features (sorted by peak timestep)
  - color:  variance of activation values at masked positions

Usage:
    python compute_activation_variance.py \
        --sae_root_dir /path/to/saes/saes_mask_Dream-org_Dream-v0-Base-7B_top_k \
        --output_dir ./variance_results \
        --num_sequences 5000000 \
        --batch_size 8
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Add repo paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dlm_order"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train_dlm_sae", "dictionary_learning"))

from sae_utils import load_saes, resolve_layers_container


def parse_args():
    parser = argparse.ArgumentParser(description="Compute activation variance for issue #25")
    parser.add_argument("--sae_root_dir", type=str, required=True,
                        help="Path to SAE checkpoints")
    parser.add_argument("--output_dir", type=str, default="./variance_results")
    parser.add_argument("--model_name", type=str, default="Dream-org/Dream-v0-Base-7B")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 23])
    parser.add_argument("--trainer_name", type=str, default="trainer_0")
    parser.add_argument("--num_sequences", type=int, default=5_000_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--num_timesteps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class WelfordVarianceTracker:
    """
    Welford's online algorithm for computing per-feature variance at each timestep.
    Avoids storing all activations in memory.
    """
    def __init__(self, num_features: int, num_timesteps: int):
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.count = np.zeros((num_timesteps,), dtype=np.float64)
        self.mean = np.zeros((num_timesteps, num_features), dtype=np.float64)
        self.m2 = np.zeros((num_timesteps, num_features), dtype=np.float64)

    def update_batch(self, timestep_idx: int, activations: np.ndarray):
        """
        Update with a batch of activations.
        activations: [N, num_features] - SAE feature activations for masked tokens
        """
        for i in range(activations.shape[0]):
            x = activations[i]
            self.count[timestep_idx] += 1
            n = self.count[timestep_idx]
            delta = x - self.mean[timestep_idx]
            self.mean[timestep_idx] += delta / n
            delta2 = x - self.mean[timestep_idx]
            self.m2[timestep_idx] += delta * delta2

    def variance(self) -> np.ndarray:
        """Returns [num_timesteps, num_features] variance array."""
        var = np.zeros_like(self.m2)
        for t in range(self.num_timesteps):
            if self.count[t] > 1:
                var[t] = self.m2[t] / (self.count[t] - 1)
        return var


class MultiLayerActivationCapture:
    """
    Captures hidden states from multiple layers in a single forward pass
    using hooks. Unlike LayerCaptureManager, this captures ALL batch items.
    """
    def __init__(self, model: nn.Module, layers: list):
        self.model = model
        self.layers = layers
        self.layers_container = resolve_layers_container(model)
        self.captured = {}
        self.handles = []

        for layer in layers:
            handle = self.layers_container[layer].register_forward_hook(
                self._make_hook(layer)
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, (tuple, list)):
                hidden = output[0]
            else:
                hidden = output
            if torch.is_tensor(hidden) and hidden.ndim == 3:
                self.captured[layer_idx] = hidden.detach()
        return hook_fn

    def clear(self):
        self.captured = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_data_generator(tokenizer, context_length, seed=42):
    """Load FineWeb dataset as streaming generator."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    for example in dataset:
        text = example.get("text", "")
        if text and len(text.strip()) >= 50:
            yield text


@torch.inference_mode()
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discrete timesteps: mask probabilities from 0.05 to 0.95
    timesteps = np.linspace(0.05, 0.95, args.num_timesteps)
    print(f"Timesteps (mask probabilities): {[f'{t:.3f}' for t in timesteps]}")

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dream models require AutoModel (not AutoModelForCausalLM)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device).eval()
    model.eval()

    # Get mask token ID
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    print(f"Mask token ID: {mask_token_id}")

    # Load SAEs for layers 1, 10, 23 (trainer_0)
    print(f"Loading SAEs for layers {args.layers} from {args.sae_root_dir}")
    saes = load_saes(
        sae_root_dir=args.sae_root_dir,
        layers=args.layers,
        device=args.device,
        trainer_name=args.trainer_name,
    )

    # Get number of features
    num_features = saes[args.layers[0]].W_dec.shape[0]
    print(f"Number of SAE features: {num_features}")

    # Initialize variance trackers per layer
    trackers = {layer: WelfordVarianceTracker(num_features, args.num_timesteps)
                for layer in args.layers}

    # Set up multi-layer capture hooks
    capture = MultiLayerActivationCapture(model, args.layers)

    # Data generator
    data_gen = get_data_generator(tokenizer, args.context_length, seed=args.seed)

    # Main loop
    total_sequences = 0
    pbar = tqdm(total=args.num_sequences, desc="Processing sequences")

    try:
        batch_texts = []
        for text in data_gen:
            batch_texts.append(text)

            if len(batch_texts) < args.batch_size:
                continue

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=args.context_length,
                padding=True,
                truncation=True,
            ).to(args.device)

            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            B, T = input_ids.shape

            # For each timestep
            for t_idx, p in enumerate(timesteps):
                # Build mask: True = position will be masked
                rand = torch.rand(B, T, device=args.device)
                mask = (rand < p) & attention_mask.bool()
                mask[:, 0] = False  # keep first token

                # Apply masking
                masked_ids = input_ids.clone()
                masked_ids[mask] = mask_token_id

                # Forward pass (captures hidden states at all layers via hooks)
                # Dream models work best without attention_mask passed explicitly
                capture.clear()
                model(input_ids=masked_ids)

                # Process each layer
                for layer in args.layers:
                    if layer not in capture.captured:
                        continue

                    hidden_BxTxD = capture.captured[layer]  # [B, T, D]

                    # Extract activations at masked positions across all batch items
                    masked_hidden = hidden_BxTxD[mask]  # [num_masked_total, D]

                    if masked_hidden.shape[0] == 0:
                        continue

                    # Encode with SAE
                    sae = saes[layer]
                    feature_acts = sae.encode(masked_hidden)  # [num_masked_total, F]
                    feature_acts = feature_acts.to(torch.float32).cpu().numpy()

                    # Update variance tracker
                    trackers[layer].update_batch(t_idx, feature_acts)

            total_sequences += len(batch_texts)
            pbar.update(len(batch_texts))
            batch_texts = []

            if total_sequences >= args.num_sequences:
                break

            # Periodic checkpoint every 10k sequences
            if total_sequences % 10_000 == 0:
                save_results(args, trackers, timesteps, total_sequences)

    finally:
        capture.close()
        pbar.close()

    # Final save
    save_results(args, trackers, timesteps, total_sequences)
    print(f"\nDone! Processed {total_sequences} sequences.")
    print(f"Results saved to {args.output_dir}")


def save_results(args, trackers, timesteps, total_sequences):
    """Save variance data and generate heatmaps."""
    for layer in args.layers:
        variance = trackers[layer].variance()
        np.save(os.path.join(args.output_dir, f"variance_L{layer}.npy"), variance)
        np.save(os.path.join(args.output_dir, f"counts_L{layer}.npy"), trackers[layer].count)
        np.save(os.path.join(args.output_dir, f"mean_L{layer}.npy"), trackers[layer].mean)
        plot_heatmap(variance, timesteps, layer, args.output_dir, total_sequences)

    meta = {
        "layers": args.layers,
        "timesteps": timesteps.tolist(),
        "total_sequences": total_sequences,
        "model_name": args.model_name,
        "sae_root_dir": args.sae_root_dir,
        "trainer_name": args.trainer_name,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def plot_heatmap(variance, timesteps, layer, output_dir, total_sequences):
    """
    Heatmap: features (y) vs denoising steps (x), color = variance.
    Features sorted by peak timestep.
    """
    var_FxT = variance.T  # [F, T]

    # Filter to active features
    max_var = var_FxT.max(axis=1)
    active_mask = max_var > 1e-8
    active_var = var_FxT[active_mask]

    if active_var.shape[0] == 0:
        print(f"  Layer {layer}: No active features, skipping plot.")
        return

    # Sort by peak timestep
    peak_timestep = np.argmax(active_var, axis=1)
    sort_idx = np.argsort(peak_timestep)
    sorted_var = active_var[sort_idx]

    # Log scale for visibility
    sorted_var_log = np.log1p(sorted_var)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    im = ax.imshow(sorted_var_log, aspect="auto", cmap="viridis", interpolation="nearest")

    num_ticks = min(len(timesteps), 10)
    tick_indices = np.linspace(0, len(timesteps) - 1, num_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{timesteps[i]:.2f}" for i in tick_indices], fontsize=9)
    ax.set_xlabel("Mask Probability (Denoising Timestep)", fontsize=12)
    ax.set_ylabel(f"SAE Features (sorted by peak timestep, {active_var.shape[0]} active)", fontsize=12)
    ax.set_title(f"Layer {layer} - Feature Activation Variance at Masked Positions\n"
                 f"({total_sequences:,} sequences)", fontsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log(1 + variance)", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_L{layer}.png"))
    plt.close()
    print(f"  Saved heatmap for Layer {layer} ({active_var.shape[0]} active features)")


if __name__ == "__main__":
    main()
