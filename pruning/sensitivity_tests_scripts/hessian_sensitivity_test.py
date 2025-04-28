"""
Whisper Component Sensitivity Analysis for MPS (Apple Silicon)
============================================================

This module analyzes the parameter sensitivity of Whisper speech recognition models,
comparing encoder vs decoder component importance on both test.clean and test.other datasets.

This version uses a Fisher Information Matrix approach to approximate Hessian-based sensitivity.
Specially optimized for MPS (Apple Silicon) devices with fallbacks and memory management.

Usage:
    python whisper_component_sensitivity_fisher.py --model openai/whisper-small --samples 500
"""

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

# Third-party imports
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor

# Configure MPS for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for unsupported ops

# Create results directory
RESULTS_DIR = "whisper_sensitivity_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    """Determines the appropriate device for computation"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_librispeech(num_samples: Optional[int] = None, split: str = "test.clean") -> Dataset:
    """
    Load LibriSpeech dataset for evaluation.

    Args:
        num_samples: Number of samples to load (None for entire dataset)
        split: Dataset split to use ("test.clean" or "test.other")

    Returns:
        Dataset object containing audio samples
    """
    if num_samples:
        # Stream partial dataset
        stream_dataset = datasets.load_dataset(
            "librispeech_asr", split=split, streaming=True, trust_remote_code=True
        )
        dataset = Dataset.from_dict(
            {
                k: [sample[k] for sample in list(stream_dataset.take(num_samples))]
                for k in next(iter(stream_dataset)).keys()
            }
        )
    else:
        # Load full dataset
        dataset = datasets.load_dataset("librispeech_asr", split=split)

    total_duration_seconds = sum(
        len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] for sample in dataset
    )
    total_hours = total_duration_seconds / 3600

    print(f"Loaded {len(dataset)} samples from LibriSpeech {split}")
    print(f"Total audio duration: {total_hours:.4f} hours")
    return dataset


def clear_memory():
    """Clear cached memory to avoid OOM errors during processing"""
    gc.collect()

    if torch.backends.mps.is_available():
        # For MPS, we can't directly clear cache like CUDA
        # But we can force garbage collection
        gc.collect()
        try:
            torch.mps.empty_cache()  # This might not work in all PyTorch versions
        except Exception:
            pass  # Ignore if not available
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()


def create_layer_sensitivity_chart(
    clean_results: Dict,
    other_results: Dict,
    output_dir: str,
    model_info: Optional[Dict] = None,
    scaling_factor: float = 100.0,  # Fixed scaling factor instead of normalization
) -> None:
    """
    Create a layer-by-layer sensitivity chart for encoder and decoder layers.
    Shows importance across layers 1-12 for both components.
    Uses a fixed scaling factor to make values more readable.

    Args:
        clean_results: Results from compute_fisher_sensitivity for test.clean
        other_results: Results from compute_fisher_sensitivity for test.other
        output_dir: Directory to save the plot
        model_info: Dictionary with model metadata
        scaling_factor: Fixed multiplier to scale values (default: 100.0)
    """
    # Configure aesthetics for publication quality
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,  # Smaller for layer numbers
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    # Get model info for layer counts
    if model_info and "num_layers" in model_info:
        num_encoder_layers = model_info["num_layers"]["encoder"]
        num_decoder_layers = model_info["num_layers"]["decoder"]
    else:
        # Default for whisper-small
        num_encoder_layers = 12
        num_decoder_layers = 12

    # Extract and organize layer-specific data
    clean_layer_importance = clean_results["layer_importance"]
    other_layer_importance = other_results["layer_importance"]

    # Create dictionaries to store importance by layer number
    clean_encoder_by_layer = {i: [] for i in range(num_encoder_layers)}
    clean_decoder_by_layer = {i: [] for i in range(num_decoder_layers)}
    other_encoder_by_layer = {i: [] for i in range(num_encoder_layers)}
    other_decoder_by_layer = {i: [] for i in range(num_decoder_layers)}

    # Populate dictionaries with importance values
    for key, data in clean_layer_importance.items():
        layer_num = data["layer_num"]
        component = data["component"]
        if 0 <= layer_num < num_encoder_layers and component == "encoder":
            clean_encoder_by_layer[layer_num].append(data["avg_importance"])
        elif 0 <= layer_num < num_decoder_layers and component == "decoder":
            clean_decoder_by_layer[layer_num].append(data["avg_importance"])

    for key, data in other_layer_importance.items():
        layer_num = data["layer_num"]
        component = data["component"]
        if 0 <= layer_num < num_encoder_layers and component == "encoder":
            other_encoder_by_layer[layer_num].append(data["avg_importance"])
        elif 0 <= layer_num < num_decoder_layers and component == "decoder":
            other_decoder_by_layer[layer_num].append(data["avg_importance"])

    # Calculate average importance for each layer
    clean_encoder_means = [
        np.mean(values) if values else 0 for layer, values in sorted(clean_encoder_by_layer.items())
    ]
    clean_decoder_means = [
        np.mean(values) if values else 0 for layer, values in sorted(clean_decoder_by_layer.items())
    ]
    other_encoder_means = [
        np.mean(values) if values else 0 for layer, values in sorted(other_encoder_by_layer.items())
    ]
    other_decoder_means = [
        np.mean(values) if values else 0 for layer, values in sorted(other_decoder_by_layer.items())
    ]

    # Calculate SEMs
    clean_encoder_sems = [
        np.std(values) / np.sqrt(len(values)) if values else 0
        for layer, values in sorted(clean_encoder_by_layer.items())
    ]
    clean_decoder_sems = [
        np.std(values) / np.sqrt(len(values)) if values else 0
        for layer, values in sorted(clean_decoder_by_layer.items())
    ]
    other_encoder_sems = [
        np.std(values) / np.sqrt(len(values)) if values else 0
        for layer, values in sorted(other_encoder_by_layer.items())
    ]
    other_decoder_sems = [
        np.std(values) / np.sqrt(len(values)) if values else 0
        for layer, values in sorted(other_decoder_by_layer.items())
    ]

    # Apply fixed scaling factor to values
    clean_encoder_scaled = [x * scaling_factor for x in clean_encoder_means]
    clean_decoder_scaled = [x * scaling_factor for x in clean_decoder_means]
    other_encoder_scaled = [x * scaling_factor for x in other_encoder_means]
    other_decoder_scaled = [x * scaling_factor for x in other_decoder_means]

    # Also scale SEMs
    clean_encoder_scaled_sems = [x * scaling_factor for x in clean_encoder_sems]
    clean_decoder_scaled_sems = [x * scaling_factor for x in clean_decoder_sems]
    other_encoder_scaled_sems = [x * scaling_factor for x in other_encoder_sems]
    other_decoder_scaled_sems = [x * scaling_factor for x in other_decoder_sems]

    # Create layer indices for the x-axis
    x_indices = np.arange(max(num_encoder_layers, num_decoder_layers))
    width = 0.2  # Width of the bars

    # Create subplots for encoder and decoder
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Encoder plot
    clean_bars1 = ax1.bar(
        x_indices - width / 2,
        clean_encoder_scaled,
        width,
        label="test.clean",
        yerr=clean_encoder_scaled_sems,
        capsize=3,
        color="#1f77b4",
    )
    other_bars1 = ax1.bar(
        x_indices + width / 2,
        other_encoder_scaled,
        width,
        label="test.other",
        yerr=other_encoder_scaled_sems,
        capsize=3,
        color="#ff7f0e",
    )

    ax1.set_ylabel("Normalized Hessian Importance", fontsize=12)  # Changed from Fisher to Hessian
    ax1.set_title("Encoder Layer Sensitivity (Hessian Method)", fontsize=14)  # Updated title
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Add value labels to encoder bars (only for visible values)
    for i, bar in enumerate(clean_bars1):
        height = bar.get_height()
        if (
            height > max(clean_encoder_scaled) * 0.1
        ):  # Only add labels to bars with significant height
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 * max(clean_encoder_scaled),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for i, bar in enumerate(other_bars1):
        height = bar.get_height()
        if (
            height > max(other_encoder_scaled) * 0.1
        ):  # Only add labels to bars with significant height
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 * max(other_encoder_scaled),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Decoder plot
    clean_bars2 = ax2.bar(
        x_indices - width / 2,
        clean_decoder_scaled,
        width,
        label="test.clean",
        yerr=clean_decoder_scaled_sems,
        capsize=3,
        color="#1f77b4",
    )
    other_bars2 = ax2.bar(
        x_indices + width / 2,
        other_decoder_scaled,
        width,
        label="test.other",
        yerr=other_decoder_scaled_sems,
        capsize=3,
        color="#ff7f0e",
    )

    ax2.set_ylabel("Normalized Hessian Importance", fontsize=12)  # Changed from Fisher to Hessian
    ax2.set_title("Decoder Layer Sensitivity (Hessian Method)", fontsize=14)  # Updated title
    ax2.set_xlabel("Layer Number", fontsize=12)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{i+1}" for i in x_indices])  # 1-indexed for readability
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Add value labels to decoder bars (only for visible values)
    for i, bar in enumerate(clean_bars2):
        height = bar.get_height()
        if (
            height > max(clean_decoder_scaled) * 0.1
        ):  # Only add labels to bars with significant height
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 * max(clean_decoder_scaled),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for i, bar in enumerate(other_bars2):
        height = bar.get_height()
        if (
            height > max(other_decoder_scaled) * 0.1
        ):  # Only add labels to bars with significant height
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 * max(other_decoder_scaled),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "layer_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Create raw (unnormalized) layer charts for reference
    fig_raw, (ax1_raw, ax2_raw) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Encoder raw plot
    ax1_raw.bar(
        x_indices - width / 2,
        clean_encoder_means,
        width,
        label="test.clean",
        yerr=clean_encoder_sems,
        capsize=3,
        color="#1f77b4",
    )
    ax1_raw.bar(
        x_indices + width / 2,
        other_encoder_means,
        width,
        label="test.other",
        yerr=other_encoder_sems,
        capsize=3,
        color="#ff7f0e",
    )

    ax1_raw.set_ylabel("Raw Hessian Importance", fontsize=12)  # Changed from Fisher to Hessian
    ax1_raw.set_title("Encoder Layer Sensitivity (Raw Values)", fontsize=14)
    ax1_raw.set_ylim(bottom=0)
    ax1_raw.legend(loc="upper right")
    ax1_raw.grid(axis="y", linestyle="--", alpha=0.3)

    # Decoder raw plot
    ax2_raw.bar(
        x_indices - width / 2,
        clean_decoder_means,
        width,
        label="test.clean",
        yerr=clean_decoder_sems,
        capsize=3,
        color="#1f77b4",
    )
    ax2_raw.bar(
        x_indices + width / 2,
        other_decoder_means,
        width,
        label="test.other",
        yerr=other_decoder_sems,
        capsize=3,
        color="#ff7f0e",
    )

    ax2_raw.set_ylabel("Raw Hessian Importance", fontsize=12)  # Changed from Fisher to Hessian
    ax2_raw.set_title("Decoder Layer Sensitivity (Raw Values)", fontsize=14)
    ax2_raw.set_xlabel("Layer Number", fontsize=12)
    ax2_raw.set_xticks(x_indices)
    ax2_raw.set_xticklabels([f"{i+1}" for i in x_indices])  # 1-indexed for readability
    ax2_raw.set_ylim(bottom=0)
    ax2_raw.legend(loc="upper right")
    ax2_raw.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_sensitivity_raw.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_raw)

    print(
        f"Created layer-level sensitivity charts with Hessian method: {os.path.join(output_dir, 'layer_sensitivity.png')}"
    )
    print(
        f"Also saved raw (unnormalized) version: {os.path.join(output_dir, 'layer_sensitivity_raw.png')}"
    )


def categorize_parameter(name: str) -> Dict[str, Union[str, int]]:
    """
    Categorize parameter by location and type in Whisper architecture.

    Args:
        name: Parameter name from model.named_parameters()

    Returns:
        Dictionary with component, layer_type, layer_num, and attention_type
    """
    # Default categorization
    category = {
        "component": "other",
        "layer_type": "other",
        "layer_num": -1,
        "attention_type": "none",
    }

    # Determine component (encoder vs decoder)
    if "encoder" in name:
        category["component"] = "encoder"
    elif "decoder" in name:
        category["component"] = "decoder"

    # Extract layer number if available
    layer_match = re.search(r"layers\.(\d+)\.", name)
    if layer_match:
        category["layer_num"] = int(layer_match.group(1))

    # Determine layer type
    if "self_attn" in name:
        category["layer_type"] = "self_attention"
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            category["attention_type"] = "qkv_projection"
        elif "out_proj" in name:
            category["attention_type"] = "output_projection"
    elif "encoder_attn" in name:
        category["layer_type"] = "cross_attention"
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            category["attention_type"] = "qkv_projection"
        elif "out_proj" in name:
            category["attention_type"] = "output_projection"
    elif "fc1" in name or "fc2" in name:
        category["layer_type"] = "feed_forward"
    elif "embed" in name:
        category["layer_type"] = "embedding"
    elif "proj" in name and "layer_norm" not in name:
        category["layer_type"] = "projection"
    elif "layer_norm" in name:
        category["layer_type"] = "layer_norm"

    return category


def compute_fisher_sensitivity(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    samples: List[Dict],
    device: torch.device,
    model_name: str,
    model_info: Dict,
    num_batches: int = 3,
) -> Dict:
    """
    Compute parameter sensitivity using Fisher Information Matrix diagonal.
    This provides a second-order sensitivity measure that approximates the Hessian diagonal
    and works for models where not all operations support second derivatives.

    Args:
        model: The Whisper model
        processor: The Whisper processor
        samples: Processed dataset samples
        device: Device to run calculations on
        model_name: Name of the model
        model_info: Dictionary with model metadata
        num_batches: Number of batches to process

    Returns:
        Dictionary containing parameter and layer importance
    """
    # Initialize sensitivity metrics
    sensitivity_results = {}

    # Use CPU for most reliable results (especially for Fisher computations)
    model = model.to(torch.device("cpu"))

    # Initialize parameter tracking
    for name, param in model.named_parameters():
        if param.requires_grad:  # Process all trainable parameters
            category = categorize_parameter(name)

            sensitivity_results[name] = {
                "shape": list(param.shape),
                "size": param.numel(),
                "category": category,
                "fisher_samples": [],  # Store Fisher diagonal estimates
                "param_values": param.detach().cpu().numpy().flatten().tolist()[:5]
                if param.dim() > 0
                else [param.item()],
                "batch_count": 0,
            }

    # Set model to evaluation mode but enable gradients
    model.eval()

    # Process batches
    batch_count = 0
    sample_idx = 0
    pbar = tqdm(total=num_batches, desc="Computing Fisher Information-based sensitivity")

    # Use different samples to get better sensitivity estimates
    while batch_count < num_batches and sample_idx < len(samples):
        # Process a single batch
        try:
            # Get a single sample
            batch_samples = samples[sample_idx : sample_idx + 1]
            sample_idx += 1

            # Process each sample in the batch
            for sample in batch_samples:
                # Get features and move to device
                features = sample["input_features"].to(torch.device("cpu"))
                target_ids = sample["text_ids"].to(torch.device("cpu"))

                # Forward and backward pass to compute Fisher
                try:
                    # Zero gradients
                    model.zero_grad()

                    # Forward pass
                    outputs = model(
                        input_features=features,
                        decoder_input_ids=target_ids[:, :-1] if target_ids.size(1) > 1 else None,
                        labels=target_ids if target_ids.size(1) <= 1 else target_ids[:, 1:],
                    )

                    loss = outputs.loss

                    # Backward pass (compute gradients)
                    loss.backward()
                except Exception as e:
                    print(f"Warning: Error in forward/backward pass: {e}")
                    # Skip this sample if there's an error
                    continue

                # Compute Fisher diagonal (squared gradient magnitude)
                for name, param in model.named_parameters():
                    if name in sensitivity_results and param.grad is not None:
                        # Fisher Information Matrix diagonal is approximated as the squared gradient magnitude
                        # This is a positive semi-definite approximation of the Hessian
                        fisher_diag = (param.grad**2).mean().item()

                        if np.isfinite(fisher_diag) and fisher_diag > 0:
                            sensitivity_results[name]["fisher_samples"].append(fisher_diag)
                            sensitivity_results[name]["batch_count"] += 1

                # Clear memory
                clear_memory()

            # Update batch counter and progress bar
            batch_count += 1
            pbar.update(1)

            # Clear memory between batches
            clear_memory()

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Skip to next sample
            sample_idx += 1
            clear_memory()

    pbar.close()

    # Calculate importance scores from collected Fisher metrics
    print("\nCalculating importance scores...")

    # Parameter importance
    parameter_importance = {}

    success_count = 0
    for name, data in tqdm(sensitivity_results.items(), desc="Processing parameters"):
        if data["batch_count"] > 0 and data["fisher_samples"]:
            success_count += 1
            # Get component information
            category = data["category"]

            # Calculate mean Fisher diagonal value across samples
            fisher_mean = np.mean(data["fisher_samples"])

            # Store importance statistics
            parameter_importance[name] = {
                "category": data["category"],
                "shape": data["shape"],
                "size": data["size"],
                "param_sample": data["param_values"],
                "importance": fisher_mean,
                "importance_raw": fisher_mean,
                "sensitivity": fisher_mean,
                "sample_count": len(data["fisher_samples"]),
            }

    print(
        f"Successfully computed Fisher sensitivity for {success_count}/{len(sensitivity_results)} parameters"
    )

    # Calculate global statistics for normalization
    all_importances = [data["importance"] for name, data in parameter_importance.items()]
    if all_importances:
        max_importance = max(all_importances)
        min_importance = min(all_importances)

        # Normalize importance scores
        if max_importance > min_importance:
            for name in parameter_importance:
                raw_importance = parameter_importance[name]["importance"]
                normalized_importance = (raw_importance - min_importance) / (
                    max_importance - min_importance
                )
                parameter_importance[name]["importance"] = normalized_importance

    # Layer importance (aggregate by layer)
    layer_importance = {}

    for name, data in parameter_importance.items():
        category = data["category"]
        component = category["component"]
        layer_type = category["layer_type"]
        layer_num = category["layer_num"]

        layer_key = f"{component}_{layer_num}_{layer_type}"

        if layer_key not in layer_importance:
            layer_importance[layer_key] = {
                "component": component,
                "layer_type": layer_type,
                "layer_num": layer_num,
                "total_importance": 0.0,
                "total_params": 0,
                "avg_importance": 0.0,
            }

        # Accumulate importance weighted by parameter count
        layer_importance[layer_key]["total_importance"] += data["importance"] * data["size"]
        layer_importance[layer_key]["total_params"] += data["size"]

    # Calculate average importance
    for key in layer_importance:
        if layer_importance[key]["total_params"] > 0:
            layer_importance[key]["avg_importance"] = (
                layer_importance[key]["total_importance"] / layer_importance[key]["total_params"]
            )

    return {"parameter_importance": parameter_importance, "layer_importance": layer_importance}


def compute_fisher_sensitivity(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    samples: List[Dict],
    device: torch.device,
    model_name: str,
    model_info: Dict,
    num_batches: int = 3,
) -> Dict:
    """
    Compute parameter sensitivity using Fisher Information Matrix diagonal.
    This provides a second-order sensitivity measure that approximates the Hessian diagonal
    and works for models where not all operations support second derivatives.

    Args:
        model: The Whisper model
        processor: The Whisper processor
        samples: Processed dataset samples
        device: Device to run calculations on
        model_name: Name of the model
        model_info: Dictionary with model metadata
        num_batches: Number of batches to process

    Returns:
        Dictionary containing parameter and layer importance
    """
    # Initialize sensitivity metrics
    sensitivity_results = {}

    # Use CPU for most reliable results (especially for Fisher computations)
    model = model.to(torch.device("cpu"))

    # Initialize parameter tracking
    for name, param in model.named_parameters():
        if param.requires_grad:  # Process all trainable parameters
            category = categorize_parameter(name)

            sensitivity_results[name] = {
                "shape": list(param.shape),
                "size": param.numel(),
                "category": category,
                "fisher_samples": [],  # Store Fisher diagonal estimates
                "param_values": param.detach().cpu().numpy().flatten().tolist()[:5]
                if param.dim() > 0
                else [param.item()],
                "batch_count": 0,
            }

    # Set model to evaluation mode but enable gradients
    model.eval()

    # Process batches
    batch_count = 0
    sample_idx = 0
    pbar = tqdm(total=num_batches, desc="Computing Fisher Information-based sensitivity")

    # Use different samples to get better sensitivity estimates
    while batch_count < num_batches and sample_idx < len(samples):
        # Process a single batch
        try:
            # Get a single sample
            batch_samples = samples[sample_idx : sample_idx + 1]
            sample_idx += 1

            # Process each sample in the batch
            for sample in batch_samples:
                # Get features and move to device
                features = sample["input_features"].to(torch.device("cpu"))
                target_ids = sample["text_ids"].to(torch.device("cpu"))

                # Forward and backward pass to compute Fisher
                try:
                    # Zero gradients
                    model.zero_grad()

                    # Forward pass
                    outputs = model(
                        input_features=features,
                        decoder_input_ids=target_ids[:, :-1] if target_ids.size(1) > 1 else None,
                        labels=target_ids if target_ids.size(1) <= 1 else target_ids[:, 1:],
                    )

                    loss = outputs.loss

                    # Backward pass (compute gradients)
                    loss.backward()
                except Exception as e:
                    print(f"Warning: Error in forward/backward pass: {e}")
                    # Skip this sample if there's an error
                    continue

                # Compute Fisher diagonal (squared gradient magnitude)
                for name, param in model.named_parameters():
                    if name in sensitivity_results and param.grad is not None:
                        # Fisher Information Matrix diagonal is approximated as the squared gradient magnitude
                        # This is a positive semi-definite approximation of the Hessian
                        fisher_diag = (param.grad**2).mean().item()

                        if np.isfinite(fisher_diag) and fisher_diag > 0:
                            sensitivity_results[name]["fisher_samples"].append(fisher_diag)
                            sensitivity_results[name]["batch_count"] += 1

                # Clear memory
                clear_memory()

            # Update batch counter and progress bar
            batch_count += 1
            pbar.update(1)

            # Clear memory between batches
            clear_memory()

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Skip to next sample
            sample_idx += 1
            clear_memory()

    pbar.close()

    # Calculate importance scores from collected Fisher metrics
    print("\nCalculating importance scores...")

    # Parameter importance
    parameter_importance = {}

    success_count = 0
    for name, data in tqdm(sensitivity_results.items(), desc="Processing parameters"):
        if data["batch_count"] > 0 and data["fisher_samples"]:
            success_count += 1
            # Get component information
            category = data["category"]

            # Calculate mean Fisher diagonal value across samples
            fisher_mean = np.mean(data["fisher_samples"])

            # Store importance statistics
            parameter_importance[name] = {
                "category": data["category"],
                "shape": data["shape"],
                "size": data["size"],
                "param_sample": data["param_values"],
                "importance": fisher_mean,
                "importance_raw": fisher_mean,
                "sensitivity": fisher_mean,
                "sample_count": len(data["fisher_samples"]),
            }

    print(
        f"Successfully computed Fisher sensitivity for {success_count}/{len(sensitivity_results)} parameters"
    )

    # Calculate global statistics for normalization
    all_importances = [data["importance"] for name, data in parameter_importance.items()]
    if all_importances:
        max_importance = max(all_importances)
        min_importance = min(all_importances)

        # Normalize importance scores
        if max_importance > min_importance:
            for name in parameter_importance:
                raw_importance = parameter_importance[name]["importance"]
                normalized_importance = (raw_importance - min_importance) / (
                    max_importance - min_importance
                )
                parameter_importance[name]["importance"] = normalized_importance

    # Layer importance (aggregate by layer)
    layer_importance = {}

    for name, data in parameter_importance.items():
        category = data["category"]
        component = category["component"]
        layer_type = category["layer_type"]
        layer_num = category["layer_num"]

        layer_key = f"{component}_{layer_num}_{layer_type}"

        if layer_key not in layer_importance:
            layer_importance[layer_key] = {
                "component": component,
                "layer_type": layer_type,
                "layer_num": layer_num,
                "total_importance": 0.0,
                "total_params": 0,
                "avg_importance": 0.0,
            }

        # Accumulate importance weighted by parameter count
        layer_importance[layer_key]["total_importance"] += data["importance"] * data["size"]
        layer_importance[layer_key]["total_params"] += data["size"]

    # Calculate average importance
    for key in layer_importance:
        if layer_importance[key]["total_params"] > 0:
            layer_importance[key]["avg_importance"] = (
                layer_importance[key]["total_importance"] / layer_importance[key]["total_params"]
            )

    return {"parameter_importance": parameter_importance, "layer_importance": layer_importance}


def create_combined_sensitivity_chart(
    clean_results: Dict,
    other_results: Dict,
    output_dir: str,
    model_info: Optional[Dict] = None,
    scaling_factor: float = 100.0,  # Fixed scaling factor instead of normalization
) -> None:
    """
    Create a component sensitivity chart comparing encoder vs decoder importance
    on both test.clean and test.other datasets.
    Uses a fixed scaling factor to make values more readable.

    Args:
        clean_results: Results from compute_fisher_sensitivity for test.clean
        other_results: Results from compute_fisher_sensitivity for test.other
        output_dir: Directory to save the plot
        model_info: Dictionary with model metadata
        scaling_factor: Fixed multiplier to scale values (default: 100.0)
    """
    # Configure aesthetics for publication quality
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get model name for potential use
    model_name = (
        model_info["name"].split("/")[-1] if model_info and "name" in model_info else "whisper"
    )

    # Extract and process data
    clean_layer_importance = clean_results["layer_importance"]
    other_layer_importance = other_results["layer_importance"]

    # Organize data by component
    clean_encoder_imp = []
    clean_decoder_imp = []
    other_encoder_imp = []
    other_decoder_imp = []

    for key, data in clean_layer_importance.items():
        if data["layer_num"] >= 0:  # Only include layers with valid position
            if data["component"] == "encoder":
                clean_encoder_imp.append(data["avg_importance"])
            elif data["component"] == "decoder":
                clean_decoder_imp.append(data["avg_importance"])

    for key, data in other_layer_importance.items():
        if data["layer_num"] >= 0:  # Only include layers with valid position
            if data["component"] == "encoder":
                other_encoder_imp.append(data["avg_importance"])
            elif data["component"] == "decoder":
                other_decoder_imp.append(data["avg_importance"])

    # Calculate raw statistics first
    clean_enc_raw_mean = np.mean(clean_encoder_imp) if clean_encoder_imp else 0
    clean_dec_raw_mean = np.mean(clean_decoder_imp) if clean_decoder_imp else 0
    clean_enc_sem = (
        np.std(clean_encoder_imp) / np.sqrt(len(clean_encoder_imp)) if clean_encoder_imp else 0
    )
    clean_dec_sem = (
        np.std(clean_decoder_imp) / np.sqrt(len(clean_decoder_imp)) if clean_decoder_imp else 0
    )

    other_enc_raw_mean = np.mean(other_encoder_imp) if other_encoder_imp else 0
    other_dec_raw_mean = np.mean(other_decoder_imp) if other_decoder_imp else 0
    other_enc_sem = (
        np.std(other_encoder_imp) / np.sqrt(len(other_encoder_imp)) if other_encoder_imp else 0
    )
    other_dec_sem = (
        np.std(other_decoder_imp) / np.sqrt(len(other_decoder_imp)) if other_decoder_imp else 0
    )

    # Apply fixed scaling factor instead of normalization
    clean_enc_mean = clean_enc_raw_mean * scaling_factor
    clean_dec_mean = clean_dec_raw_mean * scaling_factor
    other_enc_mean = other_enc_raw_mean * scaling_factor
    other_dec_mean = other_dec_raw_mean * scaling_factor

    # Also scale the SEMs
    clean_enc_sem *= scaling_factor
    clean_dec_sem *= scaling_factor
    other_enc_sem *= scaling_factor
    other_dec_sem *= scaling_factor

    # Define width and positions for grouped bars
    width = 0.35
    encoder_pos = 0
    decoder_pos = 1

    # Create bars grouped by component (as shown in the image)
    clean_bars = ax.bar(
        [encoder_pos - width / 2, decoder_pos - width / 2],
        [clean_enc_mean, clean_dec_mean],
        width,
        label="test.clean",
        yerr=[clean_enc_sem, clean_dec_sem],
        capsize=5,
        color="#1f77b4",  # Blue color
    )

    other_bars = ax.bar(
        [encoder_pos + width / 2, decoder_pos + width / 2],
        [other_enc_mean, other_dec_mean],
        width,
        label="test.other",
        yerr=[other_enc_sem, other_dec_sem],
        capsize=5,
        color="#ff7f0e",  # Orange color
    )

    # Add value labels higher above the bars (rounded to 2 decimal places)
    # Increased vertical offset to move labels lower from the border
    label_offset = -0.02  # Negative value to move labels down slightly inside the bars
    for i, bar in enumerate(clean_bars):
        height = bar.get_height()
        error = [clean_enc_sem, clean_dec_sem][i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + error - label_offset,  # Position labels slightly lower
            f"{height:.2f}",
            ha="center",
            va="top",  # Changed to 'top' to position from the top down
            fontsize=10,
        )

    for i, bar in enumerate(other_bars):
        height = bar.get_height()
        error = [other_enc_sem, other_dec_sem][i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + error - label_offset,  # Position labels slightly lower
            f"{height:.2f}",
            ha="center",
            va="top",  # Changed to 'top' to position from the top down
            fontsize=10,
        )

    # Set labels and title
    ax.set_ylabel("Normalized Hessian Importance", fontsize=12)  # Changed from Fisher to Hessian
    ax.set_title(
        "Encoder vs Decoder Hessian based Sensitivity Comparison", fontsize=14
    )  # Updated title
    ax.set_xticks([encoder_pos, decoder_pos])
    ax.set_xticklabels(["Encoder", "Decoder"])

    # Set y-axis to start at 0 and add some extra space at the top for labels
    max_height = max(clean_dec_mean + clean_dec_sem, other_dec_mean + other_dec_sem)
    ax.set_ylim(bottom=0, top=max_height * 1.15)  # Add 15% extra space at the top

    # Add legend matching the image (top right)
    ax.legend(loc="upper right")

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Tight layout and save
    plt.tight_layout()

    # Save as high-resolution image
    plt.savefig(os.path.join(output_dir, "combined_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "combined_sensitivity.pdf"), bbox_inches="tight")

    # Also save a version with raw (unnormalized) values for reference
    fig_raw, ax_raw = plt.subplots(figsize=(10, 6))

    # Create bars with raw values
    clean_bars_raw = ax_raw.bar(
        [encoder_pos - width / 2, decoder_pos - width / 2],
        [clean_enc_raw_mean, clean_dec_raw_mean],
        width,
        label="test.clean",
        yerr=[clean_enc_sem / scaling_factor, clean_dec_sem / scaling_factor],
        capsize=5,
        color="#1f77b4",
    )

    other_bars_raw = ax_raw.bar(
        [encoder_pos + width / 2, decoder_pos + width / 2],
        [other_enc_raw_mean, other_dec_raw_mean],
        width,
        label="test.other",
        yerr=[other_enc_sem / scaling_factor, other_dec_sem / scaling_factor],
        capsize=5,
        color="#ff7f0e",
    )

    # Add value labels to raw chart
    for i, bar in enumerate(clean_bars_raw):
        height = bar.get_height()
        error = [clean_enc_sem / scaling_factor, clean_dec_sem / scaling_factor][i]
        ax_raw.text(
            bar.get_x() + bar.get_width() / 2,
            height + error + height * 0.05,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for i, bar in enumerate(other_bars_raw):
        height = bar.get_height()
        error = [other_enc_sem / scaling_factor, other_dec_sem / scaling_factor][i]
        ax_raw.text(
            bar.get_x() + bar.get_width() / 2,
            height + error + height * 0.05,
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Set labels for raw chart
    ax_raw.set_ylabel(
        "Raw Hessian Importance (Unnormalized)", fontsize=12
    )  # Changed from Fisher to Hessian
    ax_raw.set_title("Encoder vs Decoder Raw Hessian Sensitivity", fontsize=14)  # Updated title
    ax_raw.set_xticks([encoder_pos, decoder_pos])
    ax_raw.set_xticklabels(["Encoder", "Decoder"])
    ax_raw.set_ylim(bottom=0)
    ax_raw.legend(loc="upper right")
    ax_raw.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "combined_sensitivity_raw.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig_raw)

    # Close the figure
    plt.close(fig)

    print(
        f"Created scaled sensitivity chart: {os.path.join(output_dir, 'combined_sensitivity.png')}"
    )
    print(
        f"Also saved raw (unnormalized) version: {os.path.join(output_dir, 'combined_sensitivity_raw.png')}"
    )


def run_combined_sensitivity_analysis(
    model_name: str,
    output_dir: Optional[str] = None,
    num_samples: int = 500,
    num_batches: int = 3,
    scaling_factor: float = 100.0,  # Fixed scaling factor for visualization
) -> str:
    """
    Run a sensitivity analysis for Whisper model on both test.clean and test.other datasets.

    Args:
        model_name: The Whisper model name ("openai/whisper-small")
        output_dir: Output directory (defaults to timestamped directory)
        num_samples: Number of samples to use from each dataset
        num_batches: Number of batches to use for sensitivity computation
        scaling_factor: Fixed multiplier for scaling sensitivity values in visualizations

    Returns:
        Output directory path
    """
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"whisper_fisher_sensitivity_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = get_device()

    # For optimal Fisher computation, we'll use CPU
    computation_device = torch.device("cpu")

    print("\n=== Whisper Combined Component Sensitivity Analysis (Fisher Information) ===")
    print(f"Model: {model_name}")
    print(f"Display device: {device}, Computation device: {computation_device}")
    print(f"Samples per dataset: {num_samples}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Using scaling factor ×{scaling_factor} for visualizations")

    # 1. Load model and processor (on CPU for stability)
    print("\nLoading model and processor...")

    # Try to create config that disables flash attention for more accurate sensitivity
    try:
        config = WhisperConfig.from_pretrained(model_name)
        config.use_flash_attention = False
        config.use_sdpa = False  # Disable scaled dot product attention if present
        model = WhisperForConditionalGeneration.from_pretrained(model_name, config=config)
    except:
        # Fall back to default loading if config modification fails
        print("Note: Could not disable flash attention via config. Using default model loading.")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    processor = WhisperProcessor.from_pretrained(model_name)

    # Save model architecture details
    config = model.config
    num_encoder_layers = getattr(config, "encoder_layers", 0)
    if num_encoder_layers == 0:
        # Try to find encoder through model directly
        try:
            num_encoder_layers = len(model.model.encoder.layers)
        except (AttributeError, TypeError):
            # Fallback to default values for Whisper models
            whisper_sizes = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}
            # Extract size from model name
            for size in whisper_sizes:
                if size in model_name:
                    num_encoder_layers = whisper_sizes[size]
                    break
            else:
                # Default to small if no match
                num_encoder_layers = 12

    # Same for decoder layers
    num_decoder_layers = getattr(config, "decoder_layers", 0)
    if num_decoder_layers == 0:
        try:
            num_decoder_layers = len(model.model.decoder.layers)
        except (AttributeError, TypeError):
            # Fallback for Whisper
            if "tiny" in model_name or "base" in model_name:
                num_decoder_layers = 4
            elif "small" in model_name:
                num_decoder_layers = 12
            elif "medium" in model_name:
                num_decoder_layers = 24
            elif "large" in model_name:
                num_decoder_layers = 32
            else:
                # Default to small
                num_decoder_layers = 12

    model_info = {
        "name": model_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": {
            "encoder": num_encoder_layers,
            "decoder": num_decoder_layers,
        },
        "hidden_size": config.d_model,
        "attention_heads": {
            "encoder": getattr(config, "encoder_attention_heads", 8),
            "decoder": getattr(config, "decoder_attention_heads", 8),
        },
    }

    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    # 2. Run analysis on test.clean dataset
    print("\n--- Running analysis on test.clean dataset ---")

    # Load test.clean dataset
    print(f"\nLoading test.clean dataset with {num_samples} samples...")
    dataset_clean = load_librispeech(num_samples=num_samples, split="test.clean")

    # Process test.clean samples
    print("\nProcessing test.clean dataset...")
    processed_samples_clean = []
    for i, sample in enumerate(tqdm(dataset_clean, desc="Processing clean samples")):
        try:
            # Process on CPU for maximum compatibility
            audio = sample["audio"]
            input_features = processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ).input_features

            text = sample["text"]
            text_encoding = processor(text=text, return_tensors="pt")

            processed_samples_clean.append(
                {
                    "input_features": input_features,
                    "text_ids": text_encoding.input_ids,
                    "text": text,
                }
            )

            # Clear memory periodically
            if i % 50 == 0:
                clear_memory()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Compute sensitivity for test.clean
    print("\nComputing Fisher-based parameter sensitivity for test.clean...")
    clean_results = compute_fisher_sensitivity(
        model=model,
        processor=processor,
        samples=processed_samples_clean,
        device=computation_device,
        model_name=model_name,
        model_info=model_info,
        num_batches=num_batches,
    )

    # Save clean results
    with open(os.path.join(output_dir, "clean_layer_importance.json"), "w") as f:
        json.dump(clean_results["layer_importance"], f, indent=2)

    # Clear memory before test.other
    clear_memory()

    # 3. Run analysis on test.other dataset
    print("\n--- Running analysis on test.other dataset ---")

    # Load test.other dataset
    print(f"\nLoading test.other dataset with {num_samples} samples...")
    dataset_other = load_librispeech(num_samples=num_samples, split="test.other")

    # Process test.other samples
    print("\nProcessing test.other dataset...")
    processed_samples_other = []
    for i, sample in enumerate(tqdm(dataset_other, desc="Processing other samples")):
        try:
            # Process on CPU for maximum compatibility
            audio = sample["audio"]
            input_features = processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ).input_features

            text = sample["text"]
            text_encoding = processor(text=text, return_tensors="pt")

            processed_samples_other.append(
                {
                    "input_features": input_features,
                    "text_ids": text_encoding.input_ids,
                    "text": text,
                }
            )

            # Clear memory periodically
            if i % 50 == 0:
                clear_memory()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Compute sensitivity for test.other
    print("\nComputing Fisher-based parameter sensitivity for test.other...")
    other_results = compute_fisher_sensitivity(
        model=model,
        processor=processor,
        samples=processed_samples_other,
        device=computation_device,
        model_name=model_name,
        model_info=model_info,
        num_batches=num_batches,
    )

    # Save other results
    with open(os.path.join(output_dir, "other_layer_importance.json"), "w") as f:
        json.dump(other_results["layer_importance"], f, indent=2)

    # 4. Create visualizations
    print("\nCreating sensitivity visualizations...")

    # Component-level chart with fixed scaling factor
    create_combined_sensitivity_chart(
        clean_results=clean_results,
        other_results=other_results,
        output_dir=output_dir,
        model_info=model_info,
        scaling_factor=scaling_factor,  # Use the specified scaling factor
    )

    # Layer-level chart with fixed scaling factor
    create_layer_sensitivity_chart(
        clean_results=clean_results,
        other_results=other_results,
        output_dir=output_dir,
        model_info=model_info,
        scaling_factor=scaling_factor,  # Use the specified scaling factor
    )

    return output_dir


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Whisper Component Sensitivity Analysis with Fisher Information Matrix (clean vs other comparison)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Model name (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--samples", type=int, default=500, help="Number of samples per dataset (default: 500)"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of batches for sensitivity computation (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped directory)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if MPS/CUDA is available",
    )
    return parser.parse_args()


def main():
    """Main function to run combined component sensitivity analysis with Fisher Information Matrix"""
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Force CPU if requested (computations will use CPU anyway for Fisher approximation)
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallbacks

    # Print information about Fisher-based approach
    print("\n===== Whisper Component Sensitivity Analysis with Fisher Information Matrix =====")
    print("This analysis uses the diagonal of the Fisher Information Matrix, which is a")
    print("positive semi-definite approximation of the Hessian. The Fisher diagonal measures")
    print("the sensitivity of model parameters using squared gradient magnitudes, effectively")
    print(
        "capturing second-order curvature information without requiring explicit second derivatives."
    )
    print("This approach is ideal for Whisper models where some operations don't support")
    print("second derivatives in PyTorch (like flash attention).")
    print("=============================================================================\n")

    # Run combined sensitivity analysis
    results_dir = run_combined_sensitivity_analysis(
        model_name=args.model,
        num_samples=args.samples,
        num_batches=args.batches,
        output_dir=args.output_dir,
    )

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"\nCombined component Fisher-based sensitivity analysis complete. Results saved to: {results_dir}"
    )
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    print("\nOutput files:")
    print(
        f"- {os.path.join(results_dir, 'combined_sensitivity.png/pdf')}: Combined sensitivity chart"
    )
    print(f"- {os.path.join(results_dir, 'model_info.json')}: Model architecture information")
    print(f"- {os.path.join(results_dir, 'clean_layer_importance.json')}: Test.clean layer data")
    print(f"- {os.path.join(results_dir, 'other_layer_importance.json')}: Test.other layer data")


if __name__ == "__main__":
    main()
