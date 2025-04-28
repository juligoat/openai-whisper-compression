"""
Whisper Model Sensitivity Analysis for MPS (Apple Silicon)
===================================================

This module provides tools to analyze the parameter sensitivity of Whisper
speech recognition models and generate pruning configurations.

Specially optimized for MPS (Apple Silicon) devices with fallbacks and memory management.

Usage:
    python whisper_mps_analysis.py --model openai/whisper-small --samples 1000
"""

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# Configure MPS for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for unsupported ops

import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Create results directory
RESULTS_DIR = "pruning/whisper_sensitivity_results"
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
        torch.mps.empty_cache()  # This might not work in all PyTorch versions
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()


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


def compute_sensitivity_mps(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    samples: List[Dict],
    device: torch.device,
    model_name: str,
    model_info: Dict,
    num_batches: int = 3,
    chunk_size: int = 100,  # Process parameters in chunks to save memory
) -> Dict:
    """
    Compute parameter sensitivity optimized for MPS.
    Uses gradient magnitude as the primary metric with fallbacks for complex operations.

    Args:
        model: The Whisper model
        processor: The Whisper processor
        samples: Processed dataset samples
        device: Device to run calculations on
        model_name: Name of the model
        model_info: Dictionary with model metadata
        num_batches: Number of batches to process
        chunk_size: Number of parameters to process in a single chunk

    Returns:
        Dictionary containing parameter and layer importance
    """
    # Initialize sensitivity metrics
    sensitivity_results = {}

    # Initialize parameter tracking
    param_list = []
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:  # Focus on weight matrices
            category = categorize_parameter(name)

            sensitivity_results[name] = {
                "shape": list(param.shape),
                "size": param.numel(),
                "category": category,
                "sensitivity_samples": [],  # Store sensitivity estimates
                "param_values": param.detach().cpu().numpy().flatten().tolist()[:5],
                "batch_count": 0,
            }

            param_list.append((name, param))

    # Group parameters into chunks for processing to avoid memory issues
    param_chunks = [param_list[i : i + chunk_size] for i in range(0, len(param_list), chunk_size)]

    # Set model to evaluation mode but enable gradients
    model.eval()

    # Process batches
    batch_count = 0
    sample_idx = 0
    pbar = tqdm(total=num_batches, desc="Computing sensitivity metrics")

    # Use different samples to get better sensitivity estimates
    while batch_count < num_batches and sample_idx < len(samples):
        # Process a single batch
        try:
            # Get a single sample (keep batch size to 1 for stability on MPS)
            batch_samples = samples[sample_idx : sample_idx + 1]
            sample_idx += 1

            # Process each sample in the batch
            for sample in batch_samples:
                # Get features and move to device
                features = sample["input_features"].to(device)
                target_ids = sample["text_ids"].to(device)

                # Step 1: Compute first-order gradients for sensitivity
                model.zero_grad()

                # On MPS, some operations might need CPU fallback
                try:
                    # Forward pass with enabled gradients
                    outputs = model(
                        input_features=features,
                        decoder_input_ids=target_ids[:, :-1] if target_ids.size(1) > 1 else None,
                        labels=target_ids if target_ids.size(1) <= 1 else target_ids[:, 1:],
                    )

                    loss = outputs.loss
                    loss.backward()
                except Exception as e:
                    print(f"Error in forward/backward pass, trying CPU fallback: {e}")
                    # Move tensors to CPU for this sample
                    cpu_features = features.cpu()
                    cpu_target_ids = target_ids.cpu()
                    model.cpu()

                    # Forward pass on CPU
                    outputs = model(
                        input_features=cpu_features,
                        decoder_input_ids=cpu_target_ids[:, :-1]
                        if cpu_target_ids.size(1) > 1
                        else None,
                        labels=cpu_target_ids
                        if cpu_target_ids.size(1) <= 1
                        else cpu_target_ids[:, 1:],
                    )

                    loss = outputs.loss
                    loss.backward()

                    # Move model back to original device
                    model.to(device)

                # Process parameters in chunks to save memory
                for chunk in param_chunks:
                    for name, param in chunk:
                        if name in sensitivity_results and param.grad is not None:
                            # Calculate L1 norm of gradient (absolute gradient magnitude)
                            # Use this as a reliable sensitivity metric that works on MPS
                            grad_magnitude = param.grad.abs().mean().item()

                            # Store valid results
                            if np.isfinite(grad_magnitude) and grad_magnitude > 0:
                                sensitivity_results[name]["sensitivity_samples"].append(
                                    grad_magnitude
                                )
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

    # Calculate importance scores from collected sensitivity metrics
    print("\nCalculating importance scores...")

    # Parameter importance
    parameter_importance = {}

    for name, data in tqdm(sensitivity_results.items(), desc="Processing parameters"):
        if data["batch_count"] > 0 and data["sensitivity_samples"]:
            # Get component information
            category = data["category"]

            # Calculate mean sensitivity across samples
            sensitivity_mean = np.mean(data["sensitivity_samples"])

            # Store importance statistics
            parameter_importance[name] = {
                "category": data["category"],
                "shape": data["shape"],
                "size": data["size"],
                "param_sample": data["param_values"],
                "importance": sensitivity_mean,
                "importance_raw": sensitivity_mean,
                "sensitivity": sensitivity_mean,
                "sample_count": len(data["sensitivity_samples"]),
            }

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


def create_summary_plot(
    sensitivity_results: Dict, output_dir: str, model_info: Optional[Dict] = None
) -> None:
    """
    Create a concise, publication-ready summary plot for a research paper.
    This generates a single figure with multiple panels showing the key insights
    from the sensitivity analysis.

    Args:
        sensitivity_results: Results from compute_sensitivity_mps
        output_dir: Directory to save the plot
        model_info: Dictionary with model metadata
    """
    # Extract data from sensitivity results
    layer_importance = sensitivity_results["layer_importance"]

    # Configure aesthetics for publication quality
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )

    # Use a colorblind-friendly palette
    palette = sns.color_palette("colorblind")

    # Create custom colormap for heatmap - Use a diverging colormap
    prune_cmap = LinearSegmentedColormap.from_list(
        "prune_cmap", [(0, 0, 0.7), (0.95, 0.95, 1), (1, 0.8, 0.8), (0.7, 0, 0)]
    )

    # Set up figure with a 2x2 grid layout
    fig = plt.figure(figsize=(10, 8.5))

    # Title with model info
    model_name = model_info["name"] if model_info else "Whisper"
    fig.suptitle(f"Parameter Sensitivity Analysis: {model_name}", fontsize=14)

    # Create 2x2 grid for subplots
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

    # --- Panel 1: Encoder vs Decoder Component Importance ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Organize data by component and calculate statistics
    encoder_imp = []
    decoder_imp = []

    for key, data in layer_importance.items():
        if data["layer_num"] >= 0:  # Only include layers with valid position
            if data["component"] == "encoder":
                encoder_imp.append(data["avg_importance"])
            elif data["component"] == "decoder":
                decoder_imp.append(data["avg_importance"])

    # Calculate statistics
    enc_mean = np.mean(encoder_imp) if encoder_imp else 0
    dec_mean = np.mean(decoder_imp) if decoder_imp else 0
    enc_sem = np.std(encoder_imp) / np.sqrt(len(encoder_imp)) if encoder_imp else 0
    dec_sem = np.std(decoder_imp) / np.sqrt(len(decoder_imp)) if decoder_imp else 0

    # Create bar chart
    x_pos = [0, 1]
    components = ["Encoder", "Decoder"]
    means = [enc_mean, dec_mean]
    errors = [enc_sem, dec_sem]

    ax1.bar(
        x_pos,
        means,
        yerr=errors,
        align="center",
        color=[palette[0], palette[1]],
        capsize=5,
        width=0.6,
    )

    # Add value labels
    for i, v in enumerate(means):
        ax1.text(i, v + errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Add styling
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components)
    ax1.set_ylabel("Normalized Importance")
    ax1.set_title("A) Component Sensitivity", fontsize=12)

    # Calculate and show the ratio
    ratio = enc_mean / dec_mean if dec_mean > 0 else 0
    ax1.text(
        0.5,
        0.1,
        f"Enc/Dec Ratio: {ratio:.2f}",
        transform=ax1.transAxes,
        ha="center",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    # --- Panel 2: Layer Position Importance (Early, Middle, Late) ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Find max layer numbers to categorize
    max_enc_layer = max(
        [
            data["layer_num"]
            for k, data in layer_importance.items()
            if data["component"] == "encoder" and data["layer_num"] >= 0
        ],
        default=0,
    )
    max_dec_layer = max(
        [
            data["layer_num"]
            for k, data in layer_importance.items()
            if data["component"] == "decoder" and data["layer_num"] >= 0
        ],
        default=0,
    )

    # Create position categories
    position_data = {
        "encoder": {"early": [], "middle": [], "late": []},
        "decoder": {"early": [], "middle": [], "late": []},
    }

    # Categorize layers by relative position
    for key, data in layer_importance.items():
        component = data["component"]
        layer_num = data["layer_num"]

        if component in ["encoder", "decoder"] and layer_num >= 0:
            max_layer = max_enc_layer if component == "encoder" else max_dec_layer

            # Categorize by relative position
            if layer_num <= max_layer * 0.33:
                position_data[component]["early"].append(data["avg_importance"])
            elif layer_num <= max_layer * 0.67:
                position_data[component]["middle"].append(data["avg_importance"])
            else:
                position_data[component]["late"].append(data["avg_importance"])

    # Calculate statistics
    positions = ["Early", "Middle", "Late"]
    x_pos = np.arange(len(positions))
    width = 0.35

    enc_means = []
    enc_errors = []
    dec_means = []
    dec_errors = []

    for pos in ["early", "middle", "late"]:
        enc_data = position_data["encoder"][pos]
        dec_data = position_data["decoder"][pos]

        enc_means.append(np.mean(enc_data) if enc_data else 0)
        enc_errors.append(
            np.std(enc_data) / np.sqrt(len(enc_data)) if enc_data and len(enc_data) > 1 else 0
        )

        dec_means.append(np.mean(dec_data) if dec_data else 0)
        dec_errors.append(
            np.std(dec_data) / np.sqrt(len(dec_data)) if dec_data and len(dec_data) > 1 else 0
        )

    # Plot grouped bars
    ax2.bar(
        x_pos - width / 2,
        enc_means,
        width,
        yerr=enc_errors,
        color=palette[0],
        label="Encoder",
        capsize=4,
    )
    ax2.bar(
        x_pos + width / 2,
        dec_means,
        width,
        yerr=dec_errors,
        color=palette[1],
        label="Decoder",
        capsize=4,
    )

    # Add styling
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(positions)
    ax2.set_ylabel("Normalized Importance")
    ax2.set_title("B) Position-Based Sensitivity", fontsize=12)
    ax2.legend(loc="upper right", frameon=True)

    # --- Panel 3: Layer Type Importance ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Compile layer type data
    layer_types = {
        "encoder": {"self_attention": [], "feed_forward": [], "layer_norm": []},
        "decoder": {
            "self_attention": [],
            "cross_attention": [],
            "feed_forward": [],
            "layer_norm": [],
        },
    }

    # Collect data by layer type
    for key, data in layer_importance.items():
        component = data["component"]
        layer_type = data["layer_type"]

        if component in layer_types and layer_type in layer_types[component]:
            layer_types[component][layer_type].append(data["avg_importance"])

    # Prepare data for plotting
    all_types = []
    combined_means = []
    combined_errors = []
    colors = []

    # Order by component and importance
    for component, comp_data in layer_types.items():
        type_means = []
        for layer_type, values in comp_data.items():
            if values:
                mean_value = np.mean(values)
                type_means.append((layer_type, mean_value, values))

        # Sort by importance (highest first)
        type_means.sort(key=lambda x: x[1], reverse=True)

        # Add to combined list
        for layer_type, mean_value, values in type_means:
            error = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            all_types.append(f"{component.capitalize()}\n{layer_type.replace('_', ' ').title()}")
            combined_means.append(mean_value)
            combined_errors.append(error)
            colors.append(palette[0] if component == "encoder" else palette[1])

    # Plot horizontal bars
    y_pos = np.arange(len(all_types))
    ax3.barh(
        y_pos, combined_means, xerr=combined_errors, color=colors, height=0.6, capsize=4, alpha=0.8
    )

    # Add value labels
    for i, v in enumerate(combined_means):
        ax3.text(v + combined_errors[i] + 0.01, i, f"{v:.3f}", va="center", fontsize=8)

    # Add styling
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(all_types)
    ax3.set_xlabel("Normalized Importance")
    ax3.set_title("C) Layer Type Sensitivity", fontsize=12)

    # Add a legend for encoder/decoder
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=palette[0], label="Encoder"),
        Patch(facecolor=palette[1], label="Decoder"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right")

    # --- Panel 4: Pruning Recommendation Heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Create a matrix of recommended pruning rates based on sensitivity
    # Higher sensitivity = lower pruning rate

    # Identify all layer positions and types
    positions = ["Early", "Middle", "Late"]
    layer_categories = ["Self Attention", "Cross Attention", "Feed Forward", "Layer Norm"]

    # Initialize matrices for encoder and decoder - create two separate recommendation matrices
    encoder_pruning = np.zeros((len(positions), len(layer_categories)))
    decoder_pruning = np.zeros((len(positions), len(layer_categories)))

    # Helper function to calculate pruning rate based purely on importance
    def calc_pruning_rate(importance, max_imp=None):
        if max_imp is None or max_imp == 0:
            return 0.5  # Default when no data available

        # Normalize importance
        normalized = importance / max_imp if max_imp > 0 else 0

        # Apply sigmoid function for better separation between important and less important
        sigmoid_val = 1.0 / (1.0 + np.exp(-8 * (normalized - 0.5)))

        # Map to pruning rates: higher importance -> lower pruning rate
        # Scale to reasonable pruning range (10% to 70%)
        return 0.7 - (sigmoid_val * 0.6)

    # Find the maximum importance value across all components for normalization
    all_importances = [mean for mean in combined_means if mean > 0]
    max_importance = max(all_importances) if all_importances else 1.0

    # Map layer types from our data structure to matrix columns
    layer_type_map = {
        "self_attention": 0,  # Self Attention column
        "cross_attention": 1,  # Cross Attention column
        "feed_forward": 2,  # Feed Forward column
        "layer_norm": 3,  # Layer Norm column
    }

    # Process both encoder and decoder components
    for component in ["encoder", "decoder"]:
        pruning_matrix = encoder_pruning if component == "encoder" else decoder_pruning

        for pos_idx, pos_name in enumerate(["early", "middle", "late"]):
            for layer_type, values in layer_types[component].items():
                if not values:
                    continue

                # Get average importance for this component+position+type
                position_values = []
                for key, data in layer_importance.items():
                    if (
                        data["component"] == component
                        and data["layer_type"] == layer_type
                        and data["layer_num"] >= 0
                    ):
                        # Check if this layer belongs to the current position category
                        max_layer = max_enc_layer if component == "encoder" else max_dec_layer
                        layer_num = data["layer_num"]

                        if (
                            pos_name == "early"
                            and layer_num <= max_layer * 0.33
                            or pos_name == "middle"
                            and max_layer * 0.33 < layer_num <= max_layer * 0.67
                            or pos_name == "late"
                            and layer_num > max_layer * 0.67
                        ):
                            position_values.append(data["avg_importance"])

                # Calculate average importance for this specific position+type
                avg_imp = np.mean(position_values) if position_values else 0

                # Calculate pruning rate based solely on importance
                if layer_type in layer_type_map:
                    col_idx = layer_type_map[layer_type]
                    pruning_matrix[pos_idx, col_idx] = calc_pruning_rate(avg_imp, max_importance)

    # Fill in any remaining zeros with reasonable defaults
    for matrix in [encoder_pruning, decoder_pruning]:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    # Use a default value of 50% pruning
                    matrix[i, j] = 0.5

    # Choose decoder for the pruning recommendation heatmap display
    pruning_matrix = decoder_pruning

    # Plot heatmap
    im = ax4.imshow(pruning_matrix, cmap=prune_cmap, vmin=0.1, vmax=0.7)

    # Add percentage annotations
    for i in range(len(positions)):
        for j in range(len(layer_categories)):
            if j < pruning_matrix.shape[1]:  # Check if column exists
                text = ax4.text(
                    j,
                    i,
                    f"{pruning_matrix[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color="black" if pruning_matrix[i, j] < 0.5 else "white",
                    fontsize=10,
                    fontweight="bold",
                )

    # Add styling
    ax4.set_xticks(np.arange(len(layer_categories)))
    ax4.set_yticks(np.arange(len(positions)))
    ax4.set_xticklabels(layer_categories)
    ax4.set_yticklabels(positions)
    ax4.set_title("D) Recommended Pruning Rates (Decoder)", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, label="Pruning Rate")
    cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    cbar.set_ticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"])

    # Add pruning strategy explanation
    footnote = "Pruning recommendations are based on parameter sensitivity analysis.\n"
    footnote += "Lower pruning rates (lighter colors) preserve more important components."
    plt.figtext(
        0.5,
        0.01,
        footnote,
        ha="center",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"),
    )

    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as high-resolution image
    plt.savefig(os.path.join(output_dir, "sensitivity_summary.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "sensitivity_summary.pdf"), bbox_inches="tight")

    # Close the figure
    plt.close(fig)

    print(f"Created summary plot: {os.path.join(output_dir, 'sensitivity_summary.png')}")

    # Save pruning recommendations as JSON
    pruning_recommendations = {"encoder": {}, "decoder": {}}

    # Format the pruning recommendations data
    for position_idx, position in enumerate(positions):
        for layer_type_idx, layer_type in enumerate(layer_categories):
            # Convert from UI-friendly names back to code names
            code_layer_type = layer_type.lower().replace(" ", "_")
            position_lower = position.lower()

            # Store encoder recommendations
            if layer_type_idx < encoder_pruning.shape[1]:
                if encoder_pruning[position_idx, layer_type_idx] > 0:
                    if position_lower not in pruning_recommendations["encoder"]:
                        pruning_recommendations["encoder"][position_lower] = {}
                    pruning_recommendations["encoder"][position_lower][code_layer_type] = float(
                        encoder_pruning[position_idx, layer_type_idx]
                    )

            # Store decoder recommendations
            if layer_type_idx < decoder_pruning.shape[1]:
                if decoder_pruning[position_idx, layer_type_idx] > 0:
                    if position_lower not in pruning_recommendations["decoder"]:
                        pruning_recommendations["decoder"][position_lower] = {}
                    pruning_recommendations["decoder"][position_lower][code_layer_type] = float(
                        decoder_pruning[position_idx, layer_type_idx]
                    )

    # Save pruning recommendations
    with open(os.path.join(output_dir, "pruning_recommendations.json"), "w") as f:
        json.dump(pruning_recommendations, f, indent=2)


def create_detailed_layer_plots(
    sensitivity_results: Dict, output_dir: str, model_info: Optional[Dict] = None
) -> None:
    """
    Create detailed layer-by-layer visualization plots for research.

    Args:
        sensitivity_results: Results from compute_sensitivity_mps
        output_dir: Directory to save the plots
        model_info: Dictionary with model metadata
    """
    layer_importance = sensitivity_results["layer_importance"]

    # Configure aesthetics for publication quality
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )

    # Use a colorblind-friendly palette
    palette = sns.color_palette("colorblind")

    # Get model name for title
    model_name = model_info["name"] if model_info else "Whisper"

    # ======================= #
    # 1. Layer-by-Layer Plot
    # ======================= #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 1])

    # Prepare encoder layer data
    encoder_layers = {}
    for key, data in layer_importance.items():
        if data["component"] == "encoder" and data["layer_num"] >= 0:
            layer_num = data["layer_num"]
            if layer_num not in encoder_layers:
                encoder_layers[layer_num] = []
            encoder_layers[layer_num].append(data["avg_importance"])

    # Calculate average importance per encoder layer
    encoder_x = sorted(encoder_layers.keys())
    encoder_y = [np.mean(encoder_layers[layer]) for layer in encoder_x]
    encoder_err = [
        np.std(encoder_layers[layer]) / np.sqrt(len(encoder_layers[layer]))
        if len(encoder_layers[layer]) > 1
        else 0
        for layer in encoder_x
    ]

    # Prepare decoder layer data
    decoder_layers = {}
    for key, data in layer_importance.items():
        if data["component"] == "decoder" and data["layer_num"] >= 0:
            layer_num = data["layer_num"]
            if layer_num not in decoder_layers:
                decoder_layers[layer_num] = []
            decoder_layers[layer_num].append(data["avg_importance"])

    # Calculate average importance per decoder layer
    decoder_x = sorted(decoder_layers.keys())
    decoder_y = [np.mean(decoder_layers[layer]) for layer in decoder_x]
    decoder_err = [
        np.std(decoder_layers[layer]) / np.sqrt(len(decoder_layers[layer]))
        if len(decoder_layers[layer]) > 1
        else 0
        for layer in decoder_x
    ]

    # Plot encoder layers
    ax1.errorbar(
        encoder_x,
        encoder_y,
        yerr=encoder_err,
        marker="o",
        linestyle="-",
        color=palette[0],
        capsize=4,
        label="Encoder Layers",
    )

    # Add value labels
    for i, (x, y) in enumerate(zip(encoder_x, encoder_y)):
        ax1.text(x, y + encoder_err[i] + 0.02, f"{y:.3f}", ha="center", fontsize=8)

    # Add styling
    ax1.set_xlabel("Layer Number")
    ax1.set_ylabel("Normalized Importance")
    ax1.set_title("Encoder Layer-by-Layer Sensitivity", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_xticks(encoder_x)

    # Highlight first and last layers
    if encoder_x:
        first_layer = encoder_x[0]
        last_layer = encoder_x[-1]
        ax1.axvspan(first_layer - 0.5, first_layer + 0.5, alpha=0.2, color="green")
        ax1.axvspan(last_layer - 0.5, last_layer + 0.5, alpha=0.2, color="red")
        ax1.text(
            first_layer,
            max(encoder_y) * 1.1,
            "First Layer",
            ha="center",
            color="green",
            fontweight="bold",
        )
        ax1.text(
            last_layer,
            max(encoder_y) * 1.1,
            "Last Layer",
            ha="center",
            color="red",
            fontweight="bold",
        )

    # Plot decoder layers
    ax2.errorbar(
        decoder_x,
        decoder_y,
        yerr=decoder_err,
        marker="o",
        linestyle="-",
        color=palette[1],
        capsize=4,
        label="Decoder Layers",
    )

    # Add value labels
    for i, (x, y) in enumerate(zip(decoder_x, decoder_y)):
        ax2.text(x, y + decoder_err[i] + 0.02, f"{y:.3f}", ha="center", fontsize=8)

    # Add styling
    ax2.set_xlabel("Layer Number")
    ax2.set_ylabel("Normalized Importance")
    ax2.set_title("Decoder Layer-by-Layer Sensitivity", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_xticks(decoder_x)

    # Highlight first and last layers
    if decoder_x:
        first_layer = decoder_x[0]
        last_layer = decoder_x[-1]
        ax2.axvspan(first_layer - 0.5, first_layer + 0.5, alpha=0.2, color="green")
        ax2.axvspan(last_layer - 0.5, last_layer + 0.5, alpha=0.2, color="red")
        ax2.text(
            first_layer,
            max(decoder_y) * 1.1,
            "First Layer",
            ha="center",
            color="green",
            fontweight="bold",
        )
        ax2.text(
            last_layer,
            max(decoder_y) * 1.1,
            "Last Layer",
            ha="center",
            color="red",
            fontweight="bold",
        )

    # Add common title
    plt.suptitle(f"Layer-by-Layer Sensitivity Analysis: {model_name}", fontsize=14)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, "layer_by_layer_sensitivity.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, "layer_by_layer_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Other detailed visualizations are created in similar fashion (omitted for brevity)
    # We'll focus on the main layer-by-layer plot since it's the most informative


def generate_pruning_config(
    recommendations_path: str, model_name: str, output_path: Optional[str] = None
) -> str:
    """
    Generate a pruning configuration file from sensitivity analysis results
    that can be used for actual model pruning.

    Args:
        recommendations_path: Path to pruning_recommendations.json
        model_name: Name of the Whisper model
        output_path: Path to save the pruning config (defaults to same directory)

    Returns:
        Path to the generated pruning configuration file
    """
    # Load recommendations
    with open(recommendations_path) as f:
        recommendations = json.load(f)

    # Load model info
    model_info_path = os.path.join(os.path.dirname(recommendations_path), "model_info.json")

    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            model_info = json.load(f)

        num_encoder_layers = model_info["num_layers"]["encoder"]
        num_decoder_layers = model_info["num_layers"]["decoder"]
    else:
        # Default values for Whisper models if info not available
        print("Model info not found, using default values based on model name")
        whisper_sizes = {"tiny": 4, "base": 6, "small": 12, "medium": 24, "large": 32}

        # Extract size from model name
        found_size = False
        for size in whisper_sizes:
            if size in model_name:
                num_encoder_layers = whisper_sizes[size]
                num_decoder_layers = whisper_sizes[size] if size not in ["tiny", "base"] else 4
                found_size = True
                break

        if not found_size:
            # Default to small if no match
            print("Could not determine model size from name, defaulting to small")
            num_encoder_layers = 12
            num_decoder_layers = 12

    # Create pruning configuration
    pruning_config = {
        "model_name": model_name,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "method": "mps_gradient_sensitivity",
        "pruning_settings": {"encoder": [], "decoder": []},
    }

    # Helper function to get position category
    def get_position(layer_idx, total_layers):
        if layer_idx <= total_layers * 0.33:
            return "early"
        elif layer_idx <= total_layers * 0.67:
            return "middle"
        else:
            return "late"

    # Generate encoder layer configs
    for layer_idx in range(num_encoder_layers):
        position = get_position(layer_idx, num_encoder_layers)
        layer_config = {"layer_idx": layer_idx, "pruning_rates": {}}

        # Add pruning rates for each layer type
        for layer_type in ["self_attention", "feed_forward", "layer_norm"]:
            if (
                position in recommendations["encoder"]
                and layer_type in recommendations["encoder"][position]
            ):
                layer_config["pruning_rates"][layer_type] = recommendations["encoder"][position][
                    layer_type
                ]

        pruning_config["pruning_settings"]["encoder"].append(layer_config)

    # Generate decoder layer configs
    for layer_idx in range(num_decoder_layers):
        position = get_position(layer_idx, num_decoder_layers)
        layer_config = {"layer_idx": layer_idx, "pruning_rates": {}}

        # Add pruning rates for each layer type
        for layer_type in ["self_attention", "cross_attention", "feed_forward", "layer_norm"]:
            if (
                position in recommendations["decoder"]
                and layer_type in recommendations["decoder"][position]
            ):
                layer_config["pruning_rates"][layer_type] = recommendations["decoder"][position][
                    layer_type
                ]

        pruning_config["pruning_settings"]["decoder"].append(layer_config)

    # Save pruning configuration
    if output_path is None:
        output_dir = os.path.dirname(recommendations_path)
        output_path = os.path.join(output_dir, "pruning_config.json")

    with open(output_path, "w") as f:
        json.dump(pruning_config, f, indent=2)

    print(f"Generated pruning configuration: {output_path}")
    return output_path


def run_mps_sensitivity_analysis(
    model_name: str,
    output_dir: Optional[str] = None,
    num_samples: int = 1000,
    num_batches: int = 5,
    split: str = "test.other",
) -> Tuple[str, Dict]:
    """
    Run a sensitivity analysis for Whisper model optimized for MPS (Apple Silicon).

    Args:
        model_name: The Whisper model name ("openai/whisper-small")
        output_dir: Output directory (defaults to timestamped directory)
        num_samples: Number of samples to use from LibriSpeech
        num_batches: Number of batches to use for sensitivity computation
        split: Dataset split to use ("test.clean" or "test.other")

    Returns:
        Tuple of (output_dir, sensitivity_results)
    """
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"whisper_mps_sensitivity_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = get_device()
    print("\n=== Whisper Sensitivity Analysis (MPS Optimized) ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {split}")
    print(f"Samples: {num_samples}")
    print(f"Results will be saved to: {output_dir}")

    # 1. Load model and processor
    print("\nLoading model and processor...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        processor = WhisperProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU")
        device = torch.device("cpu")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        processor = WhisperProcessor.from_pretrained(model_name)

    # Save model architecture details
    # Get the model config properties
    config = model.config

    # Extract encoder and decoder layers count safely
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

    # 2. Load and process dataset - with MPS optimizations
    print(f"\nLoading {split} dataset with {num_samples} samples...")

    try:
        # Process in smaller chunks to avoid memory issues
        dataset = load_librispeech(num_samples=num_samples, split=split)
        print("\nProcessing dataset...")

        # Process each sample individually to avoid batch collation issues
        processed_samples = []
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Get audio features
                audio = sample["audio"]
                input_features = processor(
                    audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
                ).input_features

                # Get text features
                text = sample["text"]
                text_encoding = processor(text=text, return_tensors="pt")

                processed_samples.append(
                    {
                        "input_features": input_features,
                        "text_ids": text_encoding.input_ids,
                        "text": text,
                    }
                )

                # Clear memory periodically
                if i % 100 == 0:
                    clear_memory()

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # 3. Compute sensitivity metrics with MPS optimizations
    print("\nComputing parameter sensitivity (MPS optimized)...")
    sensitivity_results = compute_sensitivity_mps(
        model=model,
        processor=processor,
        samples=processed_samples,
        device=device,
        model_name=model_name,
        model_info=model_info,
        num_batches=num_batches,
    )

    # 4. Save results
    print("\nSaving sensitivity results...")
    parameter_importance = sensitivity_results["parameter_importance"]
    layer_importance = sensitivity_results["layer_importance"]

    # Save parameter importance
    with open(os.path.join(output_dir, "parameter_importance.json"), "w") as f:
        json.dump(parameter_importance, f, indent=2)

    # Save layer importance
    with open(os.path.join(output_dir, "layer_importance.json"), "w") as f:
        json.dump(layer_importance, f, indent=2)

    # 5. Create summary visualization
    print("\nCreating visualizations...")
    create_summary_plot(sensitivity_results, output_dir, model_info)

    # 6. Create detailed layer-by-layer visualization
    create_detailed_layer_plots(sensitivity_results, output_dir, model_info)

    # 7. Generate pruning config
    recommendations_path = os.path.join(output_dir, "pruning_recommendations.json")
    if os.path.exists(recommendations_path):
        pruning_config_path = generate_pruning_config(
            recommendations_path=recommendations_path, model_name=model_name
        )
        print(f"Generated pruning configuration at: {pruning_config_path}")

    return output_dir, sensitivity_results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Whisper Model Sensitivity Analysis for MPS")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Model name (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to use (default: 1000)"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=5,
        help="Number of batches for sensitivity computation (default: 5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test.other",
        help="Dataset split (test.clean or test.other, default: test.other)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: timestamped directory)",
    )
    return parser.parse_args()


def main():
    """Main function to run sensitivity analysis optimized for MPS"""
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Run sensitivity analysis
    results_dir, sensitivity_results = run_mps_sensitivity_analysis(
        model_name=args.model,
        num_samples=args.samples,
        num_batches=args.batches,
        split=args.split,
        output_dir=args.output_dir,
    )

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"MPS-optimized sensitivity analysis complete. Results saved to: {results_dir}")
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    print("\nKey outputs:")
    print("- sensitivity_summary.png/pdf: Visual summary for research paper inclusion")
    print("- layer_by_layer_sensitivity.png/pdf: Detailed layer-by-layer analysis")
    print("- pruning_recommendations.json: Structured pruning recommendations")
    print("- pruning_config.json: Ready-to-use pruning configuration for implementation")


if __name__ == "__main__":
    main()
