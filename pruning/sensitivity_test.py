"""
Whisper Model Sensitivity Analysis and Pruning Tool
===================================================

This module provides tools to analyze the parameter sensitivity of Whisper
speech recognition models and generate pruning configurations based on
the analysis results.

The tool performs the following steps:
1. Run sensitivity analysis on a Whisper model using LibriSpeech dataset
2. Generate visualizations of the sensitivity metrics
3. Create pruning recommendations based on the analysis
4. Output a pruning configuration file for model compression

Usage:
    python whisper_sensitivity.py --model openai/whisper-small --samples 1000 --batches 30
"""

import argparse
import gc
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

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


def clear_gpu_memory():
    """Clear cached GPU memory to avoid OOM errors during processing"""
    gc.collect()

    if torch.cuda.is_available():
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


def compute_enhanced_sensitivity(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    samples: List[Dict],
    device: torch.device,
    model_name: str,
    model_info: Dict,  # Add model_info parameter
    num_batches: int = 30,
    batch_size: int = 1,
) -> Dict:
    """
    Enhanced sensitivity calculation with specific adjustments for MLP layers
    """
    # Initialize sensitivity metrics
    sensitivity_results = {}

    # Initialize parameter tracking
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:  # Focus on weight matrices
            category = categorize_parameter(name)

            sensitivity_results[name] = {
                "shape": list(param.shape),
                "size": param.numel(),
                "category": category,
                "gradients": [],  # Store actual gradients for better statistics
                "param_values": param.detach()
                .cpu()
                .numpy()
                .flatten()
                .tolist()[:5],  # Store sample of values for debugging
                "batch_count": 0,
            }

    # Set model to evaluation mode but enable gradients
    model.eval()

    # Process batches
    batch_count = 0
    sample_idx = 0
    pbar = tqdm(total=num_batches, desc="Computing sensitivity")

    # Use different samples to get better gradient diversity
    while batch_count < num_batches and sample_idx < len(samples):
        # Process a single batch
        try:
            # Get batch of samples
            batch_size_actual = min(batch_size, len(samples) - sample_idx)
            batch_samples = samples[sample_idx : sample_idx + batch_size_actual]
            sample_idx += batch_size_actual

            # Process each sample in the batch
            for sample in batch_samples:
                # Get features and move to device
                features = sample["input_features"].to(device)
                target_ids = sample["text_ids"].to(device)

                # Forward pass with enabled gradients
                model.zero_grad()

                # Use forced decoding to engage all parts of the model
                outputs = model(
                    input_features=features,
                    decoder_input_ids=target_ids[:, :-1] if target_ids.size(1) > 1 else None,
                    labels=target_ids if target_ids.size(1) <= 1 else target_ids[:, 1:],
                )

                loss = outputs.loss
                loss.backward()

                # Accumulate sensitivity metrics
                for name, param in model.named_parameters():
                    if name in sensitivity_results and param.grad is not None:
                        # Store gradient statistics
                        grad = param.grad.detach().cpu()

                        # Calculate metrics for sensitivity
                        fisher = float((grad**2).mean().item())
                        snip = float((param.detach() * grad).abs().mean().item())

                        # Store in results
                        sensitivity_results[name]["gradients"].append(
                            {
                                "fisher": fisher,
                                "snip": snip,
                                "loss": float(loss.item()),
                            }
                        )

                        # Update count
                        sensitivity_results[name]["batch_count"] += 1

                # Clear memory
                del features, target_ids, outputs, loss

            # Update batch counter
            batch_count += 1
            pbar.update(1)

            # Clear memory between batches
            clear_gpu_memory()

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Skip to next batch
            sample_idx += batch_size

    pbar.close()

    # Calculate average importance
    print("\nCalculating importance scores...")

    # Parameter importance
    parameter_importance = {}

    # Get layer count information for proper scaling
    num_encoder_layers = model_info["num_layers"]["encoder"]
    num_decoder_layers = model_info["num_layers"]["decoder"]

    for name, data in tqdm(sensitivity_results.items(), desc="Processing parameters"):
        if data["batch_count"] > 0:
            # Get component information
            category = data["category"]
            component = category["component"]
            layer_type = category["layer_type"]

            # Calculate metrics from gradient statistics
            fisher_values = [g["fisher"] for g in data["gradients"]]
            snip_values = [g["snip"] for g in data["gradients"]]

            fisher_mean = np.mean(fisher_values) if fisher_values else 0
            snip_mean = np.mean(snip_values) if snip_values else 0

            # Balance between metrics - SNIP is better for speech models
            raw_importance = 0.25 * fisher_mean + 0.75 * snip_mean

            # Apply domain-specific scaling factors
            importance = raw_importance

            # 1. Encoder boost
            if component == "encoder":
                importance *= 3.0  # Stronger boost for encoder

            # 2. Layer type boosts
            if layer_type == "cross_attention":
                importance *= 3.0  # Critical for encoder-decoder connection
            elif layer_type == "self_attention":
                importance *= 1.5  # Important but secondary
            elif layer_type == "feed_forward":  # MLP-specific adjustments
                # Apply position-based scaling for MLP layers
                layer_num = category["layer_num"]
                if layer_num >= 0:
                    if component == "encoder":
                        # First MLP layer is critical for initial feature transformation
                        if layer_num == 0:
                            importance *= 1.7
                        # Last MLP layer prepares features for decoder
                        elif layer_num == num_encoder_layers - 1:
                            importance *= 1.5
                        # Early MLP layers (feature extraction)
                        elif layer_num < num_encoder_layers // 3:
                            importance *= 1.3
                    elif component == "decoder":
                        # First decoder MLP is important for initial transform of cross-attention
                        if layer_num == 0:
                            importance *= 1.6
                        # Last decoder MLP directly impacts output generation
                        elif layer_num == num_decoder_layers - 1:
                            importance *= 2.2  # Significantly higher importance
                        # Late decoder MLPs handle final representations
                        elif layer_num > 2 * num_decoder_layers // 3:
                            importance *= 1.7

            # 3. Position-based importance (for other layer types)
            layer_num = category["layer_num"]
            if layer_num >= 0 and layer_type != "feed_forward":  # Skip MLPs (handled above)
                if component == "encoder":
                    # First layer for feature extraction
                    if layer_num == 0:
                        importance *= 1.8
                    # Early layers
                    elif layer_num < 3:
                        importance *= 1.5
                    # Last layer (connects to decoder)
                    elif layer_num == num_encoder_layers - 1:
                        importance *= 1.4
                elif component == "decoder":
                    # First decoder layer
                    if layer_num == 0:
                        importance *= 1.6
                    # Last layer (produces final outputs)
                    elif layer_num == num_decoder_layers - 1:
                        importance *= 2.0

            # Store importance statistics
            parameter_importance[name] = {
                "category": data["category"],
                "shape": data["shape"],
                "size": data["size"],
                "param_sample": data["param_values"],
                "importance": importance,
                "importance_raw": raw_importance,
                "fisher": fisher_mean,
                "snip": snip_mean,
                "gradient_count": len(data["gradients"]),
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
        sensitivity_results: Results from compute_sensitivity
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
    fig.suptitle(f"Sensitivity Analysis Summary: {model_name}", fontsize=14)

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

    # Helper function to calculate pruning rate with improved non-linear scaling
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

                # Apply domain-specific knowledge to adjust importance
                # For cross-attention in decoder, increase importance
                if component == "decoder" and layer_type == "cross_attention":
                    avg_imp *= 1.5

                # For encoder self-attention, especially in early layers, increase importance
                if (
                    component == "encoder"
                    and layer_type == "self_attention"
                    and pos_name == "early"
                ):
                    avg_imp *= 1.3

                # Calculate pruning rate and store in matrix
                if layer_type in layer_type_map:
                    col_idx = layer_type_map[layer_type]
                    pruning_matrix[pos_idx, col_idx] = calc_pruning_rate(avg_imp, max_importance)

    # Special handling for cross-attention in decoder (cap at 40% max)
    # This is crucial for Whisper as cross-attention connects encoder to decoder
    cross_attn_col = layer_type_map["cross_attention"]
    for row in range(decoder_pruning.shape[0]):
        if decoder_pruning[row, cross_attn_col] > 0.4:
            decoder_pruning[row, cross_attn_col] = 0.4

    # Fill in any remaining zeros with domain-specific defaults
    for matrix in [encoder_pruning, decoder_pruning]:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    # Use a default value based on column (layer type) and domain knowledge
                    if j == 0:  # Self Attention
                        matrix[i, j] = 0.3  # More conservative
                    elif j == 1:  # Cross Attention
                        matrix[i, j] = 0.25  # Very conservative
                    elif j == 2:  # Feed Forward
                        matrix[i, j] = 0.5  # Moderate
                    elif j == 3:  # Layer Norm
                        matrix[i, j] = 0.6  # More aggressive

    # Choose decoder for the pruning recommendation heatmap display
    # In practice, we'd use both matrices for actual pruning decisions
    pruning_matrix = decoder_pruning

    # Plot heatmap
    im = ax4.imshow(pruning_matrix, cmap=prune_cmap, vmin=0.1, vmax=0.7)

    # Add percentage annotations
    for i in range(len(positions)):
        for j in range(len(layer_categories)):
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
    footnote = "Pruning recommendations are derived from sensitivity analysis.\n"
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

    print(
        f"Created research paper summary plot: {os.path.join(output_dir, 'sensitivity_summary.png')}"
    )

    # Also save pruning recommendations as JSON for practical use
    pruning_recommendations = {"encoder": {}, "decoder": {}}

    # Format the pruning recommendations data
    for position_idx, position in enumerate(positions):
        for layer_type_idx, layer_type in enumerate(layer_categories):
            # Convert from UI-friendly names back to code names
            code_layer_type = layer_type.lower().replace(" ", "_")
            position_lower = position.lower()

            # Store encoder recommendations
            if encoder_pruning[position_idx, layer_type_idx] > 0:
                if position_lower not in pruning_recommendations["encoder"]:
                    pruning_recommendations["encoder"][position_lower] = {}
                pruning_recommendations["encoder"][position_lower][code_layer_type] = float(
                    encoder_pruning[position_idx, layer_type_idx]
                )

            # Store decoder recommendations
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
        sensitivity_results: Results from compute_sensitivity
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

    # ======================= #
    # 2. Layer Type Comparison by Position
    # ======================= #
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Layer types to analyze
    encoder_layer_types = ["self_attention", "feed_forward", "layer_norm"]
    decoder_layer_types = ["self_attention", "cross_attention", "feed_forward", "layer_norm"]

    # Prepare data structure
    enc_layer_type_data = {lt: [] for lt in encoder_layer_types}
    dec_layer_type_data = {lt: [] for lt in decoder_layer_types}

    # Find max layer numbers
    max_enc_layer = max(
        [
            data["layer_num"]
            for key, data in layer_importance.items()
            if data["component"] == "encoder" and data["layer_num"] >= 0
        ],
        default=0,
    )
    max_dec_layer = max(
        [
            data["layer_num"]
            for key, data in layer_importance.items()
            if data["component"] == "decoder" and data["layer_num"] >= 0
        ],
        default=0,
    )

    # Group data by layer type and position
    for key, data in layer_importance.items():
        component = data["component"]
        layer_type = data["layer_type"]
        layer_num = data["layer_num"]

        if layer_num < 0:
            continue

        if component == "encoder" and layer_type in encoder_layer_types:
            position = (
                "Early"
                if layer_num <= max_enc_layer * 0.33
                else "Middle"
                if layer_num <= max_enc_layer * 0.67
                else "Late"
            )
            enc_layer_type_data[layer_type].append((position, data["avg_importance"]))

        elif component == "decoder" and layer_type in decoder_layer_types:
            position = (
                "Early"
                if layer_num <= max_dec_layer * 0.33
                else "Middle"
                if layer_num <= max_dec_layer * 0.67
                else "Late"
            )
            dec_layer_type_data[layer_type].append((position, data["avg_importance"]))

    # Process encoder data
    enc_positions = ["Early", "Middle", "Late"]
    enc_width = 0.15
    x = np.arange(len(enc_positions))

    for i, layer_type in enumerate(encoder_layer_types):
        means = []
        errors = []

        for pos in enc_positions:
            pos_values = [imp for p, imp in enc_layer_type_data[layer_type] if p == pos]
            means.append(np.mean(pos_values) if pos_values else 0)
            errors.append(
                np.std(pos_values) / np.sqrt(len(pos_values)) if len(pos_values) > 1 else 0
            )

        offset = (i - len(encoder_layer_types) / 2 + 0.5) * enc_width
        axs[0].bar(
            x + offset,
            means,
            width=enc_width,
            label=layer_type.replace("_", " ").title(),
            color=palette[i],
            yerr=errors,
            capsize=3,
        )

        # Add value labels
        for j, v in enumerate(means):
            if v > 0:
                axs[0].text(j + offset, v + errors[j] + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    # Process decoder data
    dec_positions = ["Early", "Middle", "Late"]
    dec_width = 0.15
    x = np.arange(len(dec_positions))

    for i, layer_type in enumerate(decoder_layer_types):
        means = []
        errors = []

        for pos in dec_positions:
            pos_values = [imp for p, imp in dec_layer_type_data[layer_type] if p == pos]
            means.append(np.mean(pos_values) if pos_values else 0)
            errors.append(
                np.std(pos_values) / np.sqrt(len(pos_values)) if len(pos_values) > 1 else 0
            )

        offset = (i - len(decoder_layer_types) / 2 + 0.5) * dec_width
        axs[1].bar(
            x + offset,
            means,
            width=dec_width,
            label=layer_type.replace("_", " ").title(),
            color=palette[i],
            yerr=errors,
            capsize=3,
        )

        # Add value labels
        for j, v in enumerate(means):
            if v > 0:
                axs[1].text(j + offset, v + errors[j] + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    # Add styling
    axs[0].set_xlabel("Layer Position")
    axs[0].set_ylabel("Normalized Importance")
    axs[0].set_title("Encoder: Layer Type Sensitivity by Position", fontsize=12)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(enc_positions)
    axs[0].legend()
    axs[0].grid(axis="y", linestyle="--", alpha=0.7)

    axs[1].set_xlabel("Layer Position")
    axs[1].set_ylabel("Normalized Importance")
    axs[1].set_title("Decoder: Layer Type Sensitivity by Position", fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(dec_positions)
    axs[1].legend()
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    # Add common title
    plt.suptitle(f"Layer Type Sensitivity Analysis by Position: {model_name}", fontsize=14)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, "layer_type_by_position.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, "layer_type_by_position.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ======================= #
    # 3. First vs Last Layer Detailed Comparison
    # ======================= #
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Prepare data for first vs last layer comparison
    enc_first_data = {}
    enc_last_data = {}
    dec_first_data = {}
    dec_last_data = {}

    for key, data in layer_importance.items():
        component = data["component"]
        layer_type = data["layer_type"]
        layer_num = data["layer_num"]

        if layer_num < 0:
            continue

        if component == "encoder":
            if layer_num == 0:  # First layer
                if layer_type not in enc_first_data:
                    enc_first_data[layer_type] = []
                enc_first_data[layer_type].append(data["avg_importance"])
            elif layer_num == max_enc_layer:  # Last layer
                if layer_type not in enc_last_data:
                    enc_last_data[layer_type] = []
                enc_last_data[layer_type].append(data["avg_importance"])

        elif component == "decoder":
            if layer_num == 0:  # First layer
                if layer_type not in dec_first_data:
                    dec_first_data[layer_type] = []
                dec_first_data[layer_type].append(data["avg_importance"])
            elif layer_num == max_dec_layer:  # Last layer
                if layer_type not in dec_last_data:
                    dec_last_data[layer_type] = []
                dec_last_data[layer_type].append(data["avg_importance"])

    # Plot encoder first layer
    enc_first_types = list(enc_first_data.keys())
    enc_first_means = [np.mean(enc_first_data[lt]) for lt in enc_first_types]
    enc_first_errors = [
        np.std(enc_first_data[lt]) / np.sqrt(len(enc_first_data[lt]))
        if len(enc_first_data[lt]) > 1
        else 0
        for lt in enc_first_types
    ]

    axs[0, 0].bar(
        enc_first_types,
        enc_first_means,
        yerr=enc_first_errors,
        color=[palette[i] for i in range(len(enc_first_types))],
        capsize=4,
    )
    axs[0, 0].set_title("Encoder First Layer (Layer 0)")
    axs[0, 0].set_ylabel("Normalized Importance")
    axs[0, 0].set_xticklabels([lt.replace("_", " ").title() for lt in enc_first_types])
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(enc_first_means):
        axs[0, 0].text(i, v + enc_first_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Plot encoder last layer
    enc_last_types = list(enc_last_data.keys())
    enc_last_means = [np.mean(enc_last_data[lt]) for lt in enc_last_types]
    enc_last_errors = [
        np.std(enc_last_data[lt]) / np.sqrt(len(enc_last_data[lt]))
        if len(enc_last_data[lt]) > 1
        else 0
        for lt in enc_last_types
    ]

    axs[0, 1].bar(
        enc_last_types,
        enc_last_means,
        yerr=enc_last_errors,
        color=[palette[i] for i in range(len(enc_last_types))],
        capsize=4,
    )
    axs[0, 1].set_title(f"Encoder Last Layer (Layer {max_enc_layer})")
    axs[0, 1].set_ylabel("Normalized Importance")
    axs[0, 1].set_xticklabels([lt.replace("_", " ").title() for lt in enc_last_types])
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(enc_last_means):
        axs[0, 1].text(i, v + enc_last_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Plot decoder first layer
    dec_first_types = list(dec_first_data.keys())
    dec_first_means = [np.mean(dec_first_data[lt]) for lt in dec_first_types]
    dec_first_errors = [
        np.std(dec_first_data[lt]) / np.sqrt(len(dec_first_data[lt]))
        if len(dec_first_data[lt]) > 1
        else 0
        for lt in dec_first_types
    ]

    axs[1, 0].bar(
        dec_first_types,
        dec_first_means,
        yerr=dec_first_errors,
        color=[palette[i] for i in range(len(dec_first_types))],
        capsize=4,
    )
    axs[1, 0].set_title("Decoder First Layer (Layer 0)")
    axs[1, 0].set_ylabel("Normalized Importance")
    axs[1, 0].set_xticklabels([lt.replace("_", " ").title() for lt in dec_first_types])
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(dec_first_means):
        axs[1, 0].text(i, v + dec_first_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Plot decoder last layer
    dec_last_types = list(dec_last_data.keys())
    dec_last_means = [np.mean(dec_last_data[lt]) for lt in dec_last_types]
    dec_last_errors = [
        np.std(dec_last_data[lt]) / np.sqrt(len(dec_last_data[lt]))
        if len(dec_last_data[lt]) > 1
        else 0
        for lt in dec_last_types
    ]

    axs[1, 1].bar(
        dec_last_types,
        dec_last_means,
        yerr=dec_last_errors,
        color=[palette[i] for i in range(len(dec_last_types))],
        capsize=4,
    )
    axs[1, 1].set_title(f"Decoder Last Layer (Layer {max_dec_layer})")
    axs[1, 1].set_ylabel("Normalized Importance")
    axs[1, 1].set_xticklabels([lt.replace("_", " ").title() for lt in dec_last_types])
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels
    for i, v in enumerate(dec_last_means):
        axs[1, 1].text(i, v + dec_last_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Add common title
    plt.suptitle(f"First vs Last Layer Comparison: {model_name}", fontsize=14)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "first_vs_last_layer.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "first_vs_last_layer.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Created additional visualizations for research paper")


def create_all_visualizations(
    sensitivity_results: Dict, output_dir: str, model_info: Optional[Dict] = None
) -> None:
    """
    Create all visualizations including the original summary plot, detailed plots,
    and MLP-specific visualizations.

    Args:
        sensitivity_results: Results from compute_sensitivity
        output_dir: Directory to save the plots
        model_info: Dictionary with model metadata
    """
    # Create the original summary plot
    create_summary_plot(sensitivity_results, output_dir, model_info)

    # Create the new detailed plots
    create_detailed_layer_plots(sensitivity_results, output_dir, model_info)

    # Create MLP-specific visualizations
    create_mlp_sensitivity_plot(sensitivity_results, output_dir, model_info)


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

    # Load model info instead of model
    # The model info is already saved during the sensitivity analysis
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
        "method": "sensitivity_analysis",
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


def create_mlp_sensitivity_plot(
    sensitivity_results: Dict, output_dir: str, model_info: Optional[Dict] = None
) -> None:
    """
    Create a dedicated visualization for MLP/Feed-Forward Network sensitivity
    across early, middle, and late layers.

    Args:
        sensitivity_results: Results from compute_sensitivity
        output_dir: Directory to save the plot
        model_info: Dictionary with model metadata
    """
    # Extract layer importance data
    layer_importance = sensitivity_results["layer_importance"]

    # Configure aesthetics
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    # Use a colorblind-friendly palette
    palette = sns.color_palette("colorblind")

    # Get model name for title
    model_name = model_info["name"] if model_info else "Whisper"

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract MLP-specific data
    mlp_data = {}
    for key, data in layer_importance.items():
        if data["layer_type"] == "feed_forward" and data["layer_num"] >= 0:
            component = data["component"]
            layer_num = data["layer_num"]

            if component not in mlp_data:
                mlp_data[component] = []

            mlp_data[component].append((layer_num, data["avg_importance"]))

    # Find max layer numbers for encoder and decoder
    max_enc_layer = max(
        [layer_num for layer_num, _ in mlp_data.get("encoder", [(0, 0)])], default=0
    )
    max_dec_layer = max(
        [layer_num for layer_num, _ in mlp_data.get("decoder", [(0, 0)])], default=0
    )

    # Determine early, middle, late ranges for encoder and decoder
    enc_ranges = {
        "Early": (0, max_enc_layer // 3),
        "Middle": (max_enc_layer // 3 + 1, 2 * max_enc_layer // 3),
        "Late": (2 * max_enc_layer // 3 + 1, max_enc_layer),
    }

    dec_ranges = {
        "Early": (0, max_dec_layer // 3),
        "Middle": (max_dec_layer // 3 + 1, 2 * max_dec_layer // 3),
        "Late": (2 * max_dec_layer // 3 + 1, max_dec_layer),
    }

    # Prepare data for bar chart
    enc_position_data = {"Early": [], "Middle": [], "Late": []}
    for layer_num, importance in mlp_data.get("encoder", []):
        for position, (start, end) in enc_ranges.items():
            if start <= layer_num <= end:
                enc_position_data[position].append(importance)
                break

    dec_position_data = {"Early": [], "Middle": [], "Late": []}
    for layer_num, importance in mlp_data.get("decoder", []):
        for position, (start, end) in dec_ranges.items():
            if start <= layer_num <= end:
                dec_position_data[position].append(importance)
                break

    # Calculate means and errors
    enc_positions = list(enc_position_data.keys())
    enc_means = [np.mean(values) if values else 0 for values in enc_position_data.values()]
    enc_errors = [
        np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
        for values in enc_position_data.values()
    ]

    dec_positions = list(dec_position_data.keys())
    dec_means = [np.mean(values) if values else 0 for values in dec_position_data.values()]
    dec_errors = [
        np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
        for values in dec_position_data.values()
    ]

    # Plot Encoder MLP sensitivity
    x_pos = np.arange(len(enc_positions))
    ax1.bar(x_pos, enc_means, yerr=enc_errors, capsize=5, color=palette[0])

    # Add value labels
    for i, v in enumerate(enc_means):
        ax1.text(i, v + enc_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Style the plot
    ax1.set_title("Encoder MLP/Feed-Forward Sensitivity", fontsize=12)
    ax1.set_xlabel("Layer Position")
    ax1.set_ylabel("Normalized Importance")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(enc_positions)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot Decoder MLP sensitivity
    x_pos = np.arange(len(dec_positions))
    ax2.bar(x_pos, dec_means, yerr=dec_errors, capsize=5, color=palette[1])

    # Add value labels
    for i, v in enumerate(dec_means):
        ax2.text(i, v + dec_errors[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Style the plot
    ax2.set_title("Decoder MLP/Feed-Forward Sensitivity", fontsize=12)
    ax2.set_xlabel("Layer Position")
    ax2.set_ylabel("Normalized Importance")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dec_positions)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add common title
    plt.suptitle(f"MLP/Feed-Forward Layer Sensitivity Analysis: {model_name}", fontsize=14)

    # Add detailed explanation
    explanation = (
        "Early layers tend to handle basic feature transformations, "
        "while late layers specialize in higher-level abstractions required for final output generation. "
        "This visualization shows how MLP sensitivity varies across layer positions."
    )
    plt.figtext(
        0.5,
        0.01,
        explanation,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    # Save the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "mlp_layer_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "mlp_layer_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Now create a layer-by-layer MLP comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Prepare data for layer-by-layer analysis
    enc_layers = {}
    dec_layers = {}

    for component, data_list in mlp_data.items():
        for layer_num, importance in data_list:
            if component == "encoder":
                if layer_num not in enc_layers:
                    enc_layers[layer_num] = []
                enc_layers[layer_num].append(importance)
            elif component == "decoder":
                if layer_num not in dec_layers:
                    dec_layers[layer_num] = []
                dec_layers[layer_num].append(importance)

    # Calculate means and errors
    enc_x = sorted(enc_layers.keys())
    enc_y = [np.mean(enc_layers[layer]) for layer in enc_x]
    enc_err = [
        np.std(enc_layers[layer]) / np.sqrt(len(enc_layers[layer]))
        if len(enc_layers[layer]) > 1
        else 0
        for layer in enc_x
    ]

    dec_x = sorted(dec_layers.keys())
    dec_y = [np.mean(dec_layers[layer]) for layer in dec_x]
    dec_err = [
        np.std(dec_layers[layer]) / np.sqrt(len(dec_layers[layer]))
        if len(dec_layers[layer]) > 1
        else 0
        for layer in dec_x
    ]

    # Plot encoder MLP by layer
    ax1.errorbar(
        enc_x,
        enc_y,
        yerr=enc_err,
        marker="o",
        linestyle="-",
        color=palette[0],
        capsize=4,
        label="MLP/Feed-Forward",
    )

    # Add value labels
    for i, (x, y) in enumerate(zip(enc_x, enc_y)):
        ax1.text(x, y + enc_err[i] + 0.02, f"{y:.3f}", ha="center", fontsize=8)

    # Style the plot
    ax1.set_title("Encoder MLP/Feed-Forward Layer-by-Layer Sensitivity", fontsize=12)
    ax1.set_xlabel("Layer Number")
    ax1.set_ylabel("Normalized Importance")
    ax1.set_xticks(enc_x)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Highlight first and last MLP layers
    if enc_x:
        first_layer = enc_x[0]
        last_layer = enc_x[-1]
        ax1.axvspan(first_layer - 0.5, first_layer + 0.5, alpha=0.2, color="green")
        ax1.axvspan(last_layer - 0.5, last_layer + 0.5, alpha=0.2, color="red")
        ax1.text(
            first_layer,
            max(enc_y) * 1.1,
            "First MLP",
            ha="center",
            color="green",
            fontweight="bold",
        )
        ax1.text(
            last_layer, max(enc_y) * 1.1, "Last MLP", ha="center", color="red", fontweight="bold"
        )

    # Plot decoder MLP by layer
    ax2.errorbar(
        dec_x,
        dec_y,
        yerr=dec_err,
        marker="o",
        linestyle="-",
        color=palette[1],
        capsize=4,
        label="MLP/Feed-Forward",
    )

    # Add value labels
    for i, (x, y) in enumerate(zip(dec_x, dec_y)):
        ax2.text(x, y + dec_err[i] + 0.02, f"{y:.3f}", ha="center", fontsize=8)

    # Style the plot
    ax2.set_title("Decoder MLP/Feed-Forward Layer-by-Layer Sensitivity", fontsize=12)
    ax2.set_xlabel("Layer Number")
    ax2.set_ylabel("Normalized Importance")
    ax2.set_xticks(dec_x)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Highlight first and last MLP layers
    if dec_x:
        first_layer = dec_x[0]
        last_layer = dec_x[-1]
        ax2.axvspan(first_layer - 0.5, first_layer + 0.5, alpha=0.2, color="green")
        ax2.axvspan(last_layer - 0.5, last_layer + 0.5, alpha=0.2, color="red")
        ax2.text(
            first_layer,
            max(dec_y) * 1.1,
            "First MLP",
            ha="center",
            color="green",
            fontweight="bold",
        )
        ax2.text(
            last_layer, max(dec_y) * 1.1, "Last MLP", ha="center", color="red", fontweight="bold"
        )

    # Add common title
    plt.suptitle(f"MLP/Feed-Forward Layer-by-Layer Sensitivity: {model_name}", fontsize=14)

    # Add detailed explanation
    explanation = (
        "This visualization shows the sensitivity of each MLP/Feed-Forward Network layer. "
        "The first layers often handle basic feature transformations, while final layers "
        "prepare representations for output generation."
    )
    plt.figtext(
        0.5,
        0.01,
        explanation,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
    )

    # Save the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "mlp_layer_by_layer.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "mlp_layer_by_layer.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Created MLP/Feed-Forward sensitivity visualizations")


def run_sensitivity_analysis(
    model_name: str,
    output_dir: Optional[str] = None,
    num_samples: int = 200,
    batch_size: int = 1,
    num_batches: int = 30,
    split: str = "test.other",
) -> Tuple[str, Dict]:
    """
    Run a sensitivity analysis for Whisper model with enhanced visualizations

    Args:
        model_name: The Whisper model name ("openai/whisper-small")
        output_dir: Output directory (defaults to timestamped directory)
        num_samples: Number of samples to use from LibriSpeech
        batch_size: Batch size for processing (use 1 to avoid collation issues)
        num_batches: Number of batches to use for sensitivity computation
        split: Dataset split to use ("test.clean" or "test.other")

    Returns:
        Tuple of (output_dir, sensitivity_results)
    """
    # Create output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"whisper_sensitivity_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Whisper Sensitivity Analysis ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {split}")
    print(f"Samples: {num_samples}")
    print(f"Results will be saved to: {output_dir}")

    # 1. Load model and processor
    print("\nLoading model and processor...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Save model architecture details
    # Get the model config properties
    config = model.config

    # Extract encoder and decoder layers count safely
    num_encoder_layers = getattr(config, "encoder_layers", 0)
    if num_encoder_layers == 0:
        # Try to find encoder through model directly using a more specific path
        try:
            # In Whisper, encoder is sometimes under 'model.encoder'
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

    # 2. Load and process dataset
    print(f"\nLoading {split} dataset with {num_samples} samples...")
    dataset = load_librispeech(num_samples=num_samples, split=split)
    print("\nProcessing dataset...")

    # Process each sample individually to avoid batch collation issues
    processed_samples = []
    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        # Get audio features
        audio = sample["audio"]
        input_features = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

        # Get text features
        text = sample["text"]
        text_encoding = processor(text=text, return_tensors="pt")

        processed_samples.append(
            {"input_features": input_features, "text_ids": text_encoding.input_ids, "text": text}
        )

    # 3. Compute sensitivity metrics with enhanced encoder importance
    print("\nComputing parameter sensitivity...")
    # Use the enhanced sensitivity function that better handles MLPs
    sensitivity_results = compute_enhanced_sensitivity(
        model=model,
        processor=processor,
        samples=processed_samples,
        device=device,
        model_name=model_name,
        model_info=model_info,  # Pass model_info to the function
        num_batches=num_batches,
        batch_size=batch_size,
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

    # 5. Create all visualizations
    print("\nCreating visualizations...")
    create_all_visualizations(sensitivity_results, output_dir, model_info)

    return output_dir, sensitivity_results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Whisper Model Sensitivity Analysis")
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
        default=30,
        help="Number of batches for sensitivity computation (default: 30)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing (default: 1)"
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
    """Main function to run sensitivity analysis and generate pruning config"""
    # Parse arguments
    args = parse_arguments()

    # Run sensitivity analysis with enhanced visualizations
    results_dir, sensitivity_results = run_sensitivity_analysis(
        model_name=args.model,
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_batches=args.batches,
        split=args.split,
        output_dir=args.output_dir,
    )
    print(f"Sensitivity analysis complete. Results saved to: {results_dir}")

    # Generate pruning configuration with fixed encoder/decoder layer handling
    recommendations_path = os.path.join(results_dir, "pruning_recommendations.json")
    if os.path.exists(recommendations_path):
        pruning_config_path = generate_pruning_config(
            recommendations_path=recommendations_path, model_name=args.model
        )
        print(f"Generated pruning configuration at: {pruning_config_path}")

    print("\nKey outputs:")
    print("- sensitivity_summary.png/pdf: Visual summary for research paper inclusion")
    print("- layer_by_layer_sensitivity.png/pdf: Detailed layer-by-layer analysis")
    print("- layer_type_by_position.png/pdf: Layer type analysis by position")
    print("- first_vs_last_layer.png/pdf: Detailed comparison of first vs last layers")
    print("- mlp_layer_sensitivity.png/pdf: MLP sensitivity analysis by position")
    print("- mlp_layer_by_layer.png/pdf: Detailed MLP layer-by-layer analysis")
    print("- pruning_recommendations.json: Structured pruning recommendations")
    print("- pruning_config.json: Ready-to-use pruning configuration for implementation")


if __name__ == "__main__":
    main()
