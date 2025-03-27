import gc
import json
import os
import re
from datetime import datetime

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


# Reuse your existing load_librispeech function
def load_librispeech(num_samples=None, split="test.clean"):
    """
    Load LibriSpeech clean/other data.
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

    print(f"Loaded {len(dataset)} test samples")
    print(f"Total audio duration: {total_hours:.4f} hours")
    return dataset


def clear_gpu_memory():
    """Clear cached GPU memory"""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()


def categorize_parameter(name):
    """Categorize parameter by location and type in Whisper architecture"""
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


def map_to_feats(batch, processor):
    """Process batch to extract features using the processor"""
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch


def create_summary_plot(sensitivity_results, output_dir, model_info=None):
    """
    Create a concise, publication-ready summary plot for a research paper.
    This generates a single figure with multiple panels showing the key insights
    from the sensitivity analysis.
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
    # This palette works well for both color and grayscale printing
    palette = sns.color_palette("colorblind")

    # Create custom colormap for heatmap
    blues_cmap = LinearSegmentedColormap.from_list("custom_blues", [(0.95, 0.95, 1), (0, 0, 0.7)])

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

            # Absolute position - use layer index directly
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

    # Initialize matrices for encoder and decoder
    pruning_matrix = np.zeros((len(positions), len(layer_categories)))

    # Helper function to calculate pruning rate
    def calc_pruning_rate(importance, max_imp=None):
        if max_imp is None or max_imp == 0:
            if importance == 0:
                return 0.5  # Default when no data available
            return 0.4  # Base pruning rate

        # Normalize and invert: higher importance = lower pruning rate
        # Scale to reasonable pruning range (20% to 80%)
        normalized = importance / max_imp if max_imp > 0 else 0
        return max(0.2, min(0.8, 0.8 - (normalized * 0.6)))

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

    # Compute position indices based on relative layer depth
    for component in ["encoder", "decoder"]:
        # Prefer decoder for recommendations (assuming it's less critical)
        if component != "decoder":
            continue

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
                        max_layer = max_dec_layer if component == "decoder" else max_enc_layer
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

                # Calculate pruning rate and store in matrix
                if layer_type in layer_type_map:
                    col_idx = layer_type_map[layer_type]
                    pruning_matrix[pos_idx, col_idx] = calc_pruning_rate(avg_imp, max_importance)

    # Special handling for cross-attention (only in decoder)
    for pos_idx in range(len(positions)):
        if pruning_matrix[pos_idx, 1] == 0:  # If cross-attention is empty
            # Find cross-attention values
            cross_attn_values = []
            for key, data in layer_importance.items():
                if data["component"] == "decoder" and data["layer_type"] == "cross_attention":
                    cross_attn_values.append(data["avg_importance"])

            avg_imp = np.mean(cross_attn_values) if cross_attn_values else 0
            pruning_matrix[pos_idx, 1] = calc_pruning_rate(avg_imp, max_importance)

    # Fill in any remaining zeros with default values
    for i in range(pruning_matrix.shape[0]):
        for j in range(pruning_matrix.shape[1]):
            if pruning_matrix[i, j] == 0:
                # Use a default value based on column (layer type)
                if j == 0:  # Self Attention
                    pruning_matrix[i, j] = 0.3  # More conservative
                elif j == 1:  # Cross Attention
                    pruning_matrix[i, j] = 0.4
                elif j == 2:  # Feed Forward
                    pruning_matrix[i, j] = 0.6  # More aggressive
                elif j == 3:  # Layer Norm
                    pruning_matrix[i, j] = 0.7  # Most aggressive

    # Plot heatmap
    im = ax4.imshow(pruning_matrix, cmap=blues_cmap, vmin=0.2, vmax=0.8)

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
    ax4.set_title("D) Recommended Pruning Rates", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, label="Pruning Rate")
    cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cbar.set_ticklabels(["20%", "30%", "40%", "50%", "60%", "70%", "80%"])

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
    plt.close()

    print(
        f"Created research paper summary plot: {os.path.join(output_dir, 'sensitivity_summary.png')}"
    )


def run_sensitivity_analysis(
    model_name, output_dir=None, num_samples=200, batch_size=1, num_batches=30, split="other"
):
    """
    Run a sensitivity analysis for Whisper model

    Args:
        model_name: The Whisper model name ("openai/whisper-small")
        output_dir: Output directory (defaults to timestamped directory)
        num_samples: Number of samples to use from LibriSpeech
        batch_size: Batch size for processing (use 1 to avoid collation issues)
        num_batches: Number of batches to use for sensitivity computation
        split: Dataset split to use ("test.clean" or "test.other")

    Returns:
        Path to results directory
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
    model_info = {
        "name": model_name,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": {
            "encoder": len(model.encoder.layers) if hasattr(model, "encoder") else 0,
            "decoder": len(model.decoder.layers) if hasattr(model, "decoder") else 0,
        },
        "hidden_size": model.config.d_model,
        "attention_heads": {
            "encoder": model.config.encoder_attention_heads
            if hasattr(model.config, "encoder_attention_heads")
            else 0,
            "decoder": model.config.decoder_attention_heads
            if hasattr(model.config, "decoder_attention_heads")
            else 0,
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

    # 3. Compute sensitivity metrics
    print("\nComputing parameter sensitivity...")
    sensitivity_results = compute_sensitivity(
        model=model,
        processor=processor,
        samples=processed_samples,
        device=device,
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

    # 5. Create the research summary plot (the only visualization we're keeping)
    print("\nCreating sensitivity summary visualization...")
    create_summary_plot(sensitivity_results, output_dir, model_info)

    return output_dir, sensitivity_results


def compute_sensitivity(model, processor, samples, device, num_batches=30, batch_size=1):
    """Compute sensitivity metrics for model parameters with improved gradient collection"""
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
    # This helps prevent all layers having identical importance
    while batch_count < num_batches and sample_idx < len(samples):
        # Process a single batch
        try:
            # Get batch of samples (simple approach to avoid collation issues)
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

                # Use forced decoding to get more diverse gradients
                outputs = model(
                    input_features=features,
                    decoder_input_ids=target_ids[:, :-1] if target_ids.size(1) > 1 else None,
                    labels=target_ids if target_ids.size(1) <= 1 else target_ids[:, 1:],
                )

                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Accumulate sensitivity metrics
                for name, param in model.named_parameters():
                    if name in sensitivity_results and param.grad is not None:
                        # Store gradient statistics
                        grad = param.grad.detach().cpu()
                        grad_flat = grad.flatten()

                        # Get gradient statistics - store mean, std, min, max
                        grad_mean = float(grad.mean().item())
                        grad_std = float(grad.std().item())
                        grad_min = float(grad.min().item())
                        grad_max = float(grad.max().item())

                        # Calculate Fisher (grad²) and SNIP (|param * grad|) metrics
                        fisher = float((grad**2).mean().item())
                        snip = float((param.detach() * grad).abs().mean().item())

                        # Store in results
                        sensitivity_results[name]["gradients"].append(
                            {
                                "mean": grad_mean,
                                "std": grad_std,
                                "min": grad_min,
                                "max": grad_max,
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

    for name, data in tqdm(sensitivity_results.items(), desc="Processing parameters"):
        if data["batch_count"] > 0:
            # Calculate metrics from gradient statistics
            fisher_values = [g["fisher"] for g in data["gradients"]]
            snip_values = [g["snip"] for g in data["gradients"]]

            fisher_mean = np.mean(fisher_values) if fisher_values else 0
            snip_mean = np.mean(snip_values) if snip_values else 0

            # Combined importance score from both metrics (Fisher and SNIP)
            importance = (fisher_mean + snip_mean) / 2

            # Store importance statistics
            parameter_importance[name] = {
                "category": data["category"],
                "shape": data["shape"],
                "size": data["size"],
                "param_sample": data["param_values"],
                "importance": importance,
                "fisher": fisher_mean,
                "snip": snip_mean,
                "gradient_count": len(data["gradients"]),
            }

    # Calculate global statistics for normalization
    all_importances = [data["importance"] for name, data in parameter_importance.items()]
    if all_importances:
        max_importance = max(all_importances)
        min_importance = min(all_importances)
        mean_importance = np.mean(all_importances)
        std_importance = np.std(all_importances)

        # Make sure we have reasonable variation in importance
        if max_importance > min_importance and std_importance > 0:
            # Normalize importance scores to make differences more visible
            importance_range = max_importance - min_importance
            for name in parameter_importance:
                raw_importance = parameter_importance[name]["importance"]
                normalized_importance = (raw_importance - min_importance) / importance_range
                parameter_importance[name]["importance_raw"] = raw_importance
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


if __name__ == "__main__":
    # Run sensitivity analysis on Whisper small with more batches for better statistics
    results_dir, _ = run_sensitivity_analysis(
        model_name="openai/whisper-small",
        num_samples=200,  # Use 200 samples for better statistics
        batch_size=1,  # Process one sample at a time to avoid collation issues
        num_batches=30,  # More batches for better statistics
        split="test.other",  # Use the "other" split for more challenging data
    )
    print(f"Sensitivity analysis complete. Results saved to: {results_dir}")
    print("Key output: sensitivity_summary.png/pdf - Ideal figure for research paper inclusion.")
