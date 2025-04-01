import gc
import io
import json
import os
import time
from collections import deque

import datasets
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn.utils.prune as prune
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Set seaborn style
sns.set(style="whitegrid")

# Create results directory
RESULTS_DIR = "pruning/whisper_pruning_results"
SELECTIVE_PRUNING_DIR = os.path.join(RESULTS_DIR, "selective_pruning")
PLOTS_DIR = os.path.join(SELECTIVE_PRUNING_DIR, "plots")
MODELS_DIR = os.path.join(SELECTIVE_PRUNING_DIR, "models")

for directory in [RESULTS_DIR, SELECTIVE_PRUNING_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


def calculate_pruned_dense_size(model, pruning_threshold=0.0):
    """
    Calculate the theoretical size of a dense model with pruned weights removed.
    This doesn't actually create the model, just calculates what the size would be.

    Args:
        model: The pruned model with masked weights
        pruning_threshold: Weights with absolute value below this threshold are considered pruned

    Returns:
        float: Size in MB that a dense model with pruned weights removed would have
    """
    print("\n=== Calculating theoretical dense model size with pruned weights removed ===")

    total_params_original = 0
    total_params_pruned = 0
    total_bytes_original = 0
    total_bytes_dense_pruned = 0

    # Count parameters and calculate theoretical size
    for name, param in model.named_parameters():
        param_size_bytes = param.numel() * 4  # 4 bytes per float32
        total_params_original += param.numel()
        total_bytes_original += param_size_bytes

        if "weight" in name and param.dim() > 1:  # Only consider weight matrices
            # Find pruned weights
            pruned_mask = torch.abs(param) <= pruning_threshold
            pruned_percentage = 100.0 * torch.sum(pruned_mask).item() / param.numel()

            # Track non-zero parameters
            non_zero_params = param.numel() - torch.sum(pruned_mask).item()
            total_params_pruned += non_zero_params

            # Calculate dense size without zeros
            param_dense_pruned_bytes = non_zero_params * 4  # Only non-zero elements at 4 bytes each
            total_bytes_dense_pruned += param_dense_pruned_bytes

            # For significant pruning, log details
            if pruned_percentage > 5 and param.numel() > 10000:
                print(f"Layer {name}: {pruned_percentage:.1f}% pruned")
                print(f"  Original: {param_size_bytes/1024/1024:.2f} MB")
                print(f"  Dense pruned: {param_dense_pruned_bytes/1024/1024:.2f} MB")
        else:
            # For non-weight parameters, size remains the same
            total_params_pruned += param.numel()
            total_bytes_dense_pruned += param_size_bytes

    # Convert to MB
    original_size_mb = total_bytes_original / (1024 * 1024)
    dense_pruned_size_mb = total_bytes_dense_pruned / (1024 * 1024)

    # Report on size reduction
    if total_params_original > 0:
        param_reduction = (
            100.0 * (total_params_original - total_params_pruned) / total_params_original
        )
        size_reduction = 100.0 * (original_size_mb - dense_pruned_size_mb) / original_size_mb

        print(f"Original parameters: {total_params_original:,}")
        print(f"Non-zero parameters: {total_params_pruned:,}")
        print(f"Parameter reduction: {param_reduction:.1f}%")
        print(f"Original size: {original_size_mb:.2f} MB")
        print(f"Theoretical dense pruned size: {dense_pruned_size_mb:.2f} MB")
        print(f"Size reduction: {size_reduction:.1f}%")

    return dense_pruned_size_mb


def calculate_model_gflops(model):
    """
    Calculate approximate GFLOPs for Whisper model accounting for pruning.

    Args:
        model: The WhisperForConditionalGeneration model

    Returns:
        float: Estimated GFLOPs
    """
    # Track FLOPs by module type
    flops_by_type = {
        "encoder": 0,
        "decoder": 0,
        "other": 0,
        "attention": 0,  # For attention specific tracking
        "feed_forward": 0,  # For feed-forward specific tracking
        "early_layers": 0,  # For layer-specific tracking
        "mid_layers": 0,
        "late_layers": 0,
    }

    total_params = 0
    non_zero_params = 0

    # Analyze linear layers (where most computation happens)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not hasattr(module, "weight"):
                continue

            # Get layer dimensions
            in_features = module.in_features
            out_features = module.out_features

            # Calculate theoretical FLOPs for this layer (multiply-add operations)
            # Each output element requires in_features multiplications and in_features-1 additions
            weight = module.weight

            # Calculate sparsity and non-zero operations
            weight_sparsity = (
                torch.sum(weight == 0).item() / weight.numel() if weight.numel() > 0 else 0
            )
            non_zero_ops = 2 * in_features * out_features * (1 - weight_sparsity)

            # Categorize by location in model
            if "encoder" in name:
                flops_by_type["encoder"] += non_zero_ops

                # Check if attention or feed-forward
                if any(
                    att_part in name
                    for att_part in ["attention", "attn", "k_proj", "q_proj", "v_proj", "o_proj"]
                ):
                    flops_by_type["attention"] += non_zero_ops
                elif any(ff_part in name for ff_part in ["feed_forward", "fc", "mlp"]):
                    flops_by_type["feed_forward"] += non_zero_ops

                # Track layer depth
                if "layers." in name:
                    try:
                        layer_str = name.split("layers.")[1].split(".")[0]
                        layer_num = int(layer_str)
                        if layer_num < 3:
                            flops_by_type["early_layers"] += non_zero_ops
                        elif layer_num < 6:
                            flops_by_type["mid_layers"] += non_zero_ops
                        else:
                            flops_by_type["late_layers"] += non_zero_ops
                    except (ValueError, IndexError):
                        pass

            elif "decoder" in name:
                flops_by_type["decoder"] += non_zero_ops

                # Check if attention or feed-forward
                if any(
                    att_part in name
                    for att_part in ["attention", "attn", "k_proj", "q_proj", "v_proj", "o_proj"]
                ):
                    flops_by_type["attention"] += non_zero_ops
                elif any(ff_part in name for ff_part in ["feed_forward", "fc", "mlp"]):
                    flops_by_type["feed_forward"] += non_zero_ops

                # Track layer depth
                if "layers." in name:
                    try:
                        layer_str = name.split("layers.")[1].split(".")[0]
                        layer_num = int(layer_str)
                        if layer_num < 3:
                            flops_by_type["early_layers"] += non_zero_ops
                        elif layer_num < 6:
                            flops_by_type["mid_layers"] += non_zero_ops
                        else:
                            flops_by_type["late_layers"] += non_zero_ops
                    except (ValueError, IndexError):
                        pass
            else:
                flops_by_type["other"] += non_zero_ops

            # Track parameter stats
            total_params += weight.numel()
            non_zero_params += (weight != 0).sum().item()

    # For a typical forward pass and generation in Whisper:
    # 1. Encoder processes the input once
    # 2. Decoder runs multiple times (typically sequence length)
    # Simplified assumption: avg sequence length of 25 tokens
    avg_sequence_length = 25
    total_flops = (
        flops_by_type["encoder"]
        + avg_sequence_length * flops_by_type["decoder"]
        + flops_by_type["other"]
    )

    # Convert to GFLOPs
    total_gflops = total_flops / 1e9

    # Print detailed breakdown
    print("\nEstimated GFLOPs by component:")
    for component in ["encoder", "decoder", "other"]:
        gflops = flops_by_type[component] / 1e9
        percentage = (
            flops_by_type[component]
            / (flops_by_type["encoder"] + flops_by_type["decoder"] + flops_by_type["other"])
        ) * 100
        print(f"  {component}: {gflops:.4f} GFLOPs ({percentage:.1f}%)")

    print("\nEstimated GFLOPs by neural network component:")
    attention_gflops = flops_by_type["attention"] / 1e9
    feedforward_gflops = flops_by_type["feed_forward"] / 1e9
    att_ff_total = flops_by_type["attention"] + flops_by_type["feed_forward"]
    if att_ff_total > 0:
        print(
            f"  attention: {attention_gflops:.4f} GFLOPs ({100 * flops_by_type['attention'] / att_ff_total:.1f}%)"
        )
        print(
            f"  feed_forward: {feedforward_gflops:.4f} GFLOPs ({100 * flops_by_type['feed_forward'] / att_ff_total:.1f}%)"
        )

    print("\nEstimated GFLOPs by layer depth:")
    layer_total = (
        flops_by_type["early_layers"] + flops_by_type["mid_layers"] + flops_by_type["late_layers"]
    )
    if layer_total > 0:
        print(
            f"  early_layers: {flops_by_type['early_layers']/1e9:.4f} GFLOPs ({100 * flops_by_type['early_layers'] / layer_total:.1f}%)"
        )
        print(
            f"  mid_layers: {flops_by_type['mid_layers']/1e9:.4f} GFLOPs ({100 * flops_by_type['mid_layers'] / layer_total:.1f}%)"
        )
        print(
            f"  late_layers: {flops_by_type['late_layers']/1e9:.4f} GFLOPs ({100 * flops_by_type['late_layers'] / layer_total:.1f}%)"
        )

    if total_params > 0:
        print("\nParameter efficiency:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {non_zero_params:,}")
        print(f"  Overall sparsity: {100 * (1 - non_zero_params / total_params):.2f}%")

    print(f"\nTotal estimated GFLOPs: {total_gflops:.4f}")

    return total_gflops


class WhisperMemoryTracker:
    def __init__(self, model_name: str, save_path: str):
        self.model_name = model_name
        self.save_path = save_path
        self.peak_gpu_memory = 0
        self.peak_cpu_percent = 0
        self.peak_ram_gb = 0
        self.memory_measurements = deque(maxlen=10)  # Reduced size, only for summary
        self.start_time = time.time()
        self.process = psutil.Process()
        self.device_type = "cpu"

        # Initialize GPU memory attributes even if running on CPU
        self.initial_gpu_memory = 0
        self.initial_gpu_cached = 0

        self.process.cpu_percent(interval=None)  # First call returns 0, discard it
        self.initial_cpu_percent = np.mean(
            [self.process.cpu_percent(interval=0.1) for _ in range(5)]
        )  # Stable avg
        self.initial_ram_usage = self.process.memory_info().rss / (1024**3)
        self.peak_ram_gb = self.initial_ram_usage

        # Initialize GPU memory metrics if available
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            self.initial_gpu_cached = torch.cuda.memory_reserved() / (1024**3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_type = "mps"
            # MPS doesn't have easy memory tracking like CUDA

    def log_memory(self, split, batch_idx, batch_size, audio_duration):
        current_time = time.time()
        cpu_percent = np.mean(
            [self.process.cpu_percent(interval=0.1) for _ in range(3)]
        )  # Avg over 3 readings
        current_ram = self.process.memory_info().rss / (1024**3)
        self.peak_ram_gb = max(self.peak_ram_gb, current_ram)

        memory_data = {
            "timestamp": float(current_time - self.start_time),  # Ensure it's a native float
            "cpu_percent": float(cpu_percent),  # Ensure it's a native float
            "ram_gb": float(current_ram),  # Ensure it's a native float
            "batch_info": {
                "split": split,
                "batch_idx": int(batch_idx),  # Ensure it's a native int
                "batch_size": int(batch_size),  # Ensure it's a native int
                "audio_duration": float(audio_duration),  # Ensure it's a native float
            },
        }

        if torch.cuda.is_available():
            gpu_allocated = float(torch.cuda.memory_allocated() / (1024**3))
            gpu_cached = float(torch.cuda.memory_reserved() / (1024**3))
            gpu_peak = float(torch.cuda.max_memory_allocated() / (1024**3))

            memory_data.update(
                {
                    "gpu_allocated_gb": gpu_allocated,
                    "gpu_cached_gb": gpu_cached,
                    "gpu_peak_gb": gpu_peak,
                }
            )
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_peak)
        elif self.device_type == "mps":
            # MPS does not have native memory tracking, but we'll include device info
            memory_data.update({"device_type": "mps"})

        # Append the memory measurement and explicitly make it a dict
        self.memory_measurements.append(dict(memory_data))
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)

    def get_memory_summary(self):
        """Get comprehensive memory usage statistics."""
        if not self.memory_measurements:
            return "No measurements recorded"

        # Calculate average RAM and CPU usage
        ram_measurements = [m.get("ram_gb", 0) for m in self.memory_measurements]
        avg_ram_usage = sum(ram_measurements) / len(ram_measurements) if ram_measurements else 0

        cpu_measurements = [m.get("cpu_percent", 0) for m in self.memory_measurements]
        avg_cpu_usage = sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0

        summary = {
            "duration_seconds": time.time() - self.start_time,
            "cpu": {
                "initial_percent": self.initial_cpu_percent,
                "peak_percent": self.peak_cpu_percent,
                "average_percent": avg_cpu_usage,
                "initial_ram_gb": self.initial_ram_usage,
                "peak_ram_gb": self.peak_ram_gb,
                "average_ram_gb": avg_ram_usage,
                "current_ram_gb": self.process.memory_info().rss / (1024**3),
            },
        }

        if torch.cuda.is_available():
            gpu_measurements = [
                m.get("gpu_allocated_gb", 0)
                for m in self.memory_measurements
                if "gpu_allocated_gb" in m
            ]
            if gpu_measurements:
                summary["gpu"] = {
                    "initial_allocated_gb": self.initial_gpu_memory,
                    "initial_cached_gb": self.initial_gpu_cached,
                    "peak_allocated_gb": self.peak_gpu_memory,
                    "average_allocated_gb": sum(gpu_measurements) / len(gpu_measurements),
                    "current_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "current_cached_gb": torch.cuda.memory_reserved() / (1024**3),
                }
        elif self.device_type == "mps":
            summary["mps"] = {
                "device_type": "mps",
                "note": "MPS memory tracking not available like CUDA",
            }

        return summary

    def save_metrics(self):
        """Save only summary memory metrics to a JSON file."""
        metrics_path = os.path.join(self.save_path, f"{self.model_name}_memory_summary.json")
        summary = self.get_memory_summary()

        # Create the output dictionary with only summary data
        output_data = {"summary": summary}

        try:
            with open(metrics_path, "w") as f:
                json.dump(output_data, f, indent=2)
        except TypeError as e:
            # If we still have serialization issues, let's create a simpler output
            print(f"Warning: JSON serialization error: {e}")
            simplified_output = {
                "summary": {
                    "duration_seconds": summary["duration_seconds"]
                    if isinstance(summary, dict)
                    else 0,
                    "cpu": {
                        "peak_percent": self.peak_cpu_percent,
                        "current_ram_gb": self.process.memory_info().rss / (1024**3),
                    },
                },
            }
            with open(metrics_path, "w") as f:
                json.dump(simplified_output, f, indent=2)

    def print_summary(self):
        """Print detailed memory usage summary."""
        summary = self.get_memory_summary()

        print(f"\n=== Memory Usage Summary for {self.model_name} ===")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print("\nCPU Usage:")
        print(f"  Initial CPU: {summary['cpu']['initial_percent']:.3f}%")
        print(f"  Peak CPU: {summary['cpu']['peak_percent']:.3f}%")
        print(f"  Average CPU: {summary['cpu']['average_percent']:.3f}%")
        print(f"  Initial RAM: {summary['cpu']['initial_ram_gb']:.4f} GB")
        print(f"  Peak RAM: {summary['cpu']['peak_ram_gb']:.4f} GB")
        print(f"  Average RAM: {summary['cpu']['average_ram_gb']:.4f} GB")
        print(f"  Current RAM: {summary['cpu']['current_ram_gb']:.4f} GB")

        if "gpu" in summary:
            print("\nGPU Usage:")
            print(f"  Initial Allocated: {summary['gpu']['initial_allocated_gb']:.4f} GB")
            print(f"  Peak Allocated: {summary['gpu']['peak_allocated_gb']:.4f} GB")
            print(f"  Average Allocated: {summary['gpu']['average_allocated_gb']:.4f} GB")
            print(f"  Current Allocated: {summary['gpu']['current_allocated_gb']:.4f} GB")
            print(f"  Current Cached: {summary['gpu']['current_cached_gb']:.4f} GB")
        elif "mps" in summary:
            print("\nGPU Usage (MPS):")
            print("  Note: Detailed MPS memory metrics not available")
            print("  Using system RAM metrics as proxy for memory usage")

    def close(self):
        """Cleanup and save final metrics."""
        self.print_summary()
        self.save_metrics()


def apply_selective_pruning(
    model, method="l1_unstructured", amount=0.5, target_submodules=None, make_permanent=False
):
    """
    Apply pruning to specific parts of a Whisper model.

    Args:
        model: The WhisperForConditionalGeneration model
        method: Pruning method ('l1_unstructured', 'random_unstructured', etc.)
        amount: Amount of weights to prune (0.5 = 50%)
        target_submodules: List of submodule name parts to target (e.g., ["encoder"])
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
    """
    # Default to all Linear layers
    target_modules = [torch.nn.Linear]

    if target_submodules is None:
        print("Warning: No target submodules specified. Will prune the entire model.")
        target_submodules = []  # Empty list means no filtering

    # Get parameters to prune based on target modules and submodules
    params_to_prune = []
    encoder_params = []
    decoder_params = []
    other_params = []

    for name, module in model.named_modules():
        # Check if module is of target type
        if any(isinstance(module, m) for m in target_modules):
            # Check if it belongs to target submodule (if specified)
            if not target_submodules or any(submodule in name for submodule in target_submodules):
                params_to_prune.append((module, "weight"))

                # Also categorize for reporting
                if "encoder" in name:
                    encoder_params.append((module, "weight"))
                elif "decoder" in name:
                    decoder_params.append((module, "weight"))
                else:
                    other_params.append((module, "weight"))

    if not params_to_prune:
        print(f"Warning: No parameters found to prune with filter {target_submodules}!")
        return model

    print(f"Found {len(params_to_prune)} modules to prune:")
    print(f"  - Encoder modules: {len(encoder_params)}")
    print(f"  - Decoder modules: {len(decoder_params)}")
    print(f"  - Other modules: {len(other_params)}")

    # Apply the specified pruning method
    if method == "l1_unstructured":
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=amount
        )
    elif method == "random_unstructured":
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.RandomUnstructured, amount=amount
        )
    elif method == "ln_structured":
        # For structured pruning along specific dimensions
        for module, param_name in params_to_prune:
            try:
                # Try dim=0 (output features)
                prune.ln_structured(module, param_name, amount=amount, n=2, dim=0)
            except Exception as e:
                print(f"Error applying structured pruning to {module}: {e}")
                print("Falling back to unstructured pruning for this module")
                prune.l1_unstructured(module, param_name, amount=amount)

    print(f"Applied {method} pruning with amount {amount} to {len(params_to_prune)} modules")

    # Make pruning permanent if requested
    if make_permanent:
        print("Making pruning permanent...")
        for module, param_name in params_to_prune:
            try:
                prune.remove(module, param_name)
            except Exception as e:
                print(f"Could not make pruning permanent for {module}: {e}")

    return model


def apply_attention_head_pruning(model, amount=0.5):
    """
    Prune entire attention heads based on their L1 norm.

    Args:
        model: The WhisperForConditionalGeneration model
        amount: Amount of heads to prune (0.5 = 50%)

    Returns:
        Pruned model
    """
    # Identify attention head projection layers
    head_layers = []

    # Track number of heads in each attention module
    attention_modules = {}

    for name, module in model.named_modules():
        # Look for attention projection layers
        if any(suffix in name for suffix in ["q_proj", "k_proj", "v_proj"]):
            if hasattr(module, "weight"):
                # Extract attention module name
                if "encoder" in name:
                    parts = name.split("encoder")[1].split(".")
                    att_module = "encoder" + ".".join(parts[:-1])
                    if att_module not in attention_modules:
                        # Try to determine number of heads
                        if hasattr(model.encoder, "layers"):
                            layer_idx = int(parts[1])
                            if hasattr(model.encoder.layers[layer_idx].self_attn, "num_heads"):
                                attention_modules[att_module] = model.encoder.layers[
                                    layer_idx
                                ].self_attn.num_heads
                            else:
                                attention_modules[att_module] = 8  # Default for Whisper small
                elif "decoder" in name:
                    parts = name.split("decoder")[1].split(".")
                    att_module = "decoder" + ".".join(parts[:-1])
                    if att_module not in attention_modules:
                        # Try to determine number of heads
                        if hasattr(model.decoder, "layers"):
                            layer_idx = int(parts[1])
                            if hasattr(model.decoder.layers[layer_idx].self_attn, "num_heads"):
                                attention_modules[att_module] = model.decoder.layers[
                                    layer_idx
                                ].self_attn.num_heads
                            else:
                                attention_modules[att_module] = 8  # Default for Whisper small

                head_layers.append((name, module))

    print(
        f"Found {len(head_layers)} attention projection layers in {len(attention_modules)} attention modules"
    )

    # For each attention module, calculate head importance and prune the least important heads
    heads_pruned = 0
    total_heads = 0

    for att_module, num_heads in attention_modules.items():
        print(f"Processing {att_module} with {num_heads} heads")
        total_heads += num_heads

        # Get all projection layers for this attention module
        module_layers = [(name, mod) for name, mod in head_layers if att_module in name]

        if len(module_layers) < 3:  # Should have q, k, v projections
            print(f"  Missing projection layers for {att_module}, skipping")
            continue

        # Calculate head importance
        head_importance = torch.zeros(num_heads).to(next(model.parameters()).device)

        for name, module in module_layers:
            weight = module.weight

            # Reshape to get per-head weights
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                # Extract head dimension (assuming weight shape is [out_dim, in_dim] with out_dim = num_heads * head_dim)
                out_features = weight.size(0)
                head_dim = out_features // num_heads

                # Reshape to [num_heads, head_dim, in_dim]
                reshaped = weight.view(num_heads, head_dim, -1)

                # Calculate L1 norm for each head
                head_l1 = torch.sum(torch.abs(reshaped), dim=(1, 2))

                # Add to importance scores
                head_importance += head_l1

        # Determine number of heads to prune
        num_to_prune = int(num_heads * amount)
        if num_to_prune <= 0:
            print(f"  No heads to prune for {att_module}")
            continue

        # Get indices of least important heads
        _, indices = torch.topk(head_importance, k=num_heads - num_to_prune, largest=True)
        heads_to_keep = set(indices.cpu().numpy())

        print(f"  Pruning {num_to_prune} heads out of {num_heads}")

        # Prune heads by zeroing out their weights
        for name, module in module_layers:
            weight = module.weight
            out_features = weight.size(0)
            head_dim = out_features // num_heads

            # Create pruning mask
            mask = torch.ones_like(weight)

            # Zero out weights for pruned heads
            for h in range(num_heads):
                if h not in heads_to_keep:
                    # Calculate start and end indices for this head
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    mask[start_idx:end_idx, :] = 0

            # Apply mask
            with torch.no_grad():
                module.weight.mul_(mask)

        heads_pruned += num_to_prune

    print(
        f"Pruned {heads_pruned} attention heads out of {total_heads} total heads ({100.0 * heads_pruned / total_heads:.1f}%)"
    )
    return model


def apply_layer_dropping(model, layers_to_drop):
    """
    Apply layer dropping to a model by zeroing out entire transformer layers.

    Args:
        model: The WhisperForConditionalGeneration model
        layers_to_drop: Dictionary specifying which layers to drop, e.g.,
                       {'encoder': [0, 2], 'decoder': [1, 3]}

    Returns:
        Model with dropped layers
    """
    print("Applying layer dropping...")

    total_layers = 0
    dropped_layers = 0

    # Process encoder layers
    if (
        "encoder" in layers_to_drop
        and hasattr(model, "encoder")
        and hasattr(model.encoder, "layers")
    ):
        encoder_layers = len(model.encoder.layers)
        total_layers += encoder_layers

        encoder_to_drop = [i for i in layers_to_drop["encoder"] if i < encoder_layers]
        dropped_layers += len(encoder_to_drop)

        print(
            f"Dropping {len(encoder_to_drop)} out of {encoder_layers} encoder layers: {encoder_to_drop}"
        )

        # Zero out weights in the layers to be dropped
        for layer_idx in encoder_to_drop:
            for name, param in model.encoder.layers[layer_idx].named_parameters():
                param.data.zero_()

    # Process decoder layers
    if (
        "decoder" in layers_to_drop
        and hasattr(model, "decoder")
        and hasattr(model.decoder, "layers")
    ):
        decoder_layers = len(model.decoder.layers)
        total_layers += decoder_layers

        decoder_to_drop = [i for i in layers_to_drop["decoder"] if i < decoder_layers]
        dropped_layers += len(decoder_to_drop)

        print(
            f"Dropping {len(decoder_to_drop)} out of {decoder_layers} decoder layers: {decoder_to_drop}"
        )

        # Zero out weights in the layers to be dropped
        for layer_idx in decoder_to_drop:
            for name, param in model.decoder.layers[layer_idx].named_parameters():
                param.data.zero_()

    print(
        f"Dropped {dropped_layers} out of {total_layers} total layers ({100.0 * dropped_layers / total_layers:.1f}%)"
    )
    return model


def prune_attention_vs_feedforward(model, method, config):
    """
    Prune attention and feed-forward networks with different sparsity levels.

    Args:
        model: The WhisperForConditionalGeneration model
        method: Pruning method ('l1_unstructured', 'random_unstructured', etc.)
        config: Dictionary with keys 'attention_amount' and 'ffn_amount'

    Returns:
        Pruned model
    """
    attention_amount = config.get("attention_amount", 0.3)
    ffn_amount = config.get("ffn_amount", 0.7)

    print(f"Pruning attention components with amount={attention_amount}")
    print(f"Pruning feed-forward components with amount={ffn_amount}")

    # Identify attention and feed-forward modules
    attention_params = []
    ffn_params = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if it's an attention module
            if any(
                att_part in name
                for att_part in ["attention", "attn", "k_proj", "q_proj", "v_proj", "o_proj"]
            ):
                attention_params.append((module, "weight"))
            # Check if it's a feed-forward module
            elif any(ff_part in name for ff_part in ["feed_forward", "fc", "mlp"]):
                ffn_params.append((module, "weight"))

    print(
        f"Found {len(attention_params)} attention modules and {len(ffn_params)} feed-forward modules"
    )

    # Apply pruning separately to attention and feed-forward networks
    if method == "l1_unstructured":
        if attention_params:
            prune.global_unstructured(
                attention_params, pruning_method=prune.L1Unstructured, amount=attention_amount
            )
        if ffn_params:
            prune.global_unstructured(
                ffn_params, pruning_method=prune.L1Unstructured, amount=ffn_amount
            )
    elif method == "random_unstructured":
        if attention_params:
            prune.global_unstructured(
                attention_params, pruning_method=prune.RandomUnstructured, amount=attention_amount
            )
        if ffn_params:
            prune.global_unstructured(
                ffn_params, pruning_method=prune.RandomUnstructured, amount=ffn_amount
            )

    # Make pruning permanent
    for module, param_name in attention_params + ffn_params:
        try:
            prune.remove(module, param_name)
        except Exception:
            pass

    return model


def calculate_sparsity(model, submodule_filter=None):
    """
    Calculate the sparsity percentage and parameter counts in the model or specific submodules.

    Args:
        model: The PyTorch model
        submodule_filter: Optional list of submodule name parts to filter

    Returns:
        tuple: (sparsity percentage, total parameters, non-zero parameters)
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:  # Only consider weight parameters
            # Check if it belongs to target submodule (if filter is specified)
            if submodule_filter is None or any(submodule in name for submodule in submodule_filter):
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()

    if total_params == 0:
        return 0.0, 0, 0

    sparsity = 100.0 * zero_params / total_params
    non_zero_params = total_params - zero_params
    return sparsity, total_params, non_zero_params


def calculate_component_sparsity(model):
    """Calculate sparsity separately for encoder and decoder components."""
    encoder_sparsity, encoder_total, encoder_nonzero = calculate_sparsity(model, ["encoder"])
    decoder_sparsity, decoder_total, decoder_nonzero = calculate_sparsity(model, ["decoder"])
    overall_sparsity, total_params, nonzero_params = calculate_sparsity(model)

    # Additional metrics for attention vs. feed-forward
    attention_sparsity, attention_total, attention_nonzero = calculate_sparsity(
        model, ["attention", "attn", "k_proj", "q_proj", "v_proj", "o_proj"]
    )
    ffn_sparsity, ffn_total, ffn_nonzero = calculate_sparsity(model, ["feed_forward", "fc", "mlp"])

    # Calculate early, mid, and late layer sparsity
    early_layers_sparsity, early_total, early_nonzero = 0, 0, 0
    mid_layers_sparsity, mid_total, mid_nonzero = 0, 0, 0
    late_layers_sparsity, late_total, late_nonzero = 0, 0, 0

    # Helper function to extract layer index
    def get_layer_index(name):
        if "layers." in name:
            try:
                layer_str = name.split("layers.")[1].split(".")[0]
                return int(layer_str)
            except (ValueError, IndexError):
                return -1
        return -1

    # Count parameters by layer depth
    early_total = 0
    early_zeros = 0
    mid_total = 0
    mid_zeros = 0
    late_total = 0
    late_zeros = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            layer_idx = get_layer_index(name)
            if layer_idx >= 0:
                if layer_idx < 3:
                    early_total += param.numel()
                    early_zeros += torch.sum(param == 0).item()
                elif layer_idx < 6:
                    mid_total += param.numel()
                    mid_zeros += torch.sum(param == 0).item()
                else:
                    late_total += param.numel()
                    late_zeros += torch.sum(param == 0).item()

    if early_total > 0:
        early_layers_sparsity = 100.0 * early_zeros / early_total
        early_nonzero = early_total - early_zeros
    if mid_total > 0:
        mid_layers_sparsity = 100.0 * mid_zeros / mid_total
        mid_nonzero = mid_total - mid_zeros
    if late_total > 0:
        late_layers_sparsity = 100.0 * late_zeros / late_total
        late_nonzero = late_total - late_zeros

    return {
        "encoder_sparsity": encoder_sparsity,
        "decoder_sparsity": decoder_sparsity,
        "overall_sparsity": overall_sparsity,
        "attention_sparsity": attention_sparsity,
        "ffn_sparsity": ffn_sparsity,
        "early_layers_sparsity": early_layers_sparsity,
        "mid_layers_sparsity": mid_layers_sparsity,
        "late_layers_sparsity": late_layers_sparsity,
        "total_parameters": total_params,
        "non_zero_parameters": nonzero_params,
        "encoder_total": encoder_total,
        "encoder_nonzero": encoder_nonzero,
        "decoder_total": decoder_total,
        "decoder_nonzero": decoder_nonzero,
        "attention_total": attention_total,
        "attention_nonzero": attention_nonzero,
        "ffn_total": ffn_total,
        "ffn_nonzero": ffn_nonzero,
        "early_total": early_total,
        "early_nonzero": early_nonzero,
        "mid_total": mid_total,
        "mid_nonzero": mid_nonzero,
        "late_total": late_total,
        "late_nonzero": late_nonzero,
    }


def load_whisper_model(model_name, device, pruning_config=None):
    """
    Load Whisper model and optionally apply pruning.

    Args:
        model_name: The Whisper model name
        device: Device to load the model to
        pruning_config: Dictionary with pruning configuration

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        # Load model without device_map
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_config:
            print(f"Applying pruning with config: {pruning_config}")

            method = pruning_config.get("method", "l1_unstructured")

            # Handle special pruning methods
            if method == "attention_head_pruning":
                amount = pruning_config.get("amount", 0.5)
                model = apply_attention_head_pruning(model, amount=amount)
            elif method == "layer_dropping":
                layers_to_drop = pruning_config.get(
                    "layers_to_drop", {"encoder": [], "decoder": []}
                )
                model = apply_layer_dropping(model, layers_to_drop)
            elif method == "attention_vs_ffn":
                config = {
                    "attention_amount": pruning_config.get("attention_amount", 0.3),
                    "ffn_amount": pruning_config.get("ffn_amount", 0.7),
                }
                pruning_method = pruning_config.get("pruning_method", "l1_unstructured")
                model = prune_attention_vs_feedforward(model, pruning_method, config)
            else:
                # Standard selective pruning
                amount = pruning_config.get("amount", 0.5)
                target_submodules = pruning_config.get("target_submodules", None)
                make_permanent = pruning_config.get("make_permanent", True)

                model = apply_selective_pruning(
                    model,
                    method=method,
                    amount=amount,
                    target_submodules=target_submodules,
                    make_permanent=make_permanent,
                )

            # Calculate and print sparsity by component
            sparsity_info = calculate_component_sparsity(model)
            print("Sparsity by component:")
            print(f"  - Encoder: {sparsity_info['encoder_sparsity']:.2f}%")
            print(f"  - Decoder: {sparsity_info['decoder_sparsity']:.2f}%")
            print(f"  - Attention: {sparsity_info['attention_sparsity']:.2f}%")
            print(f"  - Feed-Forward: {sparsity_info['ffn_sparsity']:.2f}%")
            print(f"  - Early layers: {sparsity_info['early_layers_sparsity']:.2f}%")
            print(f"  - Mid layers: {sparsity_info['mid_layers_sparsity']:.2f}%")
            print(f"  - Late layers: {sparsity_info['late_layers_sparsity']:.2f}%")
            print(f"  - Overall: {sparsity_info['overall_sparsity']:.2f}%")
            print(f"  - Total Parameters: {sparsity_info['total_parameters']:,}")
            print(f"  - Non-zero Parameters: {sparsity_info['non_zero_parameters']:,}")

        # Move model to device
        model = model.to(device)
        model.config.forced_decoder_ids = None
        return model

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def clear_gpu_memory():
    """Clear cached GPU memory and reset peak memory stats if CUDA or MPS is available."""
    # Force garbage collection first
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't have explicit memory management functions like CUDA,
        # but we can still force garbage collection
        print("Clearing memory on MPS device")
        # Try to force device synchronization if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # Force garbage collection again after clearing cache
    gc.collect()


def map_to_feats(batch, processor):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch


def transcribe_batch(batch, model, processor, memory_tracker, split, batch_idx):
    with torch.no_grad():
        # Prepare input features
        features = torch.from_numpy(np.array(batch["input_features"], dtype=np.float32)).squeeze(1)
        if next(model.parameters()).dtype == torch.float16:
            features = features.half()
        features = features.to(model.device)

        # Compute total audio duration for the batch
        audio_durations = [len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]]
        total_audio_duration = sum(audio_durations)

        # Measure processing time with GPU synchronization
        start_time = time.time()

        # Generate with aggressive memory management
        try:
            predicted_ids = model.generate(features)

            # Synchronize based on device type
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all CUDA ops complete
            elif hasattr(torch.mps, "synchronize") and model.device.type == "mps":
                torch.mps.synchronize()  # Ensure all MPS ops complete

            # Explicitly delete features tensor to free memory immediately
            del features

            # Force garbage collection
            gc.collect()

        except Exception as e:
            print(f"Error during generation: {e}")
            # Clean up in case of error
            try:
                del features
            except NameError:
                pass  # features wasn't defined yet
            gc.collect()
            raise

        processing_time = time.time() - start_time

        # Calculate batch RTF
        batch_rtf = processing_time / total_audio_duration

        # Log memory usage for this batch
        memory_tracker.log_memory(
            split=split,
            batch_idx=batch_idx,
            batch_size=len(batch["audio"]),
            audio_duration=total_audio_duration,
        )

    # Decode predictions
    transcription = [processor.decode(ids) for ids in predicted_ids]
    batch["prediction"] = [processor.tokenizer.normalize(x) for x in transcription]

    # Delete predicted_ids to free memory
    del predicted_ids

    # Save per-sample RTF, processing time, and audio duration (same value repeated for all samples in the batch)
    batch["rtf"] = [batch_rtf] * len(batch["audio"])
    batch["processing_time"] = [processing_time] * len(batch["audio"])
    batch["audio_duration"] = [total_audio_duration] * len(batch["audio"])

    return batch


def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=16):
    total_processing_time = 0.0
    total_audio_duration = 0.0
    batch_counter = 0
    device_type = model.device.type

    # Track batch-specific metrics
    batch_rtfs = []
    batch_times = []

    # Print device where model is running
    print(f"Model is on device: {model.device}")

    # Verify device placement of model parameters
    param_device = next(model.parameters()).device
    print(f"Model parameters are on: {param_device}")

    def process_batch(batch):
        nonlocal batch_counter, total_processing_time, total_audio_duration

        # Process the batch and update the cumulative totals
        try:
            result = transcribe_batch(batch, model, processor, memory_tracker, split, batch_counter)

            # Each sample in the batch has the same processing time and audio duration;
            # take the value from the first sample as representative.
            batch_processing_time = result["processing_time"][0]
            batch_audio_duration = result["audio_duration"][0]
            batch_rtf = batch_processing_time / batch_audio_duration

            # Store batch metrics
            batch_rtfs.append(batch_rtf)
            batch_times.append(batch_processing_time)

            print(
                f"Batch {batch_counter}: processing time = {batch_processing_time:.2f}s, "
                f"audio duration = {batch_audio_duration:.2f}s, "
                f"RTF = {batch_rtf:.6f}"
            )

            total_processing_time += batch_processing_time
            total_audio_duration += batch_audio_duration
            batch_counter += 1

            # Force synchronization based on device type
            if device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device_type == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception as e:
            print(f"Error processing batch {batch_counter}: {e}")
            # Return batch without predictions if there was an error
            batch["prediction"] = [""] * len(batch["audio"])
            batch["rtf"] = [0.0] * len(batch["audio"])
            batch["processing_time"] = [0.0] * len(batch["audio"])
            batch["audio_duration"] = [0.0] * len(batch["audio"])
            return batch

        # Clear memory after each batch
        clear_gpu_memory()

        return result

    start = time.time()
    try:
        result = dataset.map(process_batch, batched=True, batch_size=batch_size)
    except Exception as e:
        print(f"Error during dataset mapping: {e}")
        return {"error": str(e)}, {"references": [], "predictions": []}
    end = time.time()

    # Calculate overall RTF from the accumulated totals
    overall_rtf = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0

    # Calculate batch statistics
    avg_batch_rtf = sum(batch_rtfs) / len(batch_rtfs) if batch_rtfs else 0
    min_batch_rtf = min(batch_rtfs) if batch_rtfs else 0
    max_batch_rtf = max(batch_rtfs) if batch_rtfs else 0
    std_batch_rtf = float(np.std(batch_rtfs)) if batch_rtfs else 0

    print("\nRTF Statistics:")
    print(f"  Overall RTF: {overall_rtf:.6f}")
    print(f"  Average Batch RTF: {avg_batch_rtf:.6f}")
    print(f"  Min Batch RTF: {min_batch_rtf:.6f}")
    print(f"  Max Batch RTF: {max_batch_rtf:.6f}")
    print(f"  RTF Std Dev: {std_batch_rtf:.6f}")

    # Compute metrics (e.g., WER, CER)
    scores = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["WER", "CER"]:
            try:
                score = 100 * metric.compute(
                    references=result["reference"], predictions=result["prediction"]
                )
                scores[metric_name] = score
                print(f"{metric_name}: {score:.5f}")
            except Exception as e:
                print(f"Error computing {metric_name}: {e}")
                scores[metric_name] = -1.0

    # Store all metrics in scores dictionary
    scores["RTF"] = overall_rtf
    scores["avg_batch_rtf"] = avg_batch_rtf
    scores["min_batch_rtf"] = min_batch_rtf
    scores["max_batch_rtf"] = max_batch_rtf
    scores["std_batch_rtf"] = std_batch_rtf
    scores["total_processing_time"] = total_processing_time
    scores["total_audio_duration"] = total_audio_duration
    scores["avg_latency"] = total_processing_time / batch_counter if batch_counter > 0 else 0

    # Record CPU metrics
    try:
        summary = memory_tracker.get_memory_summary()
        scores["avg_cpu_percent"] = summary["cpu"]["average_percent"]
        scores["peak_cpu_percent"] = summary["cpu"]["peak_percent"]
        scores["initial_ram_gb"] = summary["cpu"]["initial_ram_gb"]
        scores["peak_ram_gb"] = summary["cpu"]["peak_ram_gb"]
        scores["avg_ram_gb"] = summary["cpu"]["average_ram_gb"]
        scores["current_ram_gb"] = summary["cpu"]["current_ram_gb"]

        # Record GPU metrics if available
        if "gpu" in summary:
            scores["gpu_peak_allocated_gb"] = summary["gpu"]["peak_allocated_gb"]
            scores["gpu_average_allocated_gb"] = summary["gpu"]["average_allocated_gb"]
    except Exception as e:
        print(f"Error recording memory metrics: {e}")

    print(f"{len(result)} sentences evaluated in {end - start:.2f} s.")
    print(f"Average batch latency: {scores['avg_latency']:.4f} s")
    print(f"Total processing time: {total_processing_time:.2f} s")
    print(f"Total audio duration: {total_audio_duration:.2f} s")

    return scores, {"references": result["reference"], "predictions": result["prediction"]}


def load_librispeech(num_samples=None, split="test.clean"):
    """
    Load LibriSpeech clean/other data.
    """
    if num_samples:
        # Stream partial dataset
        stream_dataset = datasets.load_dataset(
            "librispeech_asr", split=split, streaming=True, trust_remote_code=True
        )
        dataset = datasets.Dataset.from_dict(
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


def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    try:
        buffer = io.BytesIO()
        torch.save(
            model.state_dict(), buffer, _use_new_zipfile_serialization=True
        )  # Use new serialization
        return buffer.getbuffer().nbytes / (1024**2)
    except Exception as e:
        print(f"Error measuring model size: {e}")
        return 0.0


def compare_component_sizes(results):
    """Create a bar chart comparing sparsity in encoder vs decoder components."""
    plt.figure(figsize=(12, 8))

    # Extract data
    configs = []
    encoder_sparsities = []
    decoder_sparsities = []

    for model_name, result in results.items():
        if "clean" in model_name and "encoder_sparsity" in result:
            # Get config name without split
            config = model_name.replace("_clean", "")

            configs.append(config)
            encoder_sparsities.append(result["encoder_sparsity"])
            decoder_sparsities.append(result["decoder_sparsity"])

    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width / 2, encoder_sparsities, width, label="Encoder Sparsity")
    plt.bar(x + width / 2, decoder_sparsities, width, label="Decoder Sparsity")

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Sparsity (%)")
    plt.title("Encoder vs Decoder Sparsity by Pruning Configuration")
    plt.xticks(x, configs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "encoder_vs_decoder_sparsity.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved component comparison plot to {plot_path}")


def compare_attention_ffn_sparsity(results):
    """Create a bar chart comparing sparsity in attention vs feed-forward components."""
    plt.figure(figsize=(12, 8))

    # Extract data
    configs = []
    attention_sparsities = []
    ffn_sparsities = []

    for model_name, result in results.items():
        if "clean" in model_name and "attention_sparsity" in result:
            # Get config name without split
            config = model_name.replace("_clean", "")

            configs.append(config)
            attention_sparsities.append(result["attention_sparsity"])
            ffn_sparsities.append(result["ffn_sparsity"])

    if not configs:
        print("No attention/ffn sparsity data available for plotting")
        return

    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width / 2, attention_sparsities, width, label="Attention Sparsity")
    plt.bar(x + width / 2, ffn_sparsities, width, label="Feed-Forward Sparsity")

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Sparsity (%)")
    plt.title("Attention vs Feed-Forward Sparsity by Pruning Configuration")
    plt.xticks(x, configs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "attention_vs_ffn_sparsity.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attention vs feed-forward comparison plot to {plot_path}")


def compare_layer_depth_sparsity(results):
    """Create a bar chart comparing sparsity across layer depths."""
    plt.figure(figsize=(12, 8))

    # Extract data
    configs = []
    early_sparsities = []
    mid_sparsities = []
    late_sparsities = []

    for model_name, result in results.items():
        if "clean" in model_name and "early_layers_sparsity" in result:
            # Get config name without split
            config = model_name.replace("_clean", "")

            configs.append(config)
            early_sparsities.append(result["early_layers_sparsity"])
            mid_sparsities.append(result["mid_layers_sparsity"])
            late_sparsities.append(result["late_layers_sparsity"])

    if not configs:
        print("No layer depth sparsity data available for plotting")
        return

    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.25

    plt.bar(x - width, early_sparsities, width, label="Early Layers (0-2)")
    plt.bar(x, mid_sparsities, width, label="Middle Layers (3-5)")
    plt.bar(x + width, late_sparsities, width, label="Late Layers (6+)")

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Sparsity (%)")
    plt.title("Sparsity by Layer Depth")
    plt.xticks(x, configs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "layer_depth_sparsity.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved layer depth comparison plot to {plot_path}")


def plot_parameter_counts(results):
    """Create a plot showing parameter counts for each pruning configuration."""
    plt.figure(figsize=(12, 8))

    # Extract data
    configs = []
    total_params = []
    nonzero_params = []

    for model_name, result in results.items():
        if "clean" in model_name and "total_parameters" in result:
            # Get config name without split
            config = model_name.replace("_clean", "")

            configs.append(config)
            total_params.append(result["total_parameters"])
            nonzero_params.append(result["non_zero_parameters"])

    if not configs:
        print("No parameter count data available for plotting")
        return

    # Create grouped bar chart
    x = np.arange(len(configs))
    width = 0.35

    plt.bar(x - width / 2, [t / 1_000_000 for t in total_params], width, label="Total Parameters")
    plt.bar(
        x + width / 2, [n / 1_000_000 for n in nonzero_params], width, label="Non-zero Parameters"
    )

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Parameters (millions)")
    plt.title("Parameter Counts by Pruning Configuration")
    plt.xticks(x, configs, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(PLOTS_DIR, "parameter_counts.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter count plot to {plot_path}")


def create_plots(results, metric_names, plot_dir):
    """
    Create plots of metrics for different pruning configurations.

    Args:
        results: Dictionary of results
        metric_names: List of metric names to plot
        plot_dir: Directory to save plots
    """
    print("\nGenerating plots...")

    # Create individual plots for each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))

        # Separate results by split
        metrics_by_split = {"clean": {}, "other": {}}

        for model_name, model_results in results.items():
            # Skip baselines for these specific plots
            if "baseline" in model_name:
                continue

            split = "clean" if "clean" in model_name else "other"
            config = model_name.replace(f"_{split}", "")

            if "metrics" in model_results and metric in model_results["metrics"]:
                if config not in metrics_by_split[split]:
                    metrics_by_split[split][config] = model_results["metrics"][metric]

        # Plot for both splits
        for split, config_metrics in metrics_by_split.items():
            configs = list(config_metrics.keys())
            values = list(config_metrics.values())

            # If we have a baseline, calculate relative changes
            baseline_key = f"baseline_{split}"
            if baseline_key in results:
                baseline_value = results[baseline_key]["metrics"][metric]
                # Sort configs by alphabetical order
                configs_values = sorted(zip(configs, values), key=lambda x: x[0])
                configs = [c for c, _ in configs_values]
                values = [v for _, v in configs_values]
                plt.plot(configs, values, marker="o", label=f"{split} split")
            else:
                # Sort configs by alphabetical order
                configs_values = sorted(zip(configs, values), key=lambda x: x[0])
                configs = [c for c, _ in configs_values]
                values = [v for _, v in configs_values]
                plt.plot(configs, values, marker="o", label=f"{split} split")

        # Add labels and title
        plt.xlabel("Pruning Configuration")
        plt.ylabel(metric)
        plt.title(f"{metric} by Pruning Configuration")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(plot_dir, f"{metric}_by_config.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")

    # Create model size and theoretical dense pruned size plot
    plt.figure(figsize=(10, 6))

    # Extract model sizes by configuration
    dense_sizes = {}
    theoretical_dense_pruned_sizes = {}

    for model_name, model_results in results.items():
        if "clean" in model_name:  # Just use clean split for sizes
            config = model_name.replace("_clean", "")
            if "model_size_mb" in model_results:
                dense_sizes[config] = model_results["model_size_mb"]
            if "theoretical_dense_pruned_size_mb" in model_results:
                theoretical_dense_pruned_sizes[config] = model_results[
                    "theoretical_dense_pruned_size_mb"
                ]

    # Sort configs by name
    configs = sorted(dense_sizes.keys())
    dense_values = [dense_sizes[c] for c in configs]

    # Plot dense sizes
    plt.bar(
        [i - 0.2 for i in range(len(configs))], dense_values, width=0.4, label="Original model size"
    )

    # Plot theoretical dense pruned sizes if available
    if theoretical_dense_pruned_sizes:
        theoretical_values = [theoretical_dense_pruned_sizes.get(c, 0) for c in configs]
        plt.bar(
            [i + 0.2 for i in range(len(configs))],
            theoretical_values,
            width=0.4,
            label="Theoretical pruned size",
        )

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size by Pruning Configuration")
    plt.grid(True, axis="y")
    plt.legend()
    plt.xticks(
        range(len(configs)), configs, rotation=45, ha="right"
    )  # Rotate labels for better readability
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plot_dir, "model_size_by_config.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Create GFLOPs plot
    plt.figure(figsize=(10, 6))

    # Extract GFLOPs values by configuration
    gflops_data = {}

    for model_name, model_results in results.items():
        if "clean" in model_name:  # Just use clean split for GFLOPs
            config = model_name.replace("_clean", "")
            if "gflops" in model_results:
                gflops_data[config] = model_results["gflops"]

    if gflops_data:
        # Sort configs by name
        configs = sorted(gflops_data.keys())
        gflops_values = [gflops_data[c] for c in configs]

        # Plot GFLOPs
        plt.bar(configs, gflops_values)

        plt.xlabel("Pruning Configuration")
        plt.ylabel("GFLOPs")
        plt.title("Computational Complexity (GFLOPs) by Configuration")
        plt.grid(True, axis="y")
        plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(plot_dir, "gflops_by_config.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved GFLOPs plot to {plot_path}")

    # Create component comparison plots
    any_has_components = any("encoder_sparsity" in result for result in results.values())
    if any_has_components:
        compare_component_sizes(results)

    # Create attention vs ffn comparison if data available
    any_has_att_ffn = any("attention_sparsity" in result for result in results.values())
    if any_has_att_ffn:
        compare_attention_ffn_sparsity(results)

    # Create layer depth comparison if data available
    any_has_layer_depth = any("early_layers_sparsity" in result for result in results.values())
    if any_has_layer_depth:
        compare_layer_depth_sparsity(results)

    # Create parameter count plot
    any_has_params = any("total_parameters" in result for result in results.values())
    if any_has_params:
        plot_parameter_counts(results)


def main():
    # Configuration
    original_model_name = "openai/whisper-small"
    batch_size = 16  # Match the quantization code batch size
    save_path = SELECTIVE_PRUNING_DIR
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (
        not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    print(f"Using {device}")

    # Define the pruning configurations
    pruning_configs = {
        # Baseline (no pruning)
        "baseline": {"pruning_config": None},
        # Basic L1 Unstructured Pruning
        # Encoder-only pruning
        "encoder_only_30": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.3,
                "target_submodules": ["encoder"],
                "make_permanent": True,
            }
        },
        "encoder_only_40": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.4,
                "target_submodules": ["encoder"],
                "make_permanent": True,
            }
        },
        # Decoder-only pruning
        "decoder_only_30": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.3,
                "target_submodules": ["decoder"],
                "make_permanent": True,
            }
        },
        "decoder_only_40": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.4,
                "target_submodules": ["decoder"],
                "make_permanent": True,
            }
        },
        # Combined pruning
        "combined_encoder_decoder_30": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.3,
                "target_submodules": ["encoder", "decoder"],
                "make_permanent": True,
            }
        },
        # Layerwise Pruning
        # Early vs Late Layer Pruning
        "early_layers_40": {
            "pruning_config": {
                "method": "custom",  # Will handle specially
                "make_permanent": True,
                "description": "40% pruning on early layers (0-2)",
            }
        },
        "late_layers_40": {
            "pruning_config": {
                "method": "custom",  # Will handle specially
                "make_permanent": True,
                "description": "40% pruning on late layers (6+)",
            }
        },
        # Progressive layerwise pruning
        "progressive_layerwise": {
            "pruning_config": {
                "method": "custom",  # Will handle specially
                "make_permanent": True,
                "description": "Progressive pruning: 10% early, 20% mid, 40% late layers",
            }
        },
        # Component-Specific Pruning
        # Attention and Feed-Forward components
        "attention_only_40": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.4,
                "target_submodules": ["attention", "attn", "k_proj", "q_proj", "v_proj", "o_proj"],
                "make_permanent": True,
            }
        },
        "ffn_only_40": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.4,
                "target_submodules": ["feed_forward", "fc", "mlp"],
                "make_permanent": True,
            }
        },
        # Balanced attention vs feed-forward pruning
        "attention_vs_ffn": {
            "pruning_config": {
                "method": "attention_vs_ffn",
                "attention_amount": 0.2,  # Less pruning for attention
                "ffn_amount": 0.4,  # More pruning for feed-forward
                "pruning_method": "l1_unstructured",
            }
        },
        # Structural Pruning Methods
        # Attention Head Pruning
        "head_pruning_40": {
            "pruning_config": {
                "method": "attention_head_pruning",
                "amount": 0.4,  # Prune 40% of attention heads
            }
        },
        # Complete head removal for specific layers
        "head_removal_early": {
            "pruning_config": {
                "method": "custom_head_removal",
                "make_permanent": True,
                "description": "Remove heads completely from early layers (0-2)",
                "layers_for_head_removal": [0, 1, 2],
                "head_removal_percentage": 1.0,  # Complete removal
            }
        },
        # Layer Dropping
        "layer_dropping": {
            "pruning_config": {
                "method": "layer_dropping",
                "layers_to_drop": {
                    "encoder": [0, 2, 4],  # Drop encoder layers 0, 2, 4
                    "decoder": [1, 3, 5],  # Drop decoder layers 1, 3, 5
                },
            }
        },
        # MLP/Feed-Forward Removal
        "mlp_removal": {
            "pruning_config": {
                "method": "custom_mlp_removal",
                "make_permanent": True,
                "description": "Remove entire MLP blocks from specific layers",
                "layers_for_mlp_removal": [0, 3, 6],  # Remove MLPs from these layers
            }
        },
        # Alternative Pruning Methods
        # Global magnitude pruning
        "magnitude_pruning_30": {
            "pruning_config": {
                "method": "custom_magnitude",
                "make_permanent": True,
                "description": "30% global magnitude pruning",
                "threshold_percentage": 0.3,
            }
        },
        # Mixed strategy pruning
        "mixed_strategy": {
            "pruning_config": {
                "method": "custom_mixed",
                "make_permanent": True,
                "description": "Head pruning + unstructured weight pruning",
                "head_pruning_amount": 0.3,
                "weight_pruning_amount": 0.2,
            }
        },
    }

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load full datasets
    print("\nLoading datasets...")
    dataset_clean = load_librispeech(split="test.clean")  # Use full test.clean
    dataset_other = load_librispeech(split="test.other")  # Use full test.other

    print(f"Clean dataset: {len(dataset_clean)} samples")
    print(f"Other dataset: {len(dataset_other)} samples")

    # Process datasets
    print("\nProcessing datasets...")
    processed_test_data_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))
    processed_test_data_other = dataset_other.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Evaluate each pruning configuration
    for model_name, config in pruning_configs.items():
        print("\n" + "=" * 50)
        print(f"Evaluating {model_name}")
        print("=" * 50)

        # Clear memory before loading new model
        clear_gpu_memory()

        try:
            if model_name == "early_layers_40":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Apply pruning specifically to early layers
                print("Applying 40% pruning to early layers (0-2)...")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Check if it's an early layer
                        layer_idx = -1
                        if "layers." in name:
                            try:
                                layer_str = name.split("layers.")[1].split(".")[0]
                                layer_idx = int(layer_str)
                            except (ValueError, IndexError):
                                pass

                        if layer_idx >= 0 and layer_idx < 3:
                            # Apply heavy pruning to early layers
                            prune.l1_unstructured(module, "weight", amount=0.4)
                            print(f"  Pruned early layer: {name}")

                # Make pruning permanent
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        try:
                            prune.remove(module, "weight")
                        except:
                            pass

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "late_layers_40":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Apply pruning specifically to late layers
                print("Applying 40% pruning to late layers (6+)...")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Check if it's a late layer
                        layer_idx = -1
                        if "layers." in name:
                            try:
                                layer_str = name.split("layers.")[1].split(".")[0]
                                layer_idx = int(layer_str)
                            except (ValueError, IndexError):
                                pass

                        if layer_idx >= 6:
                            # Apply heavy pruning to late layers
                            prune.l1_unstructured(module, "weight", amount=0.4)
                            print(f"  Pruned late layer: {name}")

                # Make pruning permanent
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        try:
                            prune.remove(module, "weight")
                        except:
                            pass

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "progressive_layerwise":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Apply progressive pruning by layer depth
                print("Applying progressive layerwise pruning...")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Determine layer depth from name
                        layer_idx = -1
                        if "layers." in name:
                            try:
                                layer_str = name.split("layers.")[1].split(".")[0]
                                layer_idx = int(layer_str)
                            except (ValueError, IndexError):
                                pass

                        if layer_idx >= 0:
                            # Early layers (0-2): 10%, Middle layers (3-5): 20%, Late layers (6+): 40%
                            if layer_idx < 3:
                                amount = 0.1
                            elif layer_idx < 6:
                                amount = 0.2
                            else:
                                amount = 0.4

                            prune.l1_unstructured(module, "weight", amount=amount)
                            print(
                                f"  Pruned layer {layer_idx} ({name}) with {amount*100}% sparsity"
                            )

                # Make pruning permanent
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        try:
                            prune.remove(module, "weight")
                        except:
                            pass

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "head_removal_early":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Get the pruning config
                layers_for_head_removal = pruning_configs[model_name]["pruning_config"][
                    "layers_for_head_removal"
                ]

                print(f"Removing attention heads completely from layers: {layers_for_head_removal}")

                # Find attention modules in the target layers
                for name, module in model.named_modules():
                    layer_idx = -1
                    if "layers." in name:
                        try:
                            layer_str = name.split("layers.")[1].split(".")[0]
                            layer_idx = int(layer_str)
                        except (ValueError, IndexError):
                            pass

                    # Check if this layer should have heads removed
                    if layer_idx in layers_for_head_removal:
                        # Find attention projection layers (q, k, v)
                        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                            # Zero out entire weight matrices for these projection layers
                            if hasattr(module, "weight"):
                                print(f"  Removing attention heads in layer {layer_idx}: {name}")
                                module.weight.data.zero_()
                                # Also zero bias if it exists
                                if hasattr(module, "bias") and module.bias is not None:
                                    module.bias.data.zero_()

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "mlp_removal":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Get the pruning config
                layers_for_mlp_removal = pruning_configs[model_name]["pruning_config"][
                    "layers_for_mlp_removal"
                ]

                print(
                    f"Removing MLP/feed-forward blocks completely from layers: {layers_for_mlp_removal}"
                )

                # Find MLP modules in the target layers
                for name, module in model.named_modules():
                    layer_idx = -1
                    if "layers." in name:
                        try:
                            layer_str = name.split("layers.")[1].split(".")[0]
                            layer_idx = int(layer_str)
                        except (ValueError, IndexError):
                            pass

                    # Check if this layer should have MLP removed
                    if layer_idx in layers_for_mlp_removal:
                        # Find feed-forward/MLP components
                        if any(ff_part in name for ff_part in ["feed_forward", "fc", "mlp"]):
                            # Zero out weights for these feed-forward components
                            if hasattr(module, "weight"):
                                print(f"  Removing MLP in layer {layer_idx}: {name}")
                                module.weight.data.zero_()
                                # Also zero bias if it exists
                                if hasattr(module, "bias") and module.bias is not None:
                                    module.bias.data.zero_()

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "magnitude_pruning_30":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Get the pruning config
                threshold_percentage = pruning_configs[model_name]["pruning_config"][
                    "threshold_percentage"
                ]

                print(
                    f"Applying global magnitude pruning with threshold {threshold_percentage*100}%"
                )

                # Collect all weights
                all_weights = []
                for name, param in model.named_parameters():
                    if "weight" in name:
                        all_weights.append(param.abs().flatten())

                # Concatenate all weights and find the magnitude threshold
                all_weights_tensor = torch.cat(all_weights)
                threshold_idx = int(threshold_percentage * all_weights_tensor.numel())
                threshold_value = torch.sort(all_weights_tensor)[0][threshold_idx]

                print(f"  Magnitude threshold: {threshold_value:.6f}")

                # Apply threshold to all weights
                zero_count = 0
                total_count = 0
                for name, param in model.named_parameters():
                    if "weight" in name:
                        mask = param.abs() <= threshold_value
                        param.data[mask] = 0.0
                        zero_count += mask.sum().item()
                        total_count += param.numel()

                # Report actual sparsity achieved
                actual_sparsity = 100 * zero_count / total_count
                print(f"  Actual sparsity achieved: {actual_sparsity:.2f}%")

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            elif model_name == "mixed_strategy":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Get the pruning config
                head_pruning_amount = pruning_configs[model_name]["pruning_config"][
                    "head_pruning_amount"
                ]
                weight_pruning_amount = pruning_configs[model_name]["pruning_config"][
                    "weight_pruning_amount"
                ]

                print(
                    f"Applying mixed strategy: {head_pruning_amount*100}% head pruning + {weight_pruning_amount*100}% weight pruning"
                )

                # 1. First, apply head pruning
                model = apply_attention_head_pruning(model, amount=head_pruning_amount)

                # 2. Then, apply unstructured weight pruning to the rest of the model
                # Identify non-attention modules
                non_attention_params = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Skip attention components
                        if not any(
                            att_part in name
                            for att_part in [
                                "attention",
                                "attn",
                                "k_proj",
                                "q_proj",
                                "v_proj",
                                "o_proj",
                            ]
                        ):
                            non_attention_params.append((module, "weight"))

                print(
                    f"  Applying weight pruning to {len(non_attention_params)} non-attention modules"
                )
                if non_attention_params:
                    prune.global_unstructured(
                        non_attention_params,
                        pruning_method=prune.L1Unstructured,
                        amount=weight_pruning_amount,
                    )

                # Make pruning permanent
                for module, param_name in non_attention_params:
                    try:
                        prune.remove(module, param_name)
                    except:
                        pass

                # Move to device
                model = model.to(device)
                model.config.forced_decoder_ids = None

            else:
                # Regular pruning from config
                model = load_whisper_model(
                    model_name=original_model_name,
                    device=device,
                    pruning_config=config["pruning_config"],
                )

            # Calculate component sparsity for all models
            sparsity_info = calculate_component_sparsity(model)
            print("Sparsity by component:")
            print(f"  - Encoder: {sparsity_info['encoder_sparsity']:.2f}%")
            print(f"  - Decoder: {sparsity_info['decoder_sparsity']:.2f}%")
            print(f"  - Attention: {sparsity_info['attention_sparsity']:.2f}%")
            print(f"  - Feed-Forward: {sparsity_info['ffn_sparsity']:.2f}%")
            print(f"  - Early Layers: {sparsity_info['early_layers_sparsity']:.2f}%")
            print(f"  - Mid Layers: {sparsity_info['mid_layers_sparsity']:.2f}%")
            print(f"  - Late Layers: {sparsity_info['late_layers_sparsity']:.2f}%")
            print(f"  - Overall: {sparsity_info['overall_sparsity']:.2f}%")
            print(f"  - Total Parameters: {sparsity_info['total_parameters']:,}")
            print(f"  - Non-zero Parameters: {sparsity_info['non_zero_parameters']:,}")

            # Calculate GFLOPs
            gflops = calculate_model_gflops(model)
            print(f"Estimated model complexity: {gflops:.4f} GFLOPs")

            # Calculate theoretical dense pruned size
            theoretical_dense_pruned_size = 0.0
            if model_name != "baseline":  # Skip for baseline model
                theoretical_dense_pruned_size = calculate_pruned_dense_size(
                    model, pruning_threshold=0.0
                )

            # Evaluate on both splits
            for split, dataset in [
                ("clean", processed_test_data_clean),
                ("other", processed_test_data_other),
            ]:
                print(f"\nEvaluating on {split} split...")

                # Initialize memory tracker for this run
                tracker = WhisperMemoryTracker(f"{model_name}_{split}", save_path)

                try:
                    # Run evaluation
                    scores, transcriptions = evaluate_model(
                        model=model,
                        processor=processor,
                        dataset=dataset,
                        metrics=metrics,
                        memory_tracker=tracker,
                        batch_size=batch_size,
                        split=split,
                    )

                    # Store and save results
                    if isinstance(scores, dict) and "error" not in scores:
                        # Get model size
                        model_size = get_model_disk_size_in_mb(model)

                        # Build results dictionary
                        results[f"{model_name}_{split}"] = {
                            "metrics": scores,
                            "model_size_mb": model_size,
                            "model_type": model_name,
                            "gflops": gflops,  # Add GFLOPs to results
                            "encoder_sparsity": sparsity_info["encoder_sparsity"],
                            "decoder_sparsity": sparsity_info["decoder_sparsity"],
                            "attention_sparsity": sparsity_info["attention_sparsity"],
                            "ffn_sparsity": sparsity_info["ffn_sparsity"],
                            "early_layers_sparsity": sparsity_info["early_layers_sparsity"],
                            "mid_layers_sparsity": sparsity_info["mid_layers_sparsity"],
                            "late_layers_sparsity": sparsity_info["late_layers_sparsity"],
                            "overall_sparsity": sparsity_info["overall_sparsity"],
                            "total_parameters": sparsity_info["total_parameters"],
                            "non_zero_parameters": sparsity_info["non_zero_parameters"],
                            "theoretical_dense_pruned_size_mb": theoretical_dense_pruned_size,
                        }

                        # Save metrics
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_summary.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[f"{model_name}_{split}"], f, indent=2)
                    else:
                        # Handle error
                        error_msg = (
                            scores.get("error", "Unknown error")
                            if isinstance(scores, dict)
                            else str(scores)
                        )
                        print(f"Error during evaluation: {error_msg}")
                        results[f"{model_name}_{split}"]["evaluation_error"] = error_msg

                except Exception as e:
                    print(f"Error evaluating {model_name} on {split} split: {e!s}")
                    continue

                finally:
                    # Always close tracker and clear memory
                    tracker.close()

            # Clean up model
            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"Error setting up {model_name}: {e!s}")
            continue

    # Save all results to a single file
    all_results_path = os.path.join(SELECTIVE_PRUNING_DIR, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {all_results_path}")

    # Create plots
    create_plots(
        results=results,
        metric_names=["WER", "CER", "RTF", "avg_cpu_percent", "peak_ram_gb", "gflops"],
        plot_dir=PLOTS_DIR,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SELECTIVE PRUNING EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nBaseline (no pruning):")
    if "baseline_clean" in results:
        baseline = results["baseline_clean"]
        print(f"  WER: {baseline['metrics']['WER']:.4f}")
        print(f"  CER: {baseline['metrics']['CER']:.4f}")
        print(f"  RTF: {baseline['metrics']['RTF']:.4f}")
        print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
        print(f"  GFLOPs: {baseline['gflops']:.4f}")
        print(f"  Total Parameters: {baseline['total_parameters']:,}")
        print(f"  Non-zero Parameters: {baseline['non_zero_parameters']:,}")

    # Group results by pruning approach
    config_groups = {
        "Basic L1 Unstructured Pruning": [
            "encoder_only_30",
            "encoder_only_40",
            "decoder_only_30",
            "decoder_only_40",
            "combined_encoder_decoder_30",
        ],
        "Layerwise Pruning": ["early_layers_40", "late_layers_40", "progressive_layerwise"],
        "Component-Specific Pruning": ["attention_only_40", "ffn_only_40", "attention_vs_ffn"],
        "Structural Pruning Methods": [
            "head_pruning_40",
            "head_removal_early",
            "layer_dropping",
            "mlp_removal",
        ],
        "Alternative Pruning Methods": ["magnitude_pruning_30", "mixed_strategy"],
    }

    for group_name, configs in config_groups.items():
        print(f"\n{group_name}:")
        for config in configs:
            result_key = f"{config}_clean"
            if result_key in results:
                result = results[result_key]

                # Calculate changes from baseline
                wer_change = "-"
                cer_change = "-"
                rtf_change = "-"
                theoretical_size_change = "-"
                gflops_change = "-"
                param_change = "-"

                if "baseline_clean" in results:
                    baseline = results["baseline_clean"]
                    wer_change = f"{(result['metrics']['WER'] - baseline['metrics']['WER']) / baseline['metrics']['WER'] * 100:+.2f}%"
                    cer_change = f"{(result['metrics']['CER'] - baseline['metrics']['CER']) / baseline['metrics']['CER'] * 100:+.2f}%"
                    rtf_change = f"{(result['metrics']['RTF'] - baseline['metrics']['RTF']) / baseline['metrics']['RTF'] * 100:+.2f}%"
                    gflops_change = f"{(result['gflops'] - baseline['gflops']) / baseline['gflops'] * 100:+.2f}%"
                    param_change = f"{(result['non_zero_parameters'] - baseline['non_zero_parameters']) / baseline['non_zero_parameters'] * 100:+.2f}%"

                    if (
                        "theoretical_dense_pruned_size_mb" in result
                        and result["theoretical_dense_pruned_size_mb"] > 0
                    ):
                        theoretical_size_change = f"{(result['theoretical_dense_pruned_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%"

                print(f"\n  {config}:")
                print(f"    WER: {result['metrics']['WER']:.4f} ({wer_change})")
                print(f"    CER: {result['metrics']['CER']:.4f} ({cer_change})")
                print(f"    RTF: {result['metrics']['RTF']:.4f} ({rtf_change})")
                print(f"    GFLOPs: {result['gflops']:.4f} ({gflops_change})")

                if (
                    "theoretical_dense_pruned_size_mb" in result
                    and result["theoretical_dense_pruned_size_mb"] > 0
                ):
                    print(
                        f"    Theoretical Dense Pruned Size: {result['theoretical_dense_pruned_size_mb']:.2f} MB ({theoretical_size_change})"
                    )

                print(f"    Encoder Sparsity: {result['encoder_sparsity']:.2f}%")
                print(f"    Decoder Sparsity: {result['decoder_sparsity']:.2f}%")
                print(f"    Attention Sparsity: {result['attention_sparsity']:.2f}%")
                print(f"    Feed-Forward Sparsity: {result['ffn_sparsity']:.2f}%")
                print(f"    Overall Sparsity: {result['overall_sparsity']:.2f}%")
                print(f"    Total Parameters: {result['total_parameters']:,}")
                print(
                    f"    Non-zero Parameters: {result['non_zero_parameters']:,} ({param_change})"
                )

    print("\nPlots saved to:", PLOTS_DIR)
    print("Detailed metrics saved to:", SELECTIVE_PRUNING_DIR)


if __name__ == "__main__":
    main()
