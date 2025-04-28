import gc
import io
import json
import os
import time
from collections import deque

import datasets
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn.utils.prune as prune
from evaluate import load
from optimum.quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize
from transformers import (
    BitsAndBytesConfig,
    HqqConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Set seaborn style
sns.set(style="whitegrid")

# Create results directory
RESULTS_DIR = "pruning_quantization_results"
PRUNED_MODEL_DIR = os.path.join(RESULTS_DIR, "pruned_model")
COMBINED_RESULTS_DIR = os.path.join(RESULTS_DIR, "pruned_quantized")
PLOTS_DIR = os.path.join(COMBINED_RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(COMBINED_RESULTS_DIR, "models")

for directory in [RESULTS_DIR, PRUNED_MODEL_DIR, COMBINED_RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


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


def calculate_pruned_dense_size(model, pruning_threshold=0.0):
    """
    Calculate the theoretical size of a dense model with pruned weights and biases removed.
    This doesn't actually create the model, just calculates what the size would be.

    Args:
        model: The pruned model with masked weights and biases
        pruning_threshold: Values with absolute value below this threshold are considered pruned

    Returns:
        float: Size in MB that a dense model with pruned weights and biases removed would have
    """
    print(
        "\n=== Calculating theoretical dense model size with pruned weights and biases removed ==="
    )

    total_params_original = 0
    total_params_pruned = 0
    total_bytes_original = 0
    total_bytes_dense_pruned = 0

    # Count parameters and calculate theoretical size
    for name, param in model.named_parameters():
        param_size_bytes = param.numel() * 4  # 4 bytes per float32
        total_params_original += param.numel()
        total_bytes_original += param_size_bytes

        # Find pruned values in weights or biases
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


def calculate_theoretical_quantized_size(model, quantization_type, pruning_threshold=0.0):
    """
    Calculate the theoretical size of a model that is both pruned and quantized.

    Args:
        model: The pruned model
        quantization_type: Type of quantization (e.g., "int4", "int8", "float8", etc.)
        pruning_threshold: Values with absolute value below this threshold are considered pruned

    Returns:
        float: Theoretical size in MB after pruning + quantization
    """
    print(f"\n=== Calculating theoretical size with pruning + {quantization_type} quantization ===")

    # Define bits per parameter based on quantization type
    bits_mapping = {
        "int4": 4,
        "nf4": 4,
        "fp4": 4,
        "int8": 8,
        "float8": 8,
        "fp16": 16,
        "fp32": 32,
    }

    # Default to full precision if quantization type not recognized
    bits_per_param = bits_mapping.get(quantization_type, 32)
    bytes_per_param = bits_per_param / 8

    total_params_original = 0
    total_non_zero_params = 0
    total_bytes_original = 0
    total_bytes_quantized = 0

    # Count parameters and calculate theoretical size
    for name, param in model.named_parameters():
        param_size_bytes = param.numel() * 4  # Original size (FP32)
        total_params_original += param.numel()
        total_bytes_original += param_size_bytes

        # Find non-zero values (not pruned)
        pruned_mask = torch.abs(param) <= pruning_threshold
        non_zero_params = param.numel() - torch.sum(pruned_mask).item()
        total_non_zero_params += non_zero_params

        # Calculate size with quantization
        param_quantized_bytes = non_zero_params * bytes_per_param
        total_bytes_quantized += param_quantized_bytes

    # Convert to MB
    original_size_mb = total_bytes_original / (1024 * 1024)
    quantized_size_mb = total_bytes_quantized / (1024 * 1024)

    # Report on size reduction
    if total_params_original > 0:
        pruning_reduction = (
            100.0 * (total_params_original - total_non_zero_params) / total_params_original
        )

        size_reduction = 100.0 * (original_size_mb - quantized_size_mb) / original_size_mb

        print(f"Original parameters: {total_params_original:,}")
        print(f"Non-zero parameters after pruning: {total_non_zero_params:,}")
        print(f"Parameter reduction from pruning: {pruning_reduction:.1f}%")
        print(f"Original size (FP32): {original_size_mb:.2f} MB")
        print(f"Size after quantization to {quantization_type}: {quantized_size_mb:.2f} MB")
        print(f"Overall size reduction: {size_reduction:.1f}%")

    return quantized_size_mb


def calculate_model_gflops(model):
    """
    Calculate approximate GFLOPs for Whisper model accounting for pruning.

    Args:
        model: The WhisperForConditionalGeneration model

    Returns:
        float: Estimated GFLOPs
    """
    # Track FLOPs by module type
    flops_by_type = {"encoder": 0, "decoder": 0, "other": 0}

    total_params = 0
    non_zero_params = 0
    bias_total_params = 0
    bias_non_zero_params = 0

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

            # Calculate sparsity and non-zero operations for weights
            weight_sparsity = (
                torch.sum(weight == 0).item() / weight.numel() if weight.numel() > 0 else 0
            )
            weight_flops = 2 * in_features * out_features * (1 - weight_sparsity)

            # Count weight parameters
            total_params += weight.numel()
            non_zero_params += (weight != 0).sum().item()

            # Count bias parameters separately
            bias_flops = 0
            if hasattr(module, "bias") and module.bias is not None:
                bias = module.bias
                bias_total_params += bias.numel()
                bias_non_zero_params += (bias != 0).sum().item()

                # Calculate sparsity in bias
                bias_sparsity = (
                    torch.sum(bias == 0).item() / bias.numel() if bias.numel() > 0 else 0
                )

                # Add bias FLOPs (proportional to non-zero bias elements)
                bias_flops = out_features * (1 - bias_sparsity)

            # Total FLOPs for this layer
            layer_flops = weight_flops + bias_flops

            # Categorize by location in model
            if "encoder" in name:
                flops_by_type["encoder"] += layer_flops
            elif "decoder" in name:
                flops_by_type["decoder"] += layer_flops
            else:
                flops_by_type["other"] += layer_flops

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
    for component, flops in flops_by_type.items():
        gflops = flops / 1e9
        percentage = (flops / sum(flops_by_type.values())) * 100
        print(f"  {component}: {gflops:.4f} GFLOPs ({percentage:.1f}%)")

    if total_params > 0:
        print("\nWeight parameter efficiency:")
        print(f"  Total weight parameters: {total_params:,}")
        print(f"  Non-zero weight parameters: {non_zero_params:,}")
        print(f"  Weight sparsity: {100 * (1 - non_zero_params / total_params):.2f}%")

    if bias_total_params > 0:
        print("\nBias parameter efficiency:")
        print(f"  Total bias parameters: {bias_total_params:,}")
        print(f"  Non-zero bias parameters: {bias_non_zero_params:,}")
        print(f"  Bias sparsity: {100 * (1 - bias_non_zero_params / bias_total_params):.2f}%")

    print(f"\nTotal estimated GFLOPs: {total_gflops:.4f}")

    return total_gflops


def save_sparse_model(model, output_path):
    """
    Convert pruned model to sparse format for storage efficiency.

    Args:
        model: Pruned PyTorch model
        output_path: Path to save the sparse model

    Returns:
        float: Size of saved sparse model in MB
    """
    sparse_state_dict = {}
    original_params = 0
    sparse_params = 0

    print("\n=== Converting model to sparse format ===")

    # Track sparsity statistics by layer type
    sparsity_by_type = {}

    for name, param in model.state_dict().items():
        # Calculate original size in bytes (assuming float32)
        original_params += param.numel() * 4

        # Determine parameter type for reporting
        param_type = "unknown"
        if "encoder" in name:
            param_type = "encoder"
        elif "decoder" in name:
            param_type = "decoder"

        if "weight" in name:
            param_type += "_weight"
        elif "bias" in name:
            param_type += "_bias"

        # For parameters with significant sparsity, convert to sparse format
        if param.dim() > 0 and torch.sum(param == 0) > 0.3 * param.numel():
            # Calculate sparsity percentage
            total_elements = param.numel()
            zero_elements = torch.sum(param == 0).item()
            sparsity = 100.0 * zero_elements / total_elements

            # Update sparsity stats
            if param_type not in sparsity_by_type:
                sparsity_by_type[param_type] = {"total": 0, "zeros": 0}
            sparsity_by_type[param_type]["total"] += total_elements
            sparsity_by_type[param_type]["zeros"] += zero_elements

            # Convert to sparse tensor
            sparse_param = param.to_sparse()
            sparse_state_dict[name] = sparse_param

            # Calculate expected sparse storage size (rough estimate)
            non_zeros = total_elements - zero_elements
            # COO format: indices (2 * non_zeros * 4 bytes) + values (non_zeros * 4 bytes)
            param_sparse_bytes = (2 * non_zeros * 4) + (non_zeros * 4)
            sparse_params += param_sparse_bytes

            # Print info for significant layers
            if total_elements > 10000:  # Only print for large layers
                print(
                    f"  Converting {name}: {sparsity:.1f}% sparse ({zero_elements}/{total_elements} zeros)"
                )
        else:
            # Keep as dense
            sparse_state_dict[name] = param
            sparse_params += param.numel() * 4  # 4 bytes per float32

    # Save sparse model
    try:
        print(f"Saving sparse model to {output_path}")
        torch.save(sparse_state_dict, output_path)

        # Get actual saved size
        actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        estimated_dense_mb = original_params / (1024 * 1024)
        estimated_sparse_mb = sparse_params / (1024 * 1024)

        # Print sparsity by parameter type
        print("\nSparsity by parameter type:")
        for param_type, stats in sparsity_by_type.items():
            sparsity_pct = 100.0 * stats["zeros"] / stats["total"] if stats["total"] > 0 else 0
            print(
                f"  {param_type}: {sparsity_pct:.1f}% sparse ({stats['zeros']}/{stats['total']} zeros)"
            )

        # Print size statistics
        print("\nModel size summary:")
        print(f"  Dense model size (theoretical): {estimated_dense_mb:.2f} MB")
        print(f"  Sparse model size (theoretical): {estimated_sparse_mb:.2f} MB")
        print(f"  Sparse model size (actual file): {actual_size_mb:.2f} MB")
        print(f"  Compression ratio: {estimated_dense_mb/actual_size_mb:.2f}x")
        print(
            f"  Size reduction: {100.0 * (estimated_dense_mb - actual_size_mb) / estimated_dense_mb:.1f}%"
        )

        return actual_size_mb
    except Exception as e:
        print(f"Error saving sparse model: {e}")
        return 0


def extract_layer_num(name):
    """Extract layer number from module name."""
    try:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                return int(parts[i + 1])
    except:
        pass
    return None


def determine_pruning_amount(name, param_name, module, pruning_config):
    """
    Determine pruning percentage for a specific parameter based on its name and type.

    Args:
        name: Module name
        param_name: Parameter name (weight or bias)
        module: The module
        pruning_config: Dictionary mapping component types to pruning percentages

    Returns:
        Pruning amount (0.0 to 1.0)
    """
    # Default: no pruning
    pruning_amount = 0.0

    # Global bias pruning (applied to all bias parameters)
    if param_name == "bias":
        pruning_amount = pruning_config.get("all_bias", 0.0) / 100.0
        return pruning_amount

    # Below this point, we're handling weight parameters only

    # Encoder FFN
    if "encoder" in name and ("fc1" in name or "fc2" in name):
        pruning_amount = pruning_config.get("encoder_ffn", 0.0) / 100.0

    # Decoder FFN - different amounts based on layer
    elif "decoder" in name and ("fc1" in name or "fc2" in name):
        # Extract layer number
        layer_num = extract_layer_num(name)
        if layer_num is not None:
            if layer_num < 4:  # First 4 layers (0-3)
                pruning_amount = pruning_config.get("decoder_ffn_first", 0.0) / 100.0
            elif layer_num < 8:  # Middle 4 layers (4-7)
                pruning_amount = pruning_config.get("decoder_ffn_middle", 0.0) / 100.0
            else:  # Last 4 layers (8-11)
                pruning_amount = pruning_config.get("decoder_ffn_last", 0.0) / 100.0

    # Encoder Self Attention
    elif (
        "encoder" in name
        and "self_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        pruning_amount = pruning_config.get("encoder_self_attn", 0.0) / 100.0

    # Decoder Self Attention
    elif (
        "decoder" in name
        and "self_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        pruning_amount = pruning_config.get("decoder_self_attn", 0.0) / 100.0

    # Decoder Cross Attention
    elif (
        "decoder" in name
        and "encoder_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        pruning_amount = pruning_config.get("decoder_cross_attn", 0.0) / 100.0

    # LayerNorm
    elif "layer_norm" in name.lower() or "layernorm" in name.lower():
        pruning_amount = pruning_config.get("layer_norm", 0.0) / 100.0

    # Token embeddings
    elif "embed_tokens" in name:
        pruning_amount = pruning_config.get("token_embeddings", 0.0) / 100.0

    # Positional embeddings
    elif "embed_positions" in name:
        pruning_amount = pruning_config.get("positional_embeddings", 0.0) / 100.0

    # Convolutional layers
    elif "conv" in name.lower():
        pruning_amount = pruning_config.get("conv_layers", 0.0) / 100.0

    # Final output projection
    elif "proj_out" in name:
        pruning_amount = pruning_config.get("output_projection", 0.0) / 100.0

    return pruning_amount


def apply_custom_l1_pruning(model, pruning_config, make_permanent=False):
    """
    Apply L1 unstructured pruning to a Whisper model with custom percentages for different components.

    Args:
        model: The WhisperForConditionalGeneration model
        pruning_config: Dictionary mapping component types to pruning percentages
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
    """
    # Track pruned modules by type for reporting
    components_pruned = {
        "encoder_ffn": 0,
        "decoder_ffn_first": 0,
        "decoder_ffn_middle": 0,
        "decoder_ffn_last": 0,
        "encoder_self_attn": 0,
        "decoder_self_attn": 0,
        "decoder_cross_attn": 0,
        "layer_norm": 0,
        "token_embeddings": 0,
        "positional_embeddings": 0,
        "conv_layers": 0,
        "output_projection": 0,
        "all_bias": 0,
        "other": 0,
    }

    # Dictionary to store parameters to prune with their respective amounts
    weight_params_to_prune = []
    bias_params_to_prune = []

    # Iterate through named modules
    for name, module in model.named_modules():
        # Process weight parameters
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            weight_amount = determine_pruning_amount(name, "weight", module, pruning_config)

            if weight_amount > 0:
                weight_params_to_prune.append((module, "weight", weight_amount))

                # Track component type for reporting
                component_type = "other"
                if "encoder" in name and ("fc1" in name or "fc2" in name):
                    component_type = "encoder_ffn"
                elif "decoder" in name and ("fc1" in name or "fc2" in name):
                    layer_num = extract_layer_num(name)
                    if layer_num is not None:
                        if layer_num < 4:
                            component_type = "decoder_ffn_first"
                        elif layer_num < 8:
                            component_type = "decoder_ffn_middle"
                        else:
                            component_type = "decoder_ffn_last"
                elif "encoder" in name and "self_attn" in name:
                    component_type = "encoder_self_attn"
                elif "decoder" in name and "self_attn" in name:
                    component_type = "decoder_self_attn"
                elif "decoder" in name and "encoder_attn" in name:
                    component_type = "decoder_cross_attn"
                elif "layer_norm" in name.lower() or "layernorm" in name.lower():
                    component_type = "layer_norm"
                elif "embed_tokens" in name:
                    component_type = "token_embeddings"
                elif "embed_positions" in name:
                    component_type = "positional_embeddings"
                elif "conv" in name.lower():
                    component_type = "conv_layers"
                elif "proj_out" in name:
                    component_type = "output_projection"

                components_pruned[component_type] += 1

        # Process bias parameters
        if hasattr(module, "bias") and module.bias is not None:
            bias_amount = determine_pruning_amount(name, "bias", module, pruning_config)

            if bias_amount > 0:
                bias_params_to_prune.append((module, "bias", bias_amount))
                components_pruned["all_bias"] += 1

    print(
        f"Found {len(weight_params_to_prune)} weight modules and {len(bias_params_to_prune)} bias modules to prune"
    )

    # Print summary of components to be pruned
    print("\nComponents to be pruned:")
    for component, count in components_pruned.items():
        if count > 0:
            if component == "all_bias":
                amount = pruning_config.get("all_bias", 0)
                print(f"  {component}: {count} modules at {amount}% pruning")
            else:
                amount = pruning_config.get(component, 0)
                print(f"  {component}: {count} modules at {amount}% pruning")

    # Apply pruning individually for each weight parameter
    weight_success_count = 0
    for module, param_name, amount in weight_params_to_prune:
        try:
            prune.l1_unstructured(module, param_name, amount=amount)
            weight_success_count += 1
        except Exception as e:
            print(f"Error pruning weight module {module}: {e}")

    print(
        f"Successfully applied pruning to {weight_success_count}/{len(weight_params_to_prune)} weight modules"
    )

    # Apply pruning individually for each bias parameter
    bias_success_count = 0
    for module, param_name, amount in bias_params_to_prune:
        try:
            prune.l1_unstructured(module, param_name, amount=amount)
            bias_success_count += 1
        except Exception as e:
            print(f"Error pruning bias module {module}: {e}")

    print(
        f"Successfully applied pruning to {bias_success_count}/{len(bias_params_to_prune)} bias modules"
    )

    # Make pruning permanent if requested
    if make_permanent:
        print("Making pruning permanent...")
        permanent_count = 0

        # Make weight pruning permanent
        for module, param_name, _ in weight_params_to_prune:
            try:
                if hasattr(module, f"{param_name}_mask"):
                    prune.remove(module, param_name)
                    permanent_count += 1
            except Exception as e:
                print(f"Could not make pruning permanent for {module}.{param_name}: {e}")

        # Make bias pruning permanent
        for module, param_name, _ in bias_params_to_prune:
            try:
                if hasattr(module, f"{param_name}_mask"):
                    prune.remove(module, param_name)
                    permanent_count += 1
            except Exception as e:
                print(f"Could not make pruning permanent for {module}.{param_name}: {e}")

        print(f"Made pruning permanent for {permanent_count} parameters")

    return model


def calculate_sparsity(model):
    """
    Calculate the sparsity percentage and parameter counts in the model.

    Args:
        model: The PyTorch model

    Returns:
        tuple: (sparsity percentage, total parameters, non-zero parameters,
                bias sparsity percentage, total bias parameters, non-zero bias parameters)
    """
    weight_total_params = 0
    weight_zero_params = 0
    bias_total_params = 0
    bias_zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:  # Weight parameters
            weight_total_params += param.numel()
            weight_zero_params += torch.sum(param == 0).item()
        elif "bias" in name:  # Bias parameters
            bias_total_params += param.numel()
            bias_zero_params += torch.sum(param == 0).item()

    # Calculate overall sparsity
    total_params = weight_total_params + bias_total_params
    zero_params = weight_zero_params + bias_zero_params

    if total_params == 0:
        return 0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0

    # Calculate weight sparsity
    weight_sparsity = 0.0
    if weight_total_params > 0:
        weight_sparsity = 100.0 * weight_zero_params / weight_total_params
    weight_non_zero_params = weight_total_params - weight_zero_params

    # Calculate bias sparsity
    bias_sparsity = 0.0
    if bias_total_params > 0:
        bias_sparsity = 100.0 * bias_zero_params / bias_total_params
    bias_non_zero_params = bias_total_params - bias_zero_params

    # Calculate overall sparsity
    overall_sparsity = 100.0 * zero_params / total_params
    overall_non_zero_params = total_params - zero_params

    return (
        overall_sparsity,
        total_params,
        overall_non_zero_params,
        weight_sparsity,
        weight_total_params,
        weight_non_zero_params,
        bias_sparsity,
        bias_total_params,
        bias_non_zero_params,
    )


def load_whisper_model(
    model_name,
    device,
    pruning_config=None,
    quantization=None,
    use_fp16=False,
    make_permanent=True,
    pruned_model=None,
):
    """
    Load Whisper model and optionally apply pruning and/or quantization.

    Args:
        model_name: The Whisper model name or path
        device: Device to load the model to
        pruning_config: Dictionary mapping component types to pruning percentages, or None for no pruning
        quantization: Quantization method to apply, or None for no quantization
        use_fp16: Whether to use FP16 precision
        make_permanent: Whether to make pruning permanent
        pruned_model: Pre-pruned model to use as a base for quantization

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        quant_config = None
        is_immovable_quantization = False
        model = None

        # Only load a fresh model if we don't have a pruned model provided
        if pruned_model is None:
            # Check for quantization configs that need to be applied at load time
            if quantization and quantization.startswith("bnb_"):
                is_immovable_quantization = True
                if quantization == "bnb_fp4_32":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="fp4",
                        bnb_4bit_compute_dtype=torch.float32,
                    )
                elif quantization == "bnb_fp4_16":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="fp4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif quantization == "bnb_nf4_32":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float32,
                    )
                elif quantization == "bnb_nf4_16":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif quantization == "bnb_fp4_32_double":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="fp4",
                        bnb_4bit_compute_dtype=torch.float32,
                        bnb_4bit_use_double_quant=True,
                    )
                elif quantization == "bnb_fp4_16_double":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="fp4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif quantization == "bnb_nf4_32_double":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float32,
                        bnb_4bit_use_double_quant=True,
                    )
                elif quantization == "bnb_nf4_16_double":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
            elif quantization and quantization.startswith("hqq_"):
                is_immovable_quantization = True
                if quantization == "hqq_int3":
                    quant_config = HqqConfig(nbits=3)
                if quantization == "hqq_int4":
                    quant_config = HqqConfig(nbits=4)
                elif quantization == "hqq_int8":
                    quant_config = HqqConfig(nbits=8)

            # For quantization methods that can't be moved, use device_map="auto"
            if is_immovable_quantization:
                print(
                    f"Loading with device_map='auto' as {quantization} doesn't support moving models"
                )
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name, quantization_config=quant_config, device_map="auto"
                )
            else:
                # Load without device_map for other methods
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name, quantization_config=quant_config, device_map=None
                )

            # Apply pruning if specified
            if pruning_config is not None:
                print("Applying custom L1 unstructured pruning with config:")
                for component, percentage in pruning_config.items():
                    print(f"  {component}: {percentage}%")

                model = apply_custom_l1_pruning(
                    model, pruning_config, make_permanent=make_permanent
                )

                # Calculate and print sparsity
                sparsity_metrics = calculate_sparsity(model)
                overall_sparsity = sparsity_metrics[0]
                total_params = sparsity_metrics[1]
                overall_non_zero_params = sparsity_metrics[2]
                weight_sparsity = sparsity_metrics[3]
                weight_non_zero_params = sparsity_metrics[5]
                bias_sparsity = sparsity_metrics[6]
                bias_non_zero_params = sparsity_metrics[8]

                print(f"Overall model sparsity after pruning: {overall_sparsity:.2f}%")
                print(f"Total parameters: {total_params:,}")
                print(f"Non-zero parameters: {overall_non_zero_params:,}")
                print(f"Weight sparsity: {weight_sparsity:.2f}%")
                print(f"Bias sparsity: {bias_sparsity:.2f}%")
        else:
            # Use the provided pruned model
            print("Using provided pruned model as base for quantization")
            model = pruned_model

        # Apply non-loading-time quantization if specified
        if model is not None and not is_immovable_quantization and quantization:
            if quantization == "quanto_int4":
                print("Applying Quanto INT4 quantization")
                quantize(model, weights=qint4)
                freeze(model)
            elif quantization == "quanto_int8":
                print("Applying Quanto INT8 quantization")
                quantize(model, weights=qint8)
                freeze(model)
            elif quantization == "pytorch":
                print("Applying PyTorch dynamic quantization")
                torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )

        # Move model to device if needed and quantization allows
        if model is not None and not is_immovable_quantization:
            model = model.to(device)

        # Apply FP16 if requested and not quantized
        if model is not None and use_fp16 and quantization is None and torch.cuda.is_available():
            model = model.half()
            print("Converted model to FP16")

        if model is not None:
            model.config.forced_decoder_ids = None

        return model
    except Exception as e:
        print(f"Error in load_whisper_model: {e}")
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
        return {"error": str(e)}, None

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

    # Return only two values to match the unpacking in the main function
    return scores, result


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
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer, _use_new_zipfile_serialization=True
    )  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)


def main():
    # Configuration from the pruning code
    original_model_name = "openai/whisper-small"
    batch_size = 16
    save_path = COMBINED_RESULTS_DIR
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (
        not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    print(f"Using {device}")

    # Define custom pruning configuration with the specified percentages
    pruning_config = {
        "encoder_ffn": 50,
        "decoder_ffn_first": 25,
        "decoder_ffn_middle": 45,
        "decoder_ffn_last": 30,
        "encoder_self_attn": 40,
        "decoder_self_attn": 50,
        "decoder_cross_attn": 45,
        "layer_norm": 0,
        "token_embeddings": 25,
        "positional_embeddings": 0,
        "conv_layers": 30,
        "output_projection": 25,
        "all_bias": 0,
    }

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load full datasets (matching the quantization code)
    print("\nLoading datasets...")
    dataset_clean = load_librispeech(num_samples=2620, split="test.clean")  # Use full test.clean
    dataset_other = load_librispeech(num_samples=2939, split="test.other")  # Use full test.other

    print(f"Clean dataset: {len(dataset_clean)} samples")
    print(f"Other dataset: {len(dataset_other)} samples")

    # Process datasets
    print("\nProcessing datasets...")
    processed_test_data_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))
    processed_test_data_other = dataset_other.map(lambda x: map_to_feats(x, processor))

    # Calculate calibration sets (10%)
    n_calibration_clean = len(dataset_clean) // 10
    n_calibration_other = len(dataset_other) // 10

    # Split for quantization calibration
    calibration_data_clean = processed_test_data_clean.select(range(n_calibration_clean))
    test_data_clean = processed_test_data_clean  # Use full dataset for evaluation
    calibration_data_other = processed_test_data_other.select(range(n_calibration_other))
    test_data_other = processed_test_data_other  # Use full dataset for evaluation

    # Print dataset split sizes
    print("Dataset splits:")
    print(f"  Clean test set: {len(test_data_clean)} samples")
    print(f"  Clean calibration set: {len(calibration_data_clean)} samples")
    print(f"  Other test set: {len(test_data_other)} samples")
    print(f"  Other calibration set: {len(calibration_data_other)} samples")

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Define quantization configurations to apply to the pruned model
    quant_configs = {
        "baseline_pruned": {"quantization": None},
        "pruned_quanto_int4": {"quantization": "quanto_int4"},
        "pruned_quanto_int8": {"quantization": "quanto_int8"},
        "pruned_pytorch": {"quantization": "pytorch"},
        "pruned_hqq_int3": {"quantization": "hqq_int3"},
        "pruned_hqq_int4": {"quantization": "hqq_int4"},
        "pruned_hqq_int8": {"quantization": "hqq_int8"},
        "pruned_bnb_fp4_32": {"quantization": "bnb_fp4_32"},
        "pruned_bnb_nf4_32": {"quantization": "bnb_nf4_32"},
        "pruned_static_quanto_int4_int8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "int4",
            "act_quant": "int8",
        },
        "pruned_static_quanto_int8_int8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "int8",
            "act_quant": "int8",
        },
        "pruned_static_quanto_int4_float8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "int4",
            "act_quant": "float8",
        },
        "pruned_static_quanto_int8_float8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "int8",
            "act_quant": "float8",
        },
        "pruned_static_quanto_float8_int8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "float8",
            "act_quant": "int8",
        },
        "pruned_static_quanto_float8_float8": {
            "quantization": None,
            "static_quanto": True,
            "weight_quant": "float8",
            "act_quant": "float8",
        },
    }

    # First, create and save the pruned base model
    print("\n" + "=" * 50)
    print("Creating pruned base model")
    print("=" * 50)

    # Clear memory before loading
    clear_gpu_memory()

    # Create the pruned base model
    pruned_model = load_whisper_model(
        model_name=original_model_name,
        device=device,
        pruning_config=pruning_config,
        make_permanent=True,
    )

    # Calculate metrics for the pruned model
    (
        overall_sparsity,
        total_params,
        overall_non_zero_params,
        weight_sparsity,
        weight_total_params,
        weight_non_zero_params,
        bias_sparsity,
        bias_total_params,
        bias_non_zero_params,
    ) = calculate_sparsity(pruned_model)

    # Calculate GFLOPs for the pruned model
    pruned_gflops = calculate_model_gflops(pruned_model)

    # Calculate theoretical dense pruned size
    theoretical_dense_pruned_size = calculate_pruned_dense_size(pruned_model)

    # Save pruned model sparse representation
    pruned_sparse_model_path = os.path.join(PRUNED_MODEL_DIR, "pruned_sparse_model.pt")
    pruned_sparse_size = save_sparse_model(pruned_model, pruned_sparse_model_path)

    # Save original pruned model size
    pruned_model_size = get_model_disk_size_in_mb(pruned_model)

    # Evaluate base pruned model on both splits
    pruned_results = {}

    for split, dataset in [
        ("clean", test_data_clean),
        ("other", test_data_other),
    ]:
        print(f"\nEvaluating pruned base model on {split} split...")

        # Initialize memory tracker
        tracker = WhisperMemoryTracker(f"pruned_baseline_{split}", PRUNED_MODEL_DIR)

        try:
            # Run evaluation
            scores, result = evaluate_model(
                model=pruned_model,
                processor=processor,
                dataset=dataset,
                metrics=metrics,
                memory_tracker=tracker,
                batch_size=batch_size,
                split=split,
            )

            # Store results
            pruned_results[split] = {
                "metrics": scores,
                "model_size_mb": pruned_model_size,
                "sparse_model_size_mb": pruned_sparse_size,
                "theoretical_dense_pruned_size_mb": theoretical_dense_pruned_size,
                "gflops": pruned_gflops,
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(PRUNED_MODEL_DIR, f"pruned_baseline_{split}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(pruned_results[split], f, indent=2)

        except Exception as e:
            print(f"Error evaluating pruned base model on {split} split: {e}")

        finally:
            tracker.close()

    # Save the pruned model for later quantization experiments
    pruned_model_path = os.path.join(PRUNED_MODEL_DIR, "pruned_model.pt")
    torch.save(pruned_model.state_dict(), pruned_model_path)
    print(f"Saved pruned model to {pruned_model_path}")

    # Now apply each quantization technique to a fresh instance of the pruned model
    for model_name, config in quant_configs.items():
        print("\n" + "=" * 50)
        print(f"Evaluating {model_name}")
        print("=" * 50)

        # Clear memory
        clear_gpu_memory()

        try:
            # Load fresh pruned model from saved state dict
            fresh_pruned_model = WhisperForConditionalGeneration.from_pretrained(
                original_model_name, device_map=None
            )
            fresh_pruned_model.load_state_dict(torch.load(pruned_model_path))
            fresh_pruned_model = fresh_pruned_model.to(device)

            # Record quantization details for theoretical size calculation
            quant_type = None
            if config.get("quantization"):
                # Extract quantization method (int4/int8/etc.)
                if "quanto_" in config["quantization"] or "hqq_" in config["quantization"]:
                    quant_type = config["quantization"].split("_")[-1]
                elif "bnb_" in config["quantization"]:
                    # Extract fp4/nf4
                    parts = config["quantization"].split("_")
                    quant_type = parts[1]  # e.g., fp4 or nf4
            elif config.get("static_quanto"):
                quant_type = config.get("weight_quant", "fp32")

            # Calculate theoretical size with both pruning + quantization
            if quant_type:
                theoretical_combined_size = calculate_theoretical_quantized_size(
                    fresh_pruned_model, quant_type
                )
            else:
                theoretical_combined_size = theoretical_dense_pruned_size

            # Apply the quantization
            if config.get("static_quanto"):
                # Handle static quantization with calibration
                weight_quant = config.get("weight_quant", "int8")
                act_quant = config.get("act_quant", "int8")
                print(f"Applying static Quanto {weight_quant}+{act_quant} quantization")

                # Map quantization string to quanto objects
                quanto_map = {"int4": qint4, "int8": qint8, "float8": qfloat8}

                # Apply quantization
                quantize(
                    fresh_pruned_model,
                    weights=quanto_map.get(weight_quant, qint8),
                    activations=quanto_map.get(act_quant, qint8),
                )

                print("Calibrating on dataset...")
                calibration_memory_tracker = WhisperMemoryTracker(
                    f"{model_name}_calibration", save_path
                )

                # Run calibration
                with Calibration():
                    _ = evaluate_model(
                        model=fresh_pruned_model,
                        processor=processor,
                        dataset=calibration_data_clean,
                        metrics=metrics,
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split="calibration",
                    )

                # Freeze model after calibration
                freeze(fresh_pruned_model)
                calibration_memory_tracker.close()
            elif config.get("quantization"):
                # Handle non-static quantization methods in load_whisper_model
                fresh_pruned_model = load_whisper_model(
                    model_name=original_model_name,
                    device=device,
                    quantization=config["quantization"],
                    pruned_model=fresh_pruned_model,
                )

            # Get size of quantized model
            model_size = get_model_disk_size_in_mb(fresh_pruned_model)

            # Calculate GFLOPs for the quantized model (if possible)
            try:
                model_gflops = calculate_model_gflops(fresh_pruned_model)
            except Exception as e:
                print(f"Error calculating GFLOPs: {e}")
                model_gflops = -1

            # Evaluate on both splits
            for split, dataset in [
                ("clean", test_data_clean),
                ("other", test_data_other),
            ]:
                print(f"\nEvaluating {model_name} on {split} split...")

                # Initialize memory tracker
                tracker = WhisperMemoryTracker(f"{model_name}_{split}", save_path)

                try:
                    # Run evaluation
                    scores, transcriptions = evaluate_model(
                        model=fresh_pruned_model,
                        processor=processor,
                        dataset=dataset,
                        metrics=metrics,
                        memory_tracker=tracker,
                        batch_size=batch_size,
                        split=split,
                    )

                    # Store results
                    results[f"{model_name}_{split}"] = {
                        "metrics": scores,
                        "model_size_mb": model_size,
                        "theoretical_pruned_quantized_size_mb": theoretical_combined_size,
                        "model_type": model_name,
                        "pruning_config": pruning_config,
                        "quantization_config": config,
                        "gflops": model_gflops,
                        "overall_sparsity": overall_sparsity,
                        "weight_sparsity": weight_sparsity,
                        "bias_sparsity": bias_sparsity,
                        "total_parameters": total_params,
                        "non_zero_parameters": overall_non_zero_params,
                    }

                    # Save metrics
                    metrics_path = os.path.join(save_path, f"{model_name}_{split}_metrics.json")
                    with open(metrics_path, "w") as f:
                        json.dump(results[f"{model_name}_{split}"], f, indent=2)

                    # Save transcriptions
                    transcriptions_path = os.path.join(
                        save_path, f"{model_name}_{split}_transcriptions.json"
                    )
                    with open(transcriptions_path, "w") as f:
                        json.dump(transcriptions, f, indent=2)

                except Exception as e:
                    print(f"Error evaluating {model_name} on {split} split: {e}")
                    continue

                finally:
                    tracker.close()

            # Clean up model
            del fresh_pruned_model
            clear_gpu_memory()

        except Exception as e:
            print(f"Error setting up {model_name}: {e}")
            continue

    # Save combined results
    all_results_path = os.path.join(save_path, "all_pruned_quantized_results.json")
    with open(all_results_path, "w") as f:
        results_to_save = {"pruned_baseline": pruned_results, "pruned_quantized": results}
        json.dump(results_to_save, f, indent=2)

    # Print summary of all results
    print("\n" + "=" * 60)
    print("PRUNED AND QUANTIZED EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nPruned Baseline:")
    if "clean" in pruned_results:
        baseline = pruned_results["clean"]
        print(f"  WER: {baseline['metrics']['WER']:.4f}")
        print(f"  CER: {baseline['metrics'].get('CER', 0):.4f}")
        print(f"  RTF: {baseline['metrics']['RTF']:.4f}")
        print(f"  Overall Sparsity: {baseline['overall_sparsity']:.2f}%")
        print(f"  Original Size: {baseline['model_size_mb']:.2f} MB")
        print(f"  Sparse Size: {baseline['sparse_model_size_mb']:.2f} MB")
        print(
            f"  Theoretical Dense Pruned Size: {baseline['theoretical_dense_pruned_size_mb']:.2f} MB"
        )
        print(f"  GFLOPs: {baseline['gflops']:.4f}")

    print("\nPruned + Quantized Results:")
    for key, result in results.items():
        if "_clean" in key:
            model_type = key.replace("_clean", "")
            print(f"\n{model_type}:")

            # Calculate changes from pruned baseline
            baseline = pruned_results.get("clean", {})
            baseline_metrics = baseline.get("metrics", {})

            wer_change = "-"
            cer_change = "-"
            rtf_change = "-"
            size_change = "-"

            if "WER" in result["metrics"] and "WER" in baseline_metrics:
                wer_change = f"{(result['metrics']['WER'] - baseline_metrics['WER']):.2f} ({(result['metrics']['WER'] - baseline_metrics['WER']) / baseline_metrics['WER'] * 100:+.2f}%)"

            if "CER" in result["metrics"] and "CER" in baseline_metrics:
                cer_change = f"{(result['metrics']['CER'] - baseline_metrics['CER']):.2f} ({(result['metrics']['CER'] - baseline_metrics['CER']) / baseline_metrics['CER'] * 100:+.2f}%)"

            if "RTF" in result["metrics"] and "RTF" in baseline_metrics:
                rtf_change = f"{(result['metrics']['RTF'] - baseline_metrics['RTF']):.6f} ({(result['metrics']['RTF'] - baseline_metrics['RTF']) / baseline_metrics['RTF'] * 100:+.2f}%)"

            if "model_size_mb" in result and "model_size_mb" in baseline:
                size_change = f"{(result['model_size_mb'] - baseline['model_size_mb']):.2f} MB ({(result['model_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%)"

            print(f"  WER: {result['metrics']['WER']:.4f} (Δ {wer_change})")
            if "CER" in result["metrics"]:
                print(f"  CER: {result['metrics']['CER']:.4f} (Δ {cer_change})")
            print(f"  RTF: {result['metrics']['RTF']:.6f} (Δ {rtf_change})")
            print(f"  Model Size: {result['model_size_mb']:.2f} MB (Δ {size_change})")

            if "theoretical_pruned_quantized_size_mb" in result:
                theoretical_change = f"{(result['theoretical_pruned_quantized_size_mb'] - baseline['theoretical_dense_pruned_size_mb']):.2f} MB ({(result['theoretical_pruned_quantized_size_mb'] - baseline['theoretical_dense_pruned_size_mb']) / baseline['theoretical_dense_pruned_size_mb'] * 100:+.2f}%)"
                print(
                    f"  Theoretical Pruned+Quantized Size: {result['theoretical_pruned_quantized_size_mb']:.2f} MB (Δ {theoretical_change})"
                )

            if "gflops" in result and result["gflops"] > 0:
                gflops_change = f"{(result['gflops'] - baseline['gflops']):.4f} ({(result['gflops'] - baseline['gflops']) / baseline['gflops'] * 100:+.2f}%)"
                print(f"  GFLOPs: {result['gflops']:.4f} (Δ {gflops_change})")

    print("\nResults saved to:", save_path)
    print("Full pruned model information:", PRUNED_MODEL_DIR)


if __name__ == "__main__":
    main()
