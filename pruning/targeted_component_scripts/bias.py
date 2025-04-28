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
L1_PRUNING_DIR = os.path.join(
    RESULTS_DIR,
    "l1_all_bias_pruning",  # Updated directory name to reflect all bias pruning
)
PLOTS_DIR = os.path.join(L1_PRUNING_DIR, "plots")
MODELS_DIR = os.path.join(L1_PRUNING_DIR, "models")

for directory in [RESULTS_DIR, L1_PRUNING_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


def calculate_pruned_dense_size(model, pruning_threshold=0.0):
    """
    Calculate the theoretical size of a dense model with pruned bias terms removed.
    This doesn't actually create the model, just calculates what the size would be.

    Args:
        model: The pruned model with masked biases
        pruning_threshold: Bias values with absolute value below this threshold are considered pruned

    Returns:
        float: Size in MB that a dense model with pruned bias terms removed would have
    """
    print("\n=== Calculating theoretical dense model size with pruned bias terms removed ===")

    total_params_original = 0
    total_params_pruned = 0
    total_bytes_original = 0
    total_bytes_dense_pruned = 0

    # Count parameters and calculate theoretical size
    for name, param in model.named_parameters():
        param_size_bytes = param.numel() * 4  # 4 bytes per float32
        total_params_original += param.numel()
        total_bytes_original += param_size_bytes

        if "bias" in name:  # Only consider bias parameters
            # Find pruned biases
            pruned_mask = torch.abs(param) <= pruning_threshold
            pruned_percentage = 100.0 * torch.sum(pruned_mask).item() / param.numel()

            # Track non-zero parameters
            non_zero_params = param.numel() - torch.sum(pruned_mask).item()
            total_params_pruned += non_zero_params

            # Calculate dense size without zeros
            param_dense_pruned_bytes = non_zero_params * 4  # Only non-zero elements at 4 bytes each
            total_bytes_dense_pruned += param_dense_pruned_bytes

            # For significant pruning, log details
            if (
                pruned_percentage > 5 and param.numel() > 1000
            ):  # Changed threshold for bias parameters
                print(f"Layer {name}: {pruned_percentage:.1f}% pruned")
                print(f"  Original: {param_size_bytes/1024/1024:.2f} MB")
                print(f"  Dense pruned: {param_dense_pruned_bytes/1024/1024:.2f} MB")
        else:
            # For non-bias parameters, size remains the same
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

            # Count bias parameters separately
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

                # For simplicity, we still include all weight ops since they dominate
                weight_flops = 2 * in_features * out_features

                # Categorize by location in model
                if "encoder" in name:
                    flops_by_type["encoder"] += weight_flops + bias_flops
                elif "decoder" in name:
                    flops_by_type["decoder"] += weight_flops + bias_flops
                else:
                    flops_by_type["other"] += weight_flops + bias_flops
            else:
                # No bias, just weight operations
                weight_flops = 2 * in_features * out_features

                # Categorize by location in model
                if "encoder" in name:
                    flops_by_type["encoder"] += weight_flops
                elif "decoder" in name:
                    flops_by_type["decoder"] += weight_flops
                else:
                    flops_by_type["other"] += weight_flops

            # Track weight parameter stats
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
    for component, flops in flops_by_type.items():
        gflops = flops / 1e9
        percentage = (flops / sum(flops_by_type.values())) * 100
        print(f"  {component}: {gflops:.4f} GFLOPs ({percentage:.1f}%)")

    if total_params > 0:
        print("\nParameter efficiency:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {non_zero_params:,}")
        print(f"  Weight sparsity: {100 * (1 - non_zero_params / total_params):.2f}%")

    if bias_total_params > 0:
        print("\nBias parameter efficiency:")
        print(f"  Total bias parameters: {bias_total_params:,}")
        print(f"  Non-zero bias parameters: {bias_non_zero_params:,}")
        print(f"  Bias sparsity: {100 * (1 - bias_non_zero_params / bias_total_params):.2f}%")

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

        # For bias parameters with significant sparsity, convert to sparse format
        if "bias" in name and torch.sum(param == 0) > 0.3 * param.numel():
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


def apply_l1_pruning(model, amount=0.3, target_patterns=None, make_permanent=False):
    """
    Apply L1 unstructured pruning to biases of a Whisper model.

    Args:
        model: The WhisperForConditionalGeneration model
        amount: Amount of bias values to prune (0.3 = 30%)
        target_patterns: List of strings to match bias layer names
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
    """
    # UPDATED: Target ALL bias parameters across the entire model
    if target_patterns is None:
        # Use a general pattern that will match all bias parameters
        target_patterns = ["bias"]  # This will match any parameter with "bias" in its name

    print("Will target ALL bias layers across the entire model for pruning")

    # Get parameters to prune based on pattern matching
    params_to_prune = []

    # Find modules containing bias parameters
    for name, module in model.named_modules():
        # Find modules that contain bias parameters
        if hasattr(module, "bias") and module.bias is not None:
            for param_name, param in module.named_parameters(recurse=False):
                if param_name == "bias":
                    # Get the full parameter name
                    full_param_name = f"{name}.{param_name}"
                    # Check if it matches any of our patterns
                    if any(pattern in full_param_name for pattern in target_patterns):
                        params_to_prune.append((module, "bias"))
                        break

    if not params_to_prune:
        print("Warning: No bias parameters found to prune! Check your target patterns.")
        return model

    print(
        f"Found {len(params_to_prune)} modules with bias to prune with L1 unstructured pruning, amount={amount}"
    )

    # Display a sample of modules being pruned (to avoid excessive logging)
    sample_size = min(10, len(params_to_prune))
    print(f"Sample of {sample_size} modules being pruned:")
    for i, (module, _) in enumerate(params_to_prune[:sample_size]):
        for name, mod in model.named_modules():
            if mod is module:
                print(f"  {i+1}. {name}.bias")
                break

    if len(params_to_prune) > sample_size:
        print(f"  ... and {len(params_to_prune) - sample_size} more")

    # Apply L1 unstructured pruning individually to each module to avoid errors
    success_count = 0
    for module, param_name in params_to_prune:
        try:
            # Store parameter size before pruning
            param = getattr(module, param_name)
            total = param.numel()
            non_zeros_before = torch.sum(param != 0).item()

            # Apply pruning to this specific module
            prune.l1_unstructured(module, param_name, amount=amount)

            # Calculate statistics after pruning
            pruned_param = getattr(module, param_name)
            non_zeros_after = torch.sum(pruned_param != 0).item()
            sparsity = 100.0 * (total - non_zeros_after) / total

            # Only log significant changes to avoid excessive output
            if total > 100 or (total - non_zeros_after) > 10:
                for name, mod in model.named_modules():
                    if mod is module:
                        print(
                            f"  Pruned {name}.{param_name}: {non_zeros_before} → {non_zeros_after} non-zeros ({sparsity:.2f}% sparse)"
                        )
                        break

            success_count += 1

            # Make pruning permanent if requested
            if make_permanent:
                prune.remove(module, param_name)

        except Exception as e:
            print(f"  Error pruning {param_name}: {e}")

    print(f"Successfully applied pruning to {success_count}/{len(params_to_prune)} modules")

    return model


def calculate_sparsity(model):
    """
    Calculate the sparsity percentage and parameter counts in the model's bias terms.

    Args:
        model: The PyTorch model

    Returns:
        tuple: (bias sparsity percentage, total bias parameters, non-zero bias parameters)
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "bias" in name:  # Only consider bias parameters
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    if total_params == 0:
        return 0.0, 0, 0

    sparsity = 100.0 * zero_params / total_params
    non_zero_params = total_params - zero_params
    return sparsity, total_params, non_zero_params


def load_whisper_model(model_name, device, pruning_amount=None, make_permanent=True):
    """
    Load Whisper model and optionally apply pruning to bias terms.

    Args:
        model_name: The Whisper model name
        device: Device to load the model to
        pruning_amount: Amount to prune (0.0 to 0.99) or None for no pruning
        make_permanent: Whether to make pruning permanent

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        # Load model without device_map
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_amount is not None and pruning_amount > 0:
            print(
                f"Applying L1 unstructured pruning to ALL bias terms with amount={pruning_amount}"
            )
            model = apply_l1_pruning(model, amount=pruning_amount, make_permanent=make_permanent)

            # Calculate and print sparsity
            sparsity, total_params, non_zero_params = calculate_sparsity(model)
            print(f"Bias term sparsity after pruning: {sparsity:.2f}%")
            print(f"Total bias parameters: {total_params:,}")
            print(f"Non-zero bias parameters: {non_zero_params:,}")

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

    # Prepare a small subset for testing first
    # This helps identify issues before processing the entire dataset
    test_subset = dataset.select(range(min(10, len(dataset))))
    print(f"Testing with a small subset of {len(test_subset)} samples first...")

    try:
        # Process the subset
        test_result = test_subset.map(process_batch, batched=True, batch_size=batch_size)
        print("Test subset processed successfully, continuing with full dataset...")

        # Reset counters
        total_processing_time = 0.0
        total_audio_duration = 0.0
        batch_counter = 0
        batch_rtfs = []
        batch_times = []

        # Process the full dataset
        start = time.time()
        result = dataset.map(process_batch, batched=True, batch_size=batch_size)
        end = time.time()
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        return {"error": str(e)}, None

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

    return scores, result


def load_librispeech(num_samples=None, split="test.clean"):
    """
    Load LibriSpeech clean/other data.
    """
    try:
        if num_samples is not None:
            # Stream partial dataset and convert to regular dataset
            print(f"Loading {num_samples} samples from LibriSpeech {split}...")
            # First try with streaming approach
            try:
                stream_dataset = datasets.load_dataset(
                    "librispeech_asr", split=split, streaming=True, trust_remote_code=True
                )
                dataset = datasets.Dataset.from_dict(
                    {
                        k: [sample[k] for sample in list(stream_dataset.take(num_samples))]
                        for k in next(iter(stream_dataset)).keys()
                    }
                )
            except Exception as e:
                print(f"Error with streaming approach: {e}")
                # Fallback to loading full dataset and taking a subset
                print("Falling back to loading a subset from the full dataset...")
                full_dataset = datasets.load_dataset("librispeech_asr", split=split)
                dataset = full_dataset.select(range(min(num_samples, len(full_dataset))))
        else:
            # Load full dataset
            print(f"Loading full LibriSpeech {split} dataset...")
            dataset = datasets.load_dataset("librispeech_asr", split=split)

        # Verify dataset loading succeeded
        if len(dataset) == 0:
            raise ValueError(f"Dataset {split} loaded but contains no samples")

        total_duration_seconds = sum(
            len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"] for sample in dataset
        )
        total_hours = total_duration_seconds / 3600

        print(f"Successfully loaded {len(dataset)} test samples from {split}")
        print(f"Total audio duration: {total_hours:.4f} hours")
        return dataset

    except Exception as e:
        print(f"Error loading LibriSpeech {split}: {e}")
        # Create a minimal dataset for testing if loading fails
        print("Creating minimal test dataset...")

        # Create a synthetic dataset with minimal audio
        sample_rate = 16000
        dummy_audio = {
            "array": np.zeros(sample_rate * 3, dtype=np.float32),
            "sampling_rate": sample_rate,
        }

        dummy_dataset = datasets.Dataset.from_dict(
            {
                "audio": [dummy_audio] * 10,
                "text": ["test"] * 10,
                "id": [f"test_{i}" for i in range(10)],
                "speaker_id": [1] * 10,
                "chapter_id": [1] * 10,
            }
        )

        print(f"Created minimal test dataset with {len(dummy_dataset)} samples")
        return dummy_dataset


def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer, _use_new_zipfile_serialization=True
    )  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)


def create_plots(results, metric_names, plot_dir):
    """
    Create plots of metrics vs pruning percentage.

    Args:
        results: Dictionary of results
        metric_names: List of metric names to plot
        plot_dir: Directory to save plots
    """
    print("\nGenerating plots...")

    # Extract pruning percentages and organize results
    pruning_percentages = []
    metrics_by_split = {"clean": {}, "other": {}}

    # Initialize metrics for each split
    for split in ["clean", "other"]:
        for metric in metric_names:
            metrics_by_split[split][metric] = []

    # Process results and gather data for plotting
    for model_name, model_results in results.items():
        if "baseline" in model_name:
            percent = 0
        else:
            # Extract percentage from model name (e.g., "l1_10")
            try:
                percent = int(model_name.split("_")[1])
            except (IndexError, ValueError):
                print(f"Skipping invalid model name: {model_name}")
                continue

        # Determine which split this model belongs to
        if "_clean" in model_name:
            split = "clean"
        elif "_other" in model_name:
            split = "other"
        else:
            print(f"Cannot determine split for model: {model_name}, skipping")
            continue

        # Add percent to pruning_percentages list if not already there
        if percent not in pruning_percentages:
            pruning_percentages.append(percent)

        # Check if metrics exist in the model results
        if "metrics" not in model_results:
            print(f"No metrics found for {model_name}, skipping")
            continue

        # Add data points for each metric
        for metric in metric_names:
            if metric in model_results["metrics"]:
                metrics_by_split[split][metric].append((percent, model_results["metrics"][metric]))
            else:
                print(f"Metric {metric} not found for {model_name}")

    # Sort pruning percentages
    pruning_percentages.sort()

    # Create individual plots for each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        legend_entries = []

        # Plot for both splits
        for split in ["clean", "other"]:
            # Sort data points by pruning percentage
            data_points = sorted(metrics_by_split[split][metric])

            # Skip if no data points for this metric and split
            if not data_points:
                print(f"No data points for {metric} in {split} split, skipping")
                continue

            x = [p for p, _ in data_points]
            y = [v for _, v in data_points]

            plt.plot(x, y, marker="o", label=f"{split} split")
            legend_entries.append(f"{split} split")

        # Add labels and title
        plt.xlabel("Bias Pruning Percentage (%)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs L1 Unstructured Bias Pruning Percentage (All Biases)")
        plt.grid(True)

        # Only add legend if we have plot lines
        if legend_entries:
            plt.legend()

        # Save the plot
        plot_path = os.path.join(plot_dir, f"{metric}_vs_pruning.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")

    # Create model size plot
    plt.figure(figsize=(10, 6))

    # Extract model sizes
    dense_sizes = []
    sparse_sizes = []
    theoretical_dense_pruned_sizes = []

    for percent in pruning_percentages:
        # Find the model result with this percentage (prefer clean split)
        model_name = f"l1_{percent}_clean" if percent > 0 else "baseline_clean"
        if model_name not in results and percent > 0:
            model_name = f"l1_{percent}_other"
        if model_name not in results and percent == 0:
            model_name = "baseline_other"

        if model_name in results:
            if "model_size_mb" in results[model_name]:
                dense_sizes.append((percent, results[model_name]["model_size_mb"]))
            if "sparse_model_size_mb" in results[model_name]:
                sparse_sizes.append((percent, results[model_name]["sparse_model_size_mb"]))
            if "theoretical_dense_pruned_size_mb" in results[model_name]:
                theoretical_dense_pruned_sizes.append(
                    (percent, results[model_name]["theoretical_dense_pruned_size_mb"])
                )

    # Sort by pruning percentage
    dense_sizes.sort(key=lambda x: x[0])
    sparse_sizes.sort(key=lambda x: x[0])
    theoretical_dense_pruned_sizes.sort(key=lambda x: x[0])

    # Plot model sizes
    if dense_sizes:
        plt.plot(
            [p for p, _ in dense_sizes],
            [s for _, s in dense_sizes],
            marker="o",
            label="Dense model size",
        )

    if sparse_sizes:
        plt.plot(
            [p for p, _ in sparse_sizes],
            [s for _, s in sparse_sizes],
            marker="s",
            label="Sparse model size",
        )

    if theoretical_dense_pruned_sizes:
        plt.plot(
            [p for p, _ in theoretical_dense_pruned_sizes],
            [s for _, s in theoretical_dense_pruned_sizes],
            marker="^",
            label="Theoretical dense pruned size",
        )

    plt.xlabel("Bias Pruning Percentage (%)")
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size vs L1 Unstructured Bias Pruning Percentage (All Biases)")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plot_dir, "model_size_vs_pruning.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Create GFLOPs plot
    plt.figure(figsize=(10, 6))

    # Extract GFLOPs data
    gflops_data = []

    for percent in pruning_percentages:
        # Find the model result with this percentage (prefer clean split)
        model_name = f"l1_{percent}_clean" if percent > 0 else "baseline_clean"
        if model_name not in results and percent > 0:
            model_name = f"l1_{percent}_other"
        if model_name not in results and percent == 0:
            model_name = "baseline_other"

        if model_name in results and "gflops" in results[model_name]:
            gflops_data.append((percent, results[model_name]["gflops"]))

    # Sort by pruning percentage
    gflops_data.sort(key=lambda x: x[0])

    # Plot GFLOPs
    if gflops_data:
        plt.plot(
            [p for p, _ in gflops_data], [g for _, g in gflops_data], marker="o", label="GFLOPs"
        )

    plt.xlabel("Bias Pruning Percentage (%)")
    plt.ylabel("GFLOPs")
    plt.title(
        "Computational Complexity (GFLOPs) vs L1 Unstructured Bias Pruning Percentage (All Biases)"
    )
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plot_dir, "gflops_vs_pruning.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Create parameter count plot
    plt.figure(figsize=(10, 6))

    # Extract parameter data
    total_params = []
    non_zero_params = []

    for percent in pruning_percentages:
        # Find the model result with this percentage (prefer clean split)
        model_name = f"l1_{percent}_clean" if percent > 0 else "baseline_clean"
        if model_name not in results and percent > 0:
            model_name = f"l1_{percent}_other"
        if model_name not in results and percent == 0:
            model_name = "baseline_other"

        if model_name in results:
            if "total_parameters" in results[model_name]:
                total_params.append((percent, results[model_name]["total_parameters"]))
            if "non_zero_parameters" in results[model_name]:
                non_zero_params.append((percent, results[model_name]["non_zero_parameters"]))

    # Sort by pruning percentage
    total_params.sort(key=lambda x: x[0])
    non_zero_params.sort(key=lambda x: x[0])

    # Plot parameter counts
    if total_params:
        plt.plot(
            [p for p, _ in total_params],
            [t / 1_000_000 for _, t in total_params],
            marker="o",
            label="Total bias parameters",
        )
    if non_zero_params:
        plt.plot(
            [p for p, _ in non_zero_params],
            [nz / 1_000_000 for _, nz in non_zero_params],
            marker="s",
            label="Non-zero bias parameters",
        )

    plt.xlabel("Bias Pruning Percentage (%)")
    plt.ylabel("Parameters (millions)")
    plt.title("Bias Parameter Count vs L1 Unstructured Bias Pruning Percentage (All Biases)")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plot_dir, "parameters_vs_pruning.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


def main():
    # Configuration to match the quantization code
    original_model_name = "openai/whisper-small"
    batch_size = 16  # Match the quantization code batch size
    save_path = L1_PRUNING_DIR
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (
        not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    print(f"Using {device}")

    # Define the pruning percentages to test (0% to 50% with intervals of 10%)
    pruning_percentages = [0, 10, 20, 30, 40, 50]

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    print("\nLoading smaller datasets for testing...")
    dataset_clean = load_librispeech(num_samples=2620, split="test.clean")  # Use full test.clean
    dataset_other = load_librispeech(num_samples=2939, split="test.other")  # Use full test.other

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

    # Define model configurations
    model_configs = {
        "baseline": {
            "pruning_amount": None  # No pruning for baseline
        }
    }

    # Add pruning configurations
    for percent in pruning_percentages:
        if percent > 0:  # Skip 0% as it's already in baseline
            model_configs[f"l1_{percent}"] = {
                "pruning_amount": percent / 100  # Convert percentage to fraction
            }

    # Evaluate each configuration
    for model_name, config in model_configs.items():
        print("\n" + "=" * 50)
        print(f"Evaluating {model_name}")
        print("=" * 50)

        # Clear memory before loading new model
        clear_gpu_memory()

        try:
            # Load and prune model
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                pruning_amount=config["pruning_amount"],
                make_permanent=True,
            )

            # Calculate actual sparsity and parameter counts
            sparsity, total_params, non_zero_params = calculate_sparsity(model)
            print(f"Actual bias sparsity: {sparsity:.2f}%")
            print(f"Total bias parameters: {total_params:,}")
            print(f"Non-zero bias parameters: {non_zero_params:,}")

            # Calculate model GFLOPs
            gflops = calculate_model_gflops(model)
            print(f"Estimated model complexity: {gflops:.4f} GFLOPs")

            # Evaluate on both splits
            for split, dataset in [
                ("clean", processed_test_data_clean),
                ("other", processed_test_data_other),
            ]:
                print(f"\nEvaluating on {split} split...")

                # Initialize memory tracker for this run
                tracker = WhisperMemoryTracker(f"{model_name}_{split}", save_path)

                try:
                    # Evaluate model performance
                    scores, result = evaluate_model(
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

                        # Calculate theoretical size of a dense model with pruned weights removed
                        theoretical_dense_pruned_size = 0.0
                        if config["pruning_amount"] is not None and config["pruning_amount"] > 0:
                            # Calculate what the size would be if we removed zeros (without creating the model)
                            theoretical_dense_pruned_size = calculate_pruned_dense_size(
                                model, pruning_threshold=0.0
                            )
                            print(
                                f"Theoretical dense pruned model size: {theoretical_dense_pruned_size:.2f} MB"
                            )

                        # Build results dictionary
                        results[f"{model_name}_{split}"] = {
                            "metrics": scores,
                            "model_size_mb": model_size,
                            "model_type": model_name,
                            "gflops": gflops,  # Add GFLOPs to results
                            "pruning_percentage": 0
                            if "baseline" in model_name
                            else int(model_name.split("_")[1]),
                            "actual_sparsity": sparsity,
                            "total_parameters": total_params,  # Add total bias parameter count
                            "non_zero_parameters": non_zero_params,  # Add non-zero bias parameter count
                            "theoretical_dense_pruned_size_mb": theoretical_dense_pruned_size,  # Add theoretical size
                        }

                        # Save metrics
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_summary.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[f"{model_name}_{split}"], f, indent=2)
                        print(f"Saved metrics to {metrics_path}")

                except Exception as e:
                    print(f"Error evaluating {model_name} on {split} split: {e!s}")
                    continue

                finally:
                    # Always close tracker and clear memory
                    tracker.close()

            # Save sparse model if this is a pruned model
            if "baseline" not in model_name:
                sparse_model_path = os.path.join(MODELS_DIR, f"{model_name}_sparse.pt")
                sparse_size = save_sparse_model(model, sparse_model_path)

                # Update results with sparse model size
                for split in ["clean", "other"]:
                    result_key = f"{model_name}_{split}"
                    if result_key in results:
                        results[result_key]["sparse_model_size_mb"] = sparse_size
                        if sparse_size > 0 and results[result_key]["model_size_mb"] > 0:
                            results[result_key]["size_reduction_percent"] = (
                                100.0
                                * (results[result_key]["model_size_mb"] - sparse_size)
                                / results[result_key]["model_size_mb"]
                            )

                        # Update the saved metrics file
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_summary.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[result_key], f, indent=2)

            # Clear model from memory
            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"Error setting up {model_name}: {e!s}")
            continue

    # Save all results to a single file
    all_results_path = os.path.join(L1_PRUNING_DIR, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {all_results_path}")

    # Use a safer list of metrics for plotting
    safe_metrics = []
    for metric in ["WER", "CER", "RTF", "avg_cpu_percent", "peak_ram_gb", "gflops"]:
        # Check if at least one result has this metric
        has_metric = False
        for model_result in results.values():
            if "metrics" in model_result and metric in model_result["metrics"]:
                has_metric = True
                break
        if has_metric:
            safe_metrics.append(metric)

    print(f"Plotting the following metrics: {safe_metrics}")

    # Create plots with the validated metrics
    create_plots(results=results, metric_names=safe_metrics, plot_dir=PLOTS_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("L1 UNSTRUCTURED BIAS PRUNING EXPERIMENT SUMMARY (ALL BIASES)")
    print("=" * 60)

    print("\nBaseline (0% pruning):")
    if "baseline_clean" in results:
        baseline = results["baseline_clean"]
        print(f"  WER: {baseline['metrics']['WER']:.4f}")
        print(f"  CER: {baseline['metrics']['CER']:.4f}")
        print(f"  RTF: {baseline['metrics']['RTF']:.4f}")
        print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
        print(f"  GFLOPs: {baseline['gflops']:.4f}")
        print(f"  Total Bias Parameters: {baseline['total_parameters']:,}")
        print(f"  Non-zero Bias Parameters: {baseline['non_zero_parameters']:,}")

    print("\nResults for different bias pruning percentages:")
    for percent in pruning_percentages:
        if percent == 0:
            continue

        model_key = f"l1_{percent}_clean"
        if model_key in results:
            result = results[model_key]

            # Calculate changes from baseline
            wer_change = "-"
            cer_change = "-"
            rtf_change = "-"
            size_change = "-"
            theoretical_size_change = "-"
            gflops_change = "-"
            param_change = "-"

            if "baseline_clean" in results:
                baseline = results["baseline_clean"]
                if "WER" in result["metrics"] and "WER" in baseline["metrics"]:
                    wer_change = f"{(result['metrics']['WER'] - baseline['metrics']['WER']) / baseline['metrics']['WER'] * 100:+.2f}%"
                if "CER" in result["metrics"] and "CER" in baseline["metrics"]:
                    cer_change = f"{(result['metrics']['CER'] - baseline['metrics']['CER']) / baseline['metrics']['CER'] * 100:+.2f}%"
                if "RTF" in result["metrics"] and "RTF" in baseline["metrics"]:
                    rtf_change = f"{(result['metrics']['RTF'] - baseline['metrics']['RTF']) / baseline['metrics']['RTF'] * 100:+.2f}%"

                if "sparse_model_size_mb" in result and "model_size_mb" in baseline:
                    size_change = f"{(result['sparse_model_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%"

                if (
                    "theoretical_dense_pruned_size_mb" in result
                    and result["theoretical_dense_pruned_size_mb"] > 0
                    and "model_size_mb" in baseline
                ):
                    theoretical_size_change = f"{(result['theoretical_dense_pruned_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%"

                if "gflops" in result and "gflops" in baseline:
                    gflops_change = f"{(result['gflops'] - baseline['gflops']) / baseline['gflops'] * 100:+.2f}%"

                if "non_zero_parameters" in result and "non_zero_parameters" in baseline:
                    param_change = f"{(result['non_zero_parameters'] - baseline['non_zero_parameters']) / baseline['non_zero_parameters'] * 100:+.2f}%"

            print(f"\n{percent}% bias pruning:")
            if "WER" in result["metrics"]:
                print(f"  WER: {result['metrics']['WER']:.4f} ({wer_change})")
            if "CER" in result["metrics"]:
                print(f"  CER: {result['metrics']['CER']:.4f} ({cer_change})")
            if "RTF" in result["metrics"]:
                print(f"  RTF: {result['metrics']['RTF']:.4f} ({rtf_change})")
            if "sparse_model_size_mb" in result:
                print(
                    f"  Sparse Model Size: {result['sparse_model_size_mb']:.2f} MB ({size_change})"
                )
            if (
                "theoretical_dense_pruned_size_mb" in result
                and result["theoretical_dense_pruned_size_mb"] > 0
            ):
                print(
                    f"  Theoretical Dense Pruned Size: {result['theoretical_dense_pruned_size_mb']:.2f} MB ({theoretical_size_change})"
                )
            if "gflops" in result:
                print(f"  GFLOPs: {result['gflops']:.4f} ({gflops_change})")
            if "actual_sparsity" in result:
                print(f"  Actual Bias Sparsity: {result['actual_sparsity']:.2f}%")
            if "total_parameters" in result:
                print(f"  Total Bias Parameters: {result['total_parameters']:,}")
            if "non_zero_parameters" in result:
                print(
                    f"  Non-zero Bias Parameters: {result['non_zero_parameters']:,} ({param_change})"
                )

    print("\nPlots saved to:", PLOTS_DIR)
    print("Sparse models saved to:", MODELS_DIR)
    print("Detailed metrics saved to:", L1_PRUNING_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in main execution: {e}")
        traceback.print_exc()
