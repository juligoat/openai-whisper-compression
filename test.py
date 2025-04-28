import argparse
import copy
import gc
import io
import json
import logging
import os
import time
from collections import defaultdict, deque

import datasets
import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set seaborn style
sns.set(style="whitegrid")

# Create results directories
RESULTS_DIR = "pruning/whisper_pruning_results"
COMPREHENSIVE_PRUNING_DIR = os.path.join(RESULTS_DIR, "comprehensive_pruning")
PLOTS_DIR = os.path.join(COMPREHENSIVE_PRUNING_DIR, "plots")
MODELS_DIR = os.path.join(COMPREHENSIVE_PRUNING_DIR, "models")

for directory in [RESULTS_DIR, COMPREHENSIVE_PRUNING_DIR, PLOTS_DIR, MODELS_DIR]:
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
            elif "decoder" in name:
                flops_by_type["decoder"] += non_zero_ops
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
    for component, flops in flops_by_type.items():
        gflops = flops / 1e9
        percentage = (flops / sum(flops_by_type.values())) * 100
        print(f"  {component}: {gflops:.4f} GFLOPs ({percentage:.1f}%)")

    if total_params > 0:
        print("\nParameter efficiency:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {non_zero_params:,}")
        print(f"  Overall sparsity: {100 * (1 - non_zero_params / total_params):.2f}%")

    print(f"\nTotal estimated GFLOPs: {total_gflops:.4f}")

    return total_gflops


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


def calculate_sparsity(model):
    """
    Calculate the sparsity percentage and parameter counts in the model.

    Args:
        model: The PyTorch model

    Returns:
        tuple: (sparsity percentage, total parameters, non-zero parameters)
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:  # Only consider weight parameters
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    if total_params == 0:
        return 0.0, 0, 0

    sparsity = 100.0 * zero_params / total_params
    non_zero_params = total_params - zero_params
    return sparsity, total_params, non_zero_params


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
        if param.dim() > 1 and torch.sum(param == 0) > 0.3 * param.numel():
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
            if total_elements > 100000:  # Only print for large layers
                print(
                    f"  Converting {name}: {sparsity:.1f}% sparse ({zero_elements}/{total_elements} zeros)"
                )
        else:
            # Keep as dense
            sparse_state_dict[name] = param
            sparse_params += param.numel() * 4  # 4 bytes per float32

    # Save sparse model
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


def map_to_feats(batch, processor):
    """
    Process a batch of audio samples to extract features.

    Args:
        batch: Batch of audio samples
        processor: Whisper processor

    Returns:
        Processed batch with input_features added
    """
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch


def transcribe_batch(batch, model, processor, memory_tracker, split, batch_idx):
    """
    Transcribe a batch of audio samples.

    Args:
        batch: Batch of audio samples with input_features
        model: Whisper model
        processor: Whisper processor
        memory_tracker: Memory tracker instance
        split: Dataset split name (clean or other)
        batch_idx: Batch index

    Returns:
        Batch with predictions added
    """
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
    """
    Evaluate a Whisper model on a dataset.

    Args:
        model: Whisper model
        processor: Whisper processor
        dataset: Dataset to evaluate on
        metrics: Dictionary of metrics to compute
        memory_tracker: Memory tracker instance
        split: Dataset split name (clean or other)
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (scores, result)
    """
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

    # Return scores and result
    return scores, result


def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    """
    Get the size of a model on disk in MB.

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer, _use_new_zipfile_serialization=True
    )  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)


# Implement various structured pruning methods


def attention_head_pruning(model, prune_ratio):
    """
    Prune attention heads based on their importance.

    Args:
        model: The WhisperForConditionalGeneration model
        prune_ratio: Percentage of heads to prune in each attention module

    Returns:
        Pruned model
    """
    logger.info(f"Applying attention head pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Find all attention modules
    attention_modules = []
    for name, module in pruned_model.named_modules():
        # Look for attention modules in encoder and decoder
        if ("encoder.layers" in name or "decoder.layers" in name) and name.endswith(
            ("self_attn", "encoder_attn")
        ):
            parent_name = name
            parent_module = module

            # Find the attention projections
            found_projections = 0
            for child_name, child_module in parent_module.named_children():
                if child_name in ["q_proj", "k_proj", "v_proj", "out_proj"] and isinstance(
                    child_module, nn.Linear
                ):
                    found_projections += 1

            # Only add if we found at least the q, k, v projections
            if found_projections >= 3:
                attention_modules.append((parent_name, parent_module))

    logger.info(f"Found {len(attention_modules)} attention modules")

    # Track statistics
    total_heads = 0
    pruned_heads = 0
    pruned_module_stats = {}

    for name, module in attention_modules:
        # Get the head dimension and number of heads
        try:
            # Attempt to get these directly from the module
            if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
                num_heads = module.num_heads
                head_dim = module.head_dim
            else:
                # Try to infer from the projection dimensions
                if hasattr(module, "q_proj"):
                    # Whisper uses 1280/80=16 heads in encoder and 1280/80=16 heads in decoder by default
                    embed_dim = module.q_proj.weight.shape[0]
                    if "encoder" in name:
                        num_heads = 8  # For whisper-small encoder
                    else:
                        num_heads = 8  # For whisper-small decoder
                    head_dim = embed_dim // num_heads
                else:
                    # Skip if we can't determine head information
                    logger.warning(f"Could not determine head information for {name}, skipping")
                    continue
        except Exception as e:
            logger.warning(f"Error determining head info for {name}: {e}")
            continue

        # Determine number of heads to prune
        heads_to_prune = max(1, int(num_heads * prune_ratio))

        # Ensure we don't prune all heads
        heads_to_prune = min(heads_to_prune, num_heads - 1)

        if heads_to_prune == 0:
            continue

        total_heads += num_heads

        # Calculate importance for each head
        head_importance = []

        # For each attention head, calculate its importance score
        for head_idx in range(num_heads):
            importance_score = 0

            # Calculate importance based on weight norms in q_proj, k_proj, v_proj
            for proj_name in ["q_proj", "k_proj", "v_proj"]:
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    if isinstance(proj, nn.Linear):
                        # Get the weights for this head
                        weight_slice = proj.weight[
                            head_idx * head_dim : (head_idx + 1) * head_dim, :
                        ]

                        # Add the Frobenius norm to the importance score
                        importance_score += torch.norm(weight_slice).item()

            head_importance.append((head_idx, importance_score))

        # Sort heads by importance (ascending)
        head_importance.sort(key=lambda x: x[1])

        # Get heads to prune (least important first)
        indices_to_prune = [idx for idx, _ in head_importance[:heads_to_prune]]

        # Prune each projection weight
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(module, proj_name):
                proj = getattr(module, proj_name)
                if isinstance(proj, nn.Linear):
                    # Create mask for the projection weights
                    weight = proj.weight
                    mask = torch.ones_like(weight)

                    # Mask out the weights for pruned heads
                    for head_idx in indices_to_prune:
                        mask[head_idx * head_dim : (head_idx + 1) * head_dim, :] = 0

                    # Apply mask
                    proj.weight.data *= mask

        # Also prune the output projection if it exists
        if hasattr(module, "out_proj"):
            out_proj = module.out_proj
            if isinstance(out_proj, nn.Linear):
                # Create mask for the output projection
                weight = out_proj.weight
                mask = torch.ones_like(weight)

                # Mask out the columns corresponding to pruned heads
                for head_idx in indices_to_prune:
                    mask[:, head_idx * head_dim : (head_idx + 1) * head_dim] = 0

                # Apply mask
                out_proj.weight.data *= mask

        pruned_heads += heads_to_prune
        pruned_module_stats[name] = (heads_to_prune, num_heads)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_heads / total_heads if total_heads > 0 else 0

    logger.info(f"Pruned {pruned_heads}/{total_heads} attention heads ({pruned_percentage:.2f}%)")
    logger.info(f"Pruned heads in {len(pruned_module_stats)} attention modules")

    # Log pruning details for modules
    if pruned_module_stats:
        logger.info("\nPruning details for attention modules:")
        for name, (pruned, total) in pruned_module_stats.items():
            logger.info(f"  {name}: pruned {pruned}/{total} heads ({100.0*pruned/total:.2f}%)")

    return pruned_model


def layer_type_selective_pruning(
    model, conv_ratio=0.01, ffn_ratio=0.01, attn_proj_ratio=0.005, embedding_ratio=0.0
):
    """
    Apply different pruning ratios to different layer types.

    Args:
        model: The WhisperForConditionalGeneration model
        conv_ratio: Pruning ratio for convolutional layers
        ffn_ratio: Pruning ratio for feed-forward networks
        attn_proj_ratio: Pruning ratio for attention projections
        embedding_ratio: Pruning ratio for embedding layers

    Returns:
        Pruned model
    """
    logger.info("Applying layer-type selective pruning:")
    logger.info(f"  Conv layers: {conv_ratio*100:.1f}%")
    logger.info(f"  FFN layers: {ffn_ratio*100:.1f}%")
    logger.info(f"  Attention projections: {attn_proj_ratio*100:.1f}%")
    logger.info(f"  Embedding layers: {embedding_ratio*100:.1f}%")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Determine layer type and corresponding pruning ratio
        layer_type = None
        prune_ratio = None

        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            layer_type = "conv"
            prune_ratio = conv_ratio
        elif isinstance(module, nn.Linear):
            if "embed" in name:
                layer_type = "embedding"
                prune_ratio = embedding_ratio
            elif "fc1" in name or "fc2" in name:
                layer_type = "ffn"
                prune_ratio = ffn_ratio
            elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                layer_type = "attention"
                prune_ratio = attn_proj_ratio
            else:
                layer_type = "other_linear"
                prune_ratio = ffn_ratio  # Use FFN ratio as default

        if layer_type is None or prune_ratio is None or prune_ratio <= 0:
            continue

        # Apply structured pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        # Calculate L2 norm for each output filter/neuron
        if weight.dim() > 1:
            l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)

            # Get indices of filters with smallest L2 norms
            indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

            # Create a mask to zero out pruned filters/neurons
            mask = torch.ones_like(weight)
            mask[indices_to_prune] = 0

            # Apply mask
            module.weight.data *= mask
            if module.bias is not None:
                module.bias.data[indices_to_prune] = 0

            pruned_filters += num_to_prune
            pruned_module_stats[name] = (num_to_prune, out_dim, layer_type, prune_ratio)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules with type-specific ratios")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    # Log pruning details by layer type
    layer_type_stats = defaultdict(lambda: {"pruned": 0, "total": 0})
    for name, (pruned, total, layer_type, _) in pruned_module_stats.items():
        layer_type_stats[layer_type]["pruned"] += pruned
        layer_type_stats[layer_type]["total"] += total

    logger.info("\nPruning statistics by layer type:")
    for layer_type, stats in layer_type_stats.items():
        pruned = stats["pruned"]
        total = stats["total"]
        logger.info(
            f"  {layer_type}: pruned {pruned}/{total} filters/neurons ({100.0*pruned/total:.2f}%)"
        )

    return pruned_model


def encoder_decoder_critical_path_pruning(
    model,
    encoder_ratio=0.01,
    decoder_self_attn_ratio=0.005,
    decoder_cross_attn_ratio=0.001,
    decoder_ffn_ratio=0.01,
):
    """
    Prune encoder and decoder with different ratios, preserving critical path components.
    The cross-attention is the most critical component connecting encoder and decoder.

    Args:
        model: The WhisperForConditionalGeneration model
        encoder_ratio: Pruning ratio for encoder layers
        decoder_self_attn_ratio: Pruning ratio for decoder self-attention
        decoder_cross_attn_ratio: Pruning ratio for decoder cross-attention (lowest as it's critical)
        decoder_ffn_ratio: Pruning ratio for decoder feed-forward networks

    Returns:
        Pruned model
    """
    logger.info("Applying encoder-decoder critical path pruning:")
    logger.info(f"  Encoder layers: {encoder_ratio*100:.1f}%")
    logger.info(f"  Decoder self-attention: {decoder_self_attn_ratio*100:.1f}%")
    logger.info(f"  Decoder cross-attention: {decoder_cross_attn_ratio*100:.1f}%")
    logger.info(f"  Decoder feed-forward: {decoder_ffn_ratio*100:.1f}%")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Determine component type and corresponding pruning ratio
        component_type = None
        prune_ratio = None

        if "encoder" in name:
            component_type = "encoder"
            prune_ratio = encoder_ratio
        elif "decoder" in name:
            if "self_attn" in name:
                component_type = "decoder_self_attn"
                prune_ratio = decoder_self_attn_ratio
            elif "encoder_attn" in name:
                component_type = "decoder_cross_attn"
                prune_ratio = decoder_cross_attn_ratio
            elif "fc1" in name or "fc2" in name:
                component_type = "decoder_ffn"
                prune_ratio = decoder_ffn_ratio
            else:
                component_type = "decoder_other"
                prune_ratio = decoder_ffn_ratio  # Use FFN ratio as default

        if component_type is None or prune_ratio is None or prune_ratio <= 0:
            continue

        # Apply structured pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        # Calculate L2 norm for each output filter/neuron
        if weight.dim() > 1:
            l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)

            # Get indices of filters with smallest L2 norms
            indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

            # Create a mask to zero out pruned filters/neurons
            mask = torch.ones_like(weight)
            mask[indices_to_prune] = 0

            # Apply mask
            module.weight.data *= mask
            if module.bias is not None:
                module.bias.data[indices_to_prune] = 0

            pruned_filters += num_to_prune
            pruned_module_stats[name] = (num_to_prune, out_dim, component_type, prune_ratio)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules with component-specific ratios")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    # Log pruning details by component type
    component_type_stats = defaultdict(lambda: {"pruned": 0, "total": 0})
    for name, (pruned, total, component_type, _) in pruned_module_stats.items():
        component_type_stats[component_type]["pruned"] += pruned
        component_type_stats[component_type]["total"] += total

    logger.info("\nPruning statistics by component type:")
    for component_type, stats in component_type_stats.items():
        pruned = stats["pruned"]
        total = stats["total"]
        logger.info(
            f"  {component_type}: pruned {pruned}/{total} filters/neurons ({100.0*pruned/total:.2f}%)"
        )

    return pruned_model


def correlation_based_pruning(model, prune_ratio=0.01):
    """
    Prune filters that are highly correlated with others, suggesting redundancy.

    Args:
        model: The WhisperForConditionalGeneration model
        prune_ratio: Percentage of filters to prune in each layer based on correlation

    Returns:
        Pruned model
    """
    logger.info(f"Applying correlation-based pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Apply correlation-based pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # For very large layers, subsample to avoid memory issues
        max_filters_for_correlation = 512
        if out_dim > max_filters_for_correlation:
            logger.info(
                f"Layer {name} has {out_dim} filters, limiting correlation analysis to {max_filters_for_correlation}"
            )
            selected_indices = torch.randperm(out_dim)[:max_filters_for_correlation]
            weight_for_corr = weight[selected_indices].view(max_filters_for_correlation, -1)
            using_subset = True
        else:
            weight_for_corr = weight.view(out_dim, -1)
            using_subset = False

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        try:
            # Normalize each filter for correlation calculation
            normalized_filters = weight_for_corr / (
                torch.norm(weight_for_corr, dim=1, keepdim=True) + 1e-8
            )

            # Calculate correlation matrix
            correlation_matrix = torch.mm(normalized_filters, normalized_filters.t())

            # Set diagonal to 0 to ignore self-correlation
            correlation_matrix.fill_diagonal_(0)

            # For each filter, find its maximum correlation with any other filter
            if using_subset:
                # For subsampled cases, we need a different approach
                # Instead of correlation-based on the subset, use L2 norm on the full set
                l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)
                indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]
            else:
                # Calculate maximum correlation for each filter
                max_correlations, _ = torch.max(correlation_matrix.abs(), dim=1)

                # Get indices of filters with highest correlation (most redundant)
                indices_to_prune = torch.argsort(max_correlations, descending=True)[:num_to_prune]
        except Exception as e:
            logger.warning(f"Error calculating correlations for {name}: {e}")
            logger.warning("Falling back to L2 norm pruning")

            # Fall back to L2 norm pruning
            l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)
            indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def random_structured_pruning(model, prune_ratio=0.005):
    """
    Randomly prune filters as a baseline comparison.

    Args:
        model: The WhisperForConditionalGeneration model
        prune_ratio: Percentage of filters to prune randomly in each layer

    Returns:
        Pruned model
    """
    logger.info(f"Applying random structured pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Apply random pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        # Randomly select filters to prune
        indices_to_prune = torch.randperm(out_dim)[:num_to_prune]

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def l1_norm_structured_pruning(model, prune_ratio=0.005):
    """
    Apply L1-norm structured pruning to the model (very low ratio).

    Args:
        model: The Whisper model to prune
        prune_ratio: Percentage of filters to prune in each layer

    Returns:
        Pruned model
    """
    logger.info(f"Applying L1-norm structured pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Apply L1 norm pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        # Calculate L1 norm for each output filter/neuron
        l1_norms = torch.norm(weight.view(out_dim, -1), p=1, dim=1)

        # Get indices of filters with smallest L1 norms
        indices_to_prune = torch.argsort(l1_norms)[:num_to_prune]

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def l2_norm_structured_pruning(model, prune_ratio=0.005):
    """
    Apply L2-norm structured pruning to the model (very low ratio).

    Args:
        model: The Whisper model to prune
        prune_ratio: Percentage of filters to prune in each layer

    Returns:
        Pruned model
    """
    logger.info(f"Applying L2-norm structured pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Apply L2 norm pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        # Calculate L2 norm for each output filter/neuron
        l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)

        # Get indices of filters with smallest L2 norms
        indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def geometric_median_pruning(model, prune_ratio=0.005):
    """
    Prune filters based on geometric median in feature space.
    This approach identifies filters that are closest to the geometric median
    in the feature space, which are likely to be redundant.

    Args:
        model: The WhisperForConditionalGeneration model
        prune_ratio: Percentage of filters to prune in each layer

    Returns:
        Pruned model
    """
    logger.info(f"Applying geometric median pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules
    for name, module in pruned_model.named_modules():
        if (
            not isinstance(module, nn.Linear)
            and not isinstance(module, nn.Conv1d)
            and not isinstance(module, nn.Conv2d)
        ):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        # Skip modules with only one output dimension
        if module.weight.shape[0] <= 1:
            continue

        # Apply geometric median pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Calculate the number of filters/neurons to prune
        num_to_prune = int(out_dim * prune_ratio)
        if num_to_prune == 0:
            continue

        try:
            # Reshape weights for distance calculation
            reshaped_weights = weight.view(out_dim, -1)

            # For large layers, use approximate method to avoid memory issues
            if out_dim > 512:
                # Fall back to L2 norm for large layers
                l2_norms = torch.norm(reshaped_weights, p=2, dim=1)
                indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]
            else:
                # Calculate pairwise distances
                distances = torch.zeros(out_dim, out_dim, device=weight.device)
                for i in range(out_dim):
                    distances[i] = torch.norm(
                        reshaped_weights - reshaped_weights[i].unsqueeze(0), p=2, dim=1
                    )

                # For each filter, sum its distances to all other filters
                distance_sums = torch.sum(distances, dim=1)

                # Filters with smallest distance sum are closer to the "center" (geometric median)
                # These are likely to be most redundant
                indices_to_prune = torch.argsort(distance_sums)[:num_to_prune]
        except Exception as e:
            logger.warning(f"Error in geometric median calculation for {name}: {e}")
            # Fall back to L2 norm pruning
            l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)
            indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def every_other_filter_pruning(model):
    """
    Prune every other filter in certain layer types.
    This approach is extremely simple but can sometimes work well.

    Args:
        model: The WhisperForConditionalGeneration model

    Returns:
        Pruned model
    """
    logger.info("Applying every-other-filter pruning to non-critical layers")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Process all modules, focusing only on certain layer types
    for name, module in pruned_model.named_modules():
        # Only apply to specific layer types
        if not isinstance(module, nn.Linear) or not hasattr(module, "weight"):
            continue

        # Skip cross-attention and critical output layers
        if "encoder_attn" in name or "embed" in name or "output" in name:
            continue

        # Only target feed-forward networks in the encoder
        if not ("encoder" in name and ("fc1" in name or "fc2" in name)):
            continue

        # Apply pruning to this layer
        weight = module.weight.data
        out_dim = weight.shape[0]
        total_filters += out_dim

        # Skip if layer is too small
        if out_dim < 4:
            continue

        # Select every other filter to prune (even indices)
        indices_to_prune = torch.tensor([i for i in range(0, out_dim, 2)], device=weight.device)
        num_to_prune = len(indices_to_prune)

        # Create a mask to zero out pruned filters/neurons
        mask = torch.ones_like(weight)
        mask[indices_to_prune] = 0

        # Apply mask
        module.weight.data *= mask
        if module.bias is not None:
            module.bias.data[indices_to_prune] = 0

        pruned_filters += num_to_prune
        pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(
        f"Pruned {pruned_filters}/{total_filters} filters/neurons ({pruned_percentage:.2f}%)"
    )
    logger.info(f"Pruned {len(pruned_module_stats)} modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def conv_frontend_pruning(model, prune_ratio=0.01):
    """
    Apply pruning specifically focused on Whisper's convolutional frontend.
    The frontend is less sensitive than the Transformer layers.

    Args:
        model: The WhisperForConditionalGeneration model
        prune_ratio: Percentage of filters to prune in conv layers

    Returns:
        Pruned model
    """
    logger.info(f"Applying convolutional frontend pruning with {prune_ratio*100:.1f}% ratio")

    # Create a copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Track pruning statistics
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_filters = 0
    total_filters = 0
    pruned_module_stats = {}

    # Find and prune only convolutional layers in the encoder frontend
    for name, module in pruned_model.named_modules():
        # Target only convolutional layers in encoder
        if "encoder" in name and isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Skip modules with only one output dimension
            if module.weight.shape[0] <= 1:
                continue

            # Apply pruning to this layer
            weight = module.weight.data
            out_dim = weight.shape[0]
            total_filters += out_dim

            # Calculate the number of filters to prune
            num_to_prune = int(out_dim * prune_ratio)
            if num_to_prune == 0:
                continue

            # Calculate L2 norm for each filter
            l2_norms = torch.norm(weight.view(out_dim, -1), p=2, dim=1)

            # Get indices of filters with smallest L2 norms
            indices_to_prune = torch.argsort(l2_norms)[:num_to_prune]

            # Create a mask to zero out pruned filters
            mask = torch.ones_like(weight)
            mask[indices_to_prune] = 0

            # Apply mask
            module.weight.data *= mask
            if module.bias is not None:
                module.bias.data[indices_to_prune] = 0

            pruned_filters += num_to_prune
            pruned_module_stats[name] = (num_to_prune, out_dim)

    # Calculate statistics
    pruned_percentage = 100.0 * pruned_filters / total_filters if total_filters > 0 else 0
    total_params_after = sum(p.nelement() for p in pruned_model.parameters())
    params_pruned_percentage = (
        100.0 * (total_params_before - total_params_after) / total_params_before
    )

    logger.info(f"Pruned {pruned_filters}/{total_filters} conv filters ({pruned_percentage:.2f}%)")
    logger.info(f"Pruned {len(pruned_module_stats)} convolutional modules")
    logger.info(f"Parameters before: {total_params_before:,}")
    logger.info(f"Parameters after: {total_params_after:,}")
    logger.info(f"Parameter reduction: {params_pruned_percentage:.2f}%")

    return pruned_model


def load_pruned_whisper_model(model_name, device, pruning_method):
    """
    Load and prune a Whisper model using the specified pruning method.

    Args:
        model_name: Name of the Whisper model to load
        device: Device to load the model on
        pruning_method: Dictionary with pruning method and parameters

    Returns:
        Loaded and pruned model
    """
    # Load model
    logger.info(f"Loading Whisper model: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

    # Apply pruning if specified
    if pruning_method is not None:
        method = pruning_method.get("method")

        if method == "attention_head":
            prune_ratio = pruning_method.get("prune_ratio", 0.01)
            model = attention_head_pruning(model, prune_ratio)

        elif method == "layer_type":
            conv_ratio = pruning_method.get("conv_ratio", 0.01)
            ffn_ratio = pruning_method.get("ffn_ratio", 0.01)
            attn_proj_ratio = pruning_method.get("attn_proj_ratio", 0.005)
            embedding_ratio = pruning_method.get("embedding_ratio", 0.0)
            model = layer_type_selective_pruning(
                model, conv_ratio, ffn_ratio, attn_proj_ratio, embedding_ratio
            )

        elif method == "critical_path":
            encoder_ratio = pruning_method.get("encoder_ratio", 0.01)
            decoder_self_attn_ratio = pruning_method.get("decoder_self_attn_ratio", 0.005)
            decoder_cross_attn_ratio = pruning_method.get("decoder_cross_attn_ratio", 0.001)
            decoder_ffn_ratio = pruning_method.get("decoder_ffn_ratio", 0.01)
            model = encoder_decoder_critical_path_pruning(
                model,
                encoder_ratio,
                decoder_self_attn_ratio,
                decoder_cross_attn_ratio,
                decoder_ffn_ratio,
            )

        elif method == "correlation":
            prune_ratio = pruning_method.get("prune_ratio", 0.01)
            model = correlation_based_pruning(model, prune_ratio)

        elif method == "random":
            prune_ratio = pruning_method.get("prune_ratio", 0.005)
            model = random_structured_pruning(model, prune_ratio)

        elif method == "l1_norm":
            prune_ratio = pruning_method.get("prune_ratio", 0.005)
            model = l1_norm_structured_pruning(model, prune_ratio)

        elif method == "l2_norm":
            prune_ratio = pruning_method.get("prune_ratio", 0.005)
            model = l2_norm_structured_pruning(model, prune_ratio)

        elif method == "geometric_median":
            prune_ratio = pruning_method.get("prune_ratio", 0.005)
            model = geometric_median_pruning(model, prune_ratio)

        elif method == "conv_frontend":
            prune_ratio = pruning_method.get("prune_ratio", 0.01)
            model = conv_frontend_pruning(model, prune_ratio)

        elif method == "every_other":
            model = every_other_filter_pruning(model)

        else:
            logger.warning(f"Unknown pruning method: {method}, not applying pruning")

        # Calculate and print sparsity
        sparsity, total_params, non_zero_params = calculate_sparsity(model)
        logger.info(f"Model sparsity after pruning: {sparsity:.2f}%")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Non-zero parameters: {non_zero_params:,}")

    # Move model to device
    model = model.to(device)

    # Set generation config
    model.config.forced_decoder_ids = None

    return model


def main():
    """Main function for running comprehensive structured pruning experiments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Structured Pruning for Whisper Models"
    )

    parser.add_argument(
        "--model", type=str, default="openai/whisper-small", help="Whisper model to use"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, cpu, or mps)"
    )
    parser.add_argument("--save-models", action="store_true", help="Save pruned models")
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to use for evaluation"
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(args.model)

    # Define pruning methods to test
    pruning_methods = [
        {"name": "baseline", "method": None},
        {"name": "attention_head_0.5pct", "method": "attention_head", "prune_ratio": 0.005},
        {
            "name": "layer_type_selective",
            "method": "layer_type",
            "conv_ratio": 0.01,
            "ffn_ratio": 0.01,
            "attn_proj_ratio": 0.005,
            "embedding_ratio": 0.0,
        },
        {
            "name": "critical_path",
            "method": "critical_path",
            "encoder_ratio": 0.01,
            "decoder_self_attn_ratio": 0.005,
            "decoder_cross_attn_ratio": 0.001,
            "decoder_ffn_ratio": 0.01,
        },
        {"name": "correlation_0.5pct", "method": "correlation", "prune_ratio": 0.005},
        {"name": "random_0.5pct", "method": "random", "prune_ratio": 0.005},
        {"name": "l1_norm_0.5pct", "method": "l1_norm", "prune_ratio": 0.005},
        {"name": "l2_norm_0.5pct", "method": "l2_norm", "prune_ratio": 0.005},
        {"name": "geometric_median_0.5pct", "method": "geometric_median", "prune_ratio": 0.005},
        {"name": "conv_frontend_1pct", "method": "conv_frontend", "prune_ratio": 0.01},
        {"name": "every_other", "method": "every_other"},
    ]

    # Load datasets
    logger.info("Loading LibriSpeech datasets...")
    dataset_clean = load_librispeech(num_samples=args.num_samples, split="test.clean")
    dataset_other = load_librispeech(num_samples=args.num_samples, split="test.other")

    # Process datasets
    logger.info("Processing datasets...")
    processed_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))
    processed_other = dataset_other.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Test each pruning method
    for pruning_config in pruning_methods:
        method_name = pruning_config["name"]
        logger.info(f"\n===== Evaluating method: {method_name} =====")

        # Clear GPU memory before loading
        clear_gpu_memory()

        try:
            # Load and prune model
            model = load_pruned_whisper_model(
                model_name=args.model,
                device=device,
                pruning_method=pruning_config if method_name != "baseline" else None,
            )

            # Calculate GFLOPs
            gflops = calculate_model_gflops(model)
            logger.info(f"Estimated model complexity: {gflops:.4f} GFLOPs")

            # Evaluate on both splits
            for split, dataset in [("clean", processed_clean), ("other", processed_other)]:
                logger.info(f"Evaluating on {split} split...")

                # Initialize memory tracker
                tracker = WhisperMemoryTracker(f"{method_name}_{split}", COMPREHENSIVE_PRUNING_DIR)

                try:
                    # Evaluate model
                    scores, result = evaluate_model(
                        model=model,
                        processor=processor,
                        dataset=dataset,
                        metrics=metrics,
                        memory_tracker=tracker,
                        batch_size=args.batch_size,
                        split=split,
                    )

                    # Get model size
                    model_size = get_model_disk_size_in_mb(model)

                    # Calculate theoretical pruned size
                    if method_name != "baseline":
                        theoretical_pruned_size = calculate_pruned_dense_size(model)
                    else:
                        theoretical_pruned_size = model_size

                    # Calculate sparsity
                    sparsity, total_params, non_zero_params = calculate_sparsity(model)

                    # Save results
                    results[f"{method_name}_{split}"] = {
                        "metrics": scores,
                        "model_size_mb": model_size,
                        "theoretical_pruned_size_mb": theoretical_pruned_size,
                        "gflops": gflops,
                        "sparsity": sparsity,
                        "total_parameters": total_params,
                        "non_zero_parameters": non_zero_params,
                        "pruning_method": pruning_config,
                    }

                    # Save metrics to file
                    metrics_path = os.path.join(
                        COMPREHENSIVE_PRUNING_DIR, f"{method_name}_{split}_summary.json"
                    )
                    with open(metrics_path, "w") as f:
                        json.dump(results[f"{method_name}_{split}"], f, indent=2)
                    logger.info(f"Saved metrics to {metrics_path}")

                except Exception as e:
                    logger.error(f"Error evaluating on {split} split: {e}")
                    logger.error(f"Traceback: {e.__traceback__}")
                    continue

                finally:
                    # Close tracker
                    tracker.close()

            # Save pruned model if requested
            if args.save_models and method_name != "baseline":
                model_dir = os.path.join(MODELS_DIR, method_name)
                os.makedirs(model_dir, exist_ok=True)
                model.save_pretrained(model_dir)
                processor.save_pretrained(model_dir)
                logger.info(f"Saved model to {model_dir}")

                # Try saving as sparse model for better storage
                sparse_model_path = os.path.join(model_dir, "sparse_model.pt")
                sparse_size = save_sparse_model(model, sparse_model_path)
                logger.info(f"Saved sparse model with size {sparse_size:.2f} MB")

            # Clean up
            del model
            clear_gpu_memory()

        except Exception as e:
            logger.error(f"Error processing {method_name}: {e}")
            logger.error(f"Traceback: {e.__traceback__}")
            continue

    # Save all results to a single file
    all_results_path = os.path.join(COMPREHENSIVE_PRUNING_DIR, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"All results saved to {all_results_path}")

    # Create comparative plots
    logger.info("Creating comparative plots...")

    # Extract baseline results
    baseline_clean = results.get("baseline_clean", {})
    baseline_other = results.get("baseline_other", {})

    # Extract baseline metrics
    baseline_clean_wer = baseline_clean.get("metrics", {}).get("WER", 0)
    baseline_clean_rtf = baseline_clean.get("metrics", {}).get("RTF", 0)
    baseline_other_wer = baseline_other.get("metrics", {}).get("WER", 0)
    baseline_other_rtf = baseline_other.get("metrics", {}).get("RTF", 0)

    # Create comparison table
    comparison_table = {
        "method": [],
        "clean_wer": [],
        "clean_wer_change": [],
        "clean_rtf": [],
        "clean_rtf_change": [],
        "other_wer": [],
        "other_wer_change": [],
        "other_rtf": [],
        "other_rtf_change": [],
        "sparsity": [],
        "params_reduction": [],
    }

    # Add data for each method
    for method_name in [method["name"] for method in pruning_methods]:
        # Get results for clean and other splits
        clean_result = results.get(f"{method_name}_clean", {})
        other_result = results.get(f"{method_name}_other", {})

        if not clean_result or not other_result:
            continue

        # Extract metrics
        clean_wer = clean_result.get("metrics", {}).get("WER", 0)
        clean_rtf = clean_result.get("metrics", {}).get("RTF", 0)
        other_wer = other_result.get("metrics", {}).get("WER", 0)
        other_rtf = other_result.get("metrics", {}).get("RTF", 0)

        # Calculate changes from baseline
        clean_wer_change = clean_wer - baseline_clean_wer
        clean_rtf_change = (clean_rtf - baseline_clean_rtf) / baseline_clean_rtf * 100
        other_wer_change = other_wer - baseline_other_wer
        other_rtf_change = (other_rtf - baseline_other_rtf) / baseline_other_rtf * 100

        # Calculate sparsity and parameter reduction
        sparsity = clean_result.get("sparsity", 0)
        total_params = clean_result.get("total_parameters", 0)
        non_zero_params = clean_result.get("non_zero_parameters", 0)
        params_reduction = (
            (total_params - non_zero_params) / total_params * 100 if total_params > 0 else 0
        )

        # Add to comparison table
        comparison_table["method"].append(method_name)
        comparison_table["clean_wer"].append(clean_wer)
        comparison_table["clean_wer_change"].append(clean_wer_change)
        comparison_table["clean_rtf"].append(clean_rtf)
        comparison_table["clean_rtf_change"].append(clean_rtf_change)
        comparison_table["other_wer"].append(other_wer)
        comparison_table["other_wer_change"].append(other_wer_change)
        comparison_table["other_rtf"].append(other_rtf)
        comparison_table["other_rtf_change"].append(other_rtf_change)
        comparison_table["sparsity"].append(sparsity)
        comparison_table["params_reduction"].append(params_reduction)

    # Print comparison table
    logger.info("\n===== RESULTS COMPARISON =====")
    logger.info(
        "Method                  | Clean WER (Δ)        | Other WER (Δ)        | Sparsity  | Param Reduction"
    )
    logger.info("-" * 100)

    for i in range(len(comparison_table["method"])):
        method = comparison_table["method"][i]
        clean_wer = comparison_table["clean_wer"][i]
        clean_wer_change = comparison_table["clean_wer_change"][i]
        other_wer = comparison_table["other_wer"][i]
        other_wer_change = comparison_table["other_wer_change"][i]
        sparsity = comparison_table["sparsity"][i]
        params_reduction = comparison_table["params_reduction"][i]

        logger.info(
            f"{method:24} | {clean_wer:.2f}% ({clean_wer_change:+.2f}%) | {other_wer:.2f}% ({other_wer_change:+.2f}%) | {sparsity:.2f}%  | {params_reduction:.2f}%"
        )

    # Sort methods by clean WER and print best performers
    logger.info("\n===== BEST PERFORMING METHODS =====")
    logger.info("Top 3 methods with lowest WER degradation:")

    # Convert to numpy arrays for easier sorting
    method_names = np.array(comparison_table["method"])
    clean_wer_changes = np.array(comparison_table["clean_wer_change"])

    # Sort by WER change (ascending)
    sorted_indices = np.argsort(clean_wer_changes)

    # Filter out baseline
    sorted_indices = [idx for idx in sorted_indices if method_names[idx] != "baseline"]

    # Print top 3 (or fewer if less available)
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        method = method_names[idx]
        clean_wer = comparison_table["clean_wer"][idx]
        clean_wer_change = comparison_table["clean_wer_change"][idx]
        sparsity = comparison_table["sparsity"][idx]

        logger.info(
            f"{i+1}. {method}: WER = {clean_wer:.2f}% (Δ{clean_wer_change:+.2f}%), Sparsity = {sparsity:.2f}%"
        )

    # Plot WER changes for all methods
    plt.figure(figsize=(12, 6))
    methods = [m for m in comparison_table["method"] if m != "baseline"]
    clean_wer_changes = [
        c
        for m, c in zip(comparison_table["method"], comparison_table["clean_wer_change"])
        if m != "baseline"
    ]

    plt.bar(methods, clean_wer_changes)
    plt.xlabel("Pruning Method")
    plt.ylabel("WER Change (%)")
    plt.title("WER Degradation by Pruning Method (Clean Split)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(PLOTS_DIR, "wer_change_comparison.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved comparison plot to {plot_path}")

    # Print final summary
    logger.info("\n===== SUMMARY =====")
    logger.info(f"Evaluated {len(pruning_methods)} pruning methods on Whisper {args.model}")
    logger.info(f"Results saved to {COMPREHENSIVE_PRUNING_DIR}")
    logger.info(f"Plots saved to {PLOTS_DIR}")

    if args.save_models:
        logger.info(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
