import gc
import io
import json
import os
import time
from collections import deque
from datetime import datetime

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
RESULTS_DIR = "whisper_pruning_results"
SELECTIVE_PRUNING_DIR = os.path.join(RESULTS_DIR, "selective_pruning")
PLOTS_DIR = os.path.join(SELECTIVE_PRUNING_DIR, "plots")
MODELS_DIR = os.path.join(SELECTIVE_PRUNING_DIR, "models")

for directory in [RESULTS_DIR, SELECTIVE_PRUNING_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


class WhisperMemoryTracker:
    def __init__(self, model_name: str, save_path: str):
        self.model_name = model_name
        self.save_path = save_path
        self.peak_gpu_memory = 0
        self.peak_cpu_percent = 0
        self.memory_measurements = deque(maxlen=500)  # Store last 500 measurements
        self.start_time = time.time()
        self.process = psutil.Process()

        # Initialize GPU memory attributes even if running on CPU
        self.initial_gpu_memory = 0
        self.initial_gpu_cached = 0

        self.process.cpu_percent(interval=None)  # First call returns 0, discard it
        self.initial_cpu_percent = np.mean(
            [self.process.cpu_percent(interval=0.1) for _ in range(5)]
        )  # Stable avg
        self.initial_ram_usage = self.process.memory_info().rss / (1024**3)

        # Initialize GPU memory metrics if available
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            self.initial_gpu_cached = torch.cuda.memory_reserved() / (1024**3)

    def log_memory(self, split, batch_idx, batch_size, audio_duration):
        current_time = time.time()
        cpu_percent = np.mean(
            [self.process.cpu_percent(interval=0.1) for _ in range(3)]
        )  # Avg over 3 readings

        memory_data = {
            "timestamp": float(current_time - self.start_time),  # Ensure it's a native float
            "cpu_percent": float(cpu_percent),  # Ensure it's a native float
            "ram_gb": float(
                self.process.memory_info().rss / (1024**3)
            ),  # Ensure it's a native float
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

        # Append the memory measurement and explicitly make it a dict
        self.memory_measurements.append(dict(memory_data))
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)

    def get_memory_summary(self):
        """Get comprehensive memory usage statistics."""
        if not self.memory_measurements:
            return "No measurements recorded"

        summary = {
            "duration_seconds": time.time() - self.start_time,
            "cpu": {
                "initial_percent": self.initial_cpu_percent,
                "peak_percent": self.peak_cpu_percent,
                "initial_ram_gb": self.initial_ram_usage,
                "current_ram_gb": psutil.Process().memory_info().rss / (1024**3),
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

        return summary

    def save_metrics(self):
        """Save memory metrics to a JSON file."""
        metrics_path = os.path.join(self.save_path, f"{self.model_name}_memory_metrics.json")
        summary = self.get_memory_summary()

        # Convert deque to list for JSON serialization
        measurements_list = []
        for m in self.memory_measurements:
            # Create a copy of each measurement to avoid modifying the original
            measurement_copy = m.copy() if isinstance(m, dict) else m
            # Convert any non-serializable types
            if isinstance(measurement_copy, dict):
                # Convert timestamps to strings if they're datetime objects
                if "timestamp" in measurement_copy and isinstance(
                    measurement_copy["timestamp"], datetime
                ):
                    measurement_copy["timestamp"] = measurement_copy["timestamp"].isoformat()
            measurements_list.append(measurement_copy)

        # Create the output dictionary with serializable data
        output_data = {"summary": summary, "detailed_measurements": measurements_list}

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
                "error": "Full data couldn't be serialized to JSON",
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
        print(f"  Initial RAM: {summary['cpu']['initial_ram_gb']:.4f} GB")
        print(f"  Current RAM: {summary['cpu']['current_ram_gb']:.4f} GB")

        if "gpu" in summary:
            print("\nGPU Usage:")
            print(f"  Initial Allocated: {summary['gpu']['initial_allocated_gb']:.4f} GB")
            print(f"  Peak Allocated: {summary['gpu']['peak_allocated_gb']:.4f} GB")
            print(f"  Average Allocated: {summary['gpu']['average_allocated_gb']:.4f} GB")
            print(f"  Current Allocated: {summary['gpu']['current_allocated_gb']:.4f} GB")
            print(f"  Current Cached: {summary['gpu']['current_cached_gb']:.4f} GB")

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


def calculate_sparsity(model, submodule_filter=None):
    """
    Calculate the sparsity percentage in the model or specific submodules.

    Args:
        model: The PyTorch model
        submodule_filter: Optional list of submodule name parts to filter

    Returns:
        float: Percentage of zero weights in the model/submodules
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
        return 0.0

    sparsity = 100.0 * zero_params / total_params
    return sparsity


def calculate_component_sparsity(model):
    """Calculate sparsity separately for encoder and decoder components."""
    encoder_sparsity = calculate_sparsity(model, ["encoder"])
    decoder_sparsity = calculate_sparsity(model, ["decoder"])
    overall_sparsity = calculate_sparsity(model)

    return {
        "encoder_sparsity": encoder_sparsity,
        "decoder_sparsity": decoder_sparsity,
        "overall_sparsity": overall_sparsity,
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
            print(f"  - Overall: {sparsity_info['overall_sparsity']:.2f}%")

        # Move model to device
        model = model.to(device)
        model.config.forced_decoder_ids = None
        return model

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def clear_gpu_memory():
    """Clear cached GPU memory and reset peak memory stats if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()  # More aggressive cleanup with Python garbage collection
    else:
        print("Running on CPU - no GPU memory to clear")


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
        predicted_ids = model.generate(features)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all GPU ops complete before stopping the timer
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
    # Save per-sample RTF, processing time, and audio duration (same value repeated for all samples in the batch)
    batch["rtf"] = [batch_rtf] * len(batch["audio"])
    batch["processing_time"] = [processing_time] * len(batch["audio"])
    batch["audio_duration"] = [total_audio_duration] * len(batch["audio"])
    return batch


def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=16):
    total_processing_time = 0.0
    total_audio_duration = 0.0
    batch_counter = 0

    def process_batch(batch):
        nonlocal batch_counter, total_processing_time, total_audio_duration
        # Process the batch and update the cumulative totals
        result = transcribe_batch(batch, model, processor, memory_tracker, split, batch_counter)
        # Each sample in the batch has the same processing time and audio duration;
        # take the value from the first sample as representative.
        batch_processing_time = result["processing_time"][0]
        batch_audio_duration = result["audio_duration"][0]
        total_processing_time += batch_processing_time
        total_audio_duration += batch_audio_duration
        batch_counter += 1
        return result

    start = time.time()
    result = dataset.map(process_batch, batched=True, batch_size=batch_size)
    end = time.time()

    # Calculate overall RTF from the accumulated totals
    overall_rtf = total_processing_time / total_audio_duration
    print(f"Overall RTF: {overall_rtf:.6f}")

    # Compute metrics (e.g., WER, CER)
    scores = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["WER", "CER"]:
            score = 100 * metric.compute(
                references=result["reference"], predictions=result["prediction"]
            )
            scores[metric_name] = score
            print(f"{metric_name}: {score:.5f}")

    scores["RTF"] = overall_rtf
    scores["total_processing_time"] = total_processing_time
    scores["total_audio_duration"] = total_audio_duration
    scores["avg_latency"] = total_processing_time / batch_counter if batch_counter > 0 else 0

    # Record CPU metrics
    average_cpu_usage = (
        (
            sum([m["cpu_percent"] for m in memory_tracker.memory_measurements])
            / len(memory_tracker.memory_measurements)
        )
        if memory_tracker.memory_measurements
        else 0
    )
    peak_cpu_usage = memory_tracker.peak_cpu_percent
    scores["avg_cpu_percent"] = average_cpu_usage
    scores["peak_cpu_percent"] = peak_cpu_usage

    # Record Memory metrics
    summary = memory_tracker.get_memory_summary()
    scores["ram_usage_gb"] = summary["cpu"]["current_ram_gb"]

    if "gpu" in summary:
        scores["gpu_peak_allocated_gb"] = summary["gpu"]["peak_allocated_gb"]
        scores["gpu_average_allocated_gb"] = summary["gpu"]["average_allocated_gb"]

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
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer, _use_new_zipfile_serialization=True
    )  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)


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
            # Skip baselines
            if "baseline" in model_name:
                continue

            split = "clean" if "clean" in model_name else "other"
            config = model_name.replace(f"_{split}", "")

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

    # Create model size plot
    plt.figure(figsize=(10, 6))

    # Extract model sizes by configuration
    dense_sizes = {}
    sparse_sizes = {}

    for model_name, model_results in results.items():
        if "clean" in model_name:  # Just use clean split for sizes
            config = model_name.replace("_clean", "")
            dense_sizes[config] = model_results["model_size_mb"]
            if "sparse_model_size_mb" in model_results:
                sparse_sizes[config] = model_results["sparse_model_size_mb"]

    # Sort configs by name
    configs = sorted(dense_sizes.keys())
    dense_values = [dense_sizes[c] for c in configs]

    # Plot dense sizes
    plt.bar(configs, dense_values, label="Dense model size")

    # Plot sparse sizes if available
    if sparse_sizes:
        sparse_values = [sparse_sizes.get(c, 0) for c in configs]
        plt.bar(configs, sparse_values, label="Sparse model size", alpha=0.7)

    plt.xlabel("Pruning Configuration")
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size by Pruning Configuration")
    plt.grid(True, axis="y")
    plt.legend()
    plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(plot_dir, "model_size_by_config.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Create encoder vs decoder sparsity comparison if data available
    any_has_components = any("encoder_sparsity" in result for result in results.values())
    if any_has_components:
        compare_component_sizes(results)


def main():
    # Configuration
    original_model_name = "openai/whisper-small"
    batch_size = 16  # Match the quantization code batch size
    save_path = SELECTIVE_PRUNING_DIR
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")

    # Define the pruning configurations
    pruning_configs = {
        # Baseline (no pruning)
        "baseline": {"pruning_config": None},
        # Encoder-only pruning
        "encoder_only_30": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.3,
                "target_submodules": ["encoder"],
                "make_permanent": True,
            }
        },
        "encoder_only_50": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.5,
                "target_submodules": ["encoder"],
                "make_permanent": True,
            }
        },
        "encoder_only_70": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.7,
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
        "decoder_only_50": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.5,
                "target_submodules": ["decoder"],
                "make_permanent": True,
            }
        },
        "decoder_only_70": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.7,
                "target_submodules": ["decoder"],
                "make_permanent": True,
            }
        },
        # Layer-wise variable pruning (less aggressive in early layers)
        "layerwise_variable": {
            "pruning_config": {
                "method": "custom",  # Will handle this specially
                "make_permanent": True,
            }
        },
        # Magnitude threshold pruning
        "magnitude_threshold": {
            "pruning_config": {
                "method": "custom",  # Will handle this specially
                "make_permanent": True,
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
            # Handle custom pruning methods
            if model_name == "layerwise_variable":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Apply variable pruning by layer depth
                print("Applying layerwise variable pruning...")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Determine layer depth from name
                        if "encoder" in name:
                            # Extract layer number
                            if "layers." in name:
                                try:
                                    layer_str = name.split("layers.")[1].split(".")[0]
                                    layer_num = int(layer_str)
                                    # Early layers (0-2): 30%, Middle layers (3-5): 50%, Late layers (6+): 70%
                                    if layer_num < 3:
                                        amount = 0.3
                                    elif layer_num < 6:
                                        amount = 0.5
                                    else:
                                        amount = 0.7
                                    prune.l1_unstructured(module, "weight", amount=amount)
                                    print(f"  Pruned {name} with {amount*100}% sparsity")
                                except (ValueError, IndexError):
                                    # Default to 50% if can't determine layer number
                                    prune.l1_unstructured(module, "weight", amount=0.5)
                            else:
                                # Default to 50% for non-layered encoder modules
                                prune.l1_unstructured(module, "weight", amount=0.5)
                        elif "decoder" in name:
                            # Extract layer number
                            if "layers." in name:
                                try:
                                    layer_str = name.split("layers.")[1].split(".")[0]
                                    layer_num = int(layer_str)
                                    # Early layers (0-2): 30%, Middle layers (3-5): 50%, Late layers (6+): 70%
                                    if layer_num < 3:
                                        amount = 0.3
                                    elif layer_num < 6:
                                        amount = 0.5
                                    else:
                                        amount = 0.7
                                    prune.l1_unstructured(module, "weight", amount=amount)
                                    print(f"  Pruned {name} with {amount*100}% sparsity")
                                except (ValueError, IndexError):
                                    # Default to 50% if can't determine layer number
                                    prune.l1_unstructured(module, "weight", amount=0.5)
                            else:
                                # Default to 50% for non-layered decoder modules
                                prune.l1_unstructured(module, "weight", amount=0.5)
                        else:
                            # Default to 50% for other modules
                            prune.l1_unstructured(module, "weight", amount=0.5)

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

            elif model_name == "magnitude_threshold":
                # Load model
                model = WhisperForConditionalGeneration.from_pretrained(
                    original_model_name, device_map=None
                )

                # Define threshold for pruning (weights below this absolute value will be pruned)
                threshold = 0.01
                print(f"Applying magnitude threshold pruning with threshold={threshold}...")

                # Count parameters for reporting
                total_params = 0
                pruned_params = 0

                # Apply threshold pruning to each linear layer
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Get the weight tensor
                        weight = module.weight.data

                        # Create a mask of weights to keep (magnitude above threshold)
                        mask = torch.abs(weight) > threshold

                        # Count parameters
                        total_params += weight.numel()
                        pruned_params += weight.numel() - torch.sum(mask).item()

                        # Apply mask (set small weights to zero)
                        with torch.no_grad():
                            weight.mul_(mask.float())

                        print(
                            f"  Pruned {name}: {100 * (1 - torch.sum(mask).item() / weight.numel()):.2f}% sparsity"
                        )

                sparsity = 100 * pruned_params / total_params
                print(
                    f"Overall sparsity achieved: {sparsity:.2f}% ({pruned_params}/{total_params} parameters)"
                )

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
            print(f"  - Overall: {sparsity_info['overall_sparsity']:.2f}%")

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
                    if scores is not None:
                        # Get model size
                        model_size = get_model_disk_size_in_mb(model)

                        # Build results dictionary
                        results[f"{model_name}_{split}"] = {
                            "metrics": scores,
                            "model_size_mb": model_size,
                            "model_type": model_name,
                            "encoder_sparsity": sparsity_info["encoder_sparsity"],
                            "decoder_sparsity": sparsity_info["decoder_sparsity"],
                            "overall_sparsity": sparsity_info["overall_sparsity"],
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
                        results[result_key]["size_reduction_percent"] = (
                            100.0
                            * (results[result_key]["model_size_mb"] - sparse_size)
                            / results[result_key]["model_size_mb"]
                        )

                        # Update the saved metrics file
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_metrics.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[result_key], f, indent=2)

            # Clear model from memory
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
        metric_names=["WER", "CER", "RTF", "avg_cpu_percent", "ram_usage_gb"],
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

    # Group results by pruning approach
    config_groups = {
        "Encoder-only pruning": ["encoder_only_30", "encoder_only_50", "encoder_only_70"],
        "Decoder-only pruning": ["decoder_only_30", "decoder_only_50", "decoder_only_70"],
        "Advanced pruning approaches": ["layerwise_variable", "magnitude_threshold"],
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
                size_change = "-"

                if "baseline_clean" in results:
                    baseline = results["baseline_clean"]
                    wer_change = f"{(result['metrics']['WER'] - baseline['metrics']['WER']) / baseline['metrics']['WER'] * 100:+.2f}%"
                    cer_change = f"{(result['metrics']['CER'] - baseline['metrics']['CER']) / baseline['metrics']['CER'] * 100:+.2f}%"
                    rtf_change = f"{(result['metrics']['RTF'] - baseline['metrics']['RTF']) / baseline['metrics']['RTF'] * 100:+.2f}%"
                    if "sparse_model_size_mb" in result:
                        size_change = f"{(result['sparse_model_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%"

                print(f"\n  {config}:")
                print(f"    WER: {result['metrics']['WER']:.4f} ({wer_change})")
                print(f"    CER: {result['metrics']['CER']:.4f} ({cer_change})")
                print(f"    RTF: {result['metrics']['RTF']:.4f} ({rtf_change})")
                if "sparse_model_size_mb" in result:
                    print(
                        f"    Sparse Model Size: {result['sparse_model_size_mb']:.2f} MB ({size_change})"
                    )
                print(f"    Encoder Sparsity: {result['encoder_sparsity']:.2f}%")
                print(f"    Decoder Sparsity: {result['decoder_sparsity']:.2f}%")
                print(f"    Overall Sparsity: {result['overall_sparsity']:.2f}%")

    print("\nPlots saved to:", PLOTS_DIR)
    print("Sparse models saved to:", MODELS_DIR)
    print("Detailed metrics saved to:", SELECTIVE_PRUNING_DIR)


if __name__ == "__main__":
    main()
