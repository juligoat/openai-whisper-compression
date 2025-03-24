import gc  # Add garbage collection
import io
import json
import os
import time
from collections import deque
from datetime import datetime

import datasets
import numpy as np
import psutil
import torch
import torch.nn.utils.prune as prune
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Keep the WhisperMemoryTracker class as is - it's useful for measuring memory usage
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
            gpu_measurements = [m["gpu_allocated_gb"] for m in self.memory_measurements]
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


# NEW FUNCTION: Save model in sparse format for storage efficiency
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


# New function for applying pruning to Whisper models
def apply_pruning(
    model,
    method="l1_unstructured",
    amount=0.3,
    target_modules=None,
    target_submodules=None,
    make_permanent=False,
):
    """
    Apply pruning to a Whisper model.

    Args:
        model: The WhisperForConditionalGeneration model
        method: Pruning method ('l1_unstructured', 'random_unstructured', etc.)
        amount: Amount of weights to prune (0.3 = 30%)
        target_modules: List of module types to prune (None = all Linear layers)
        target_submodules: List of submodule names to target (e.g., ["encoder", "decoder"])
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
    """
    # If no specific modules are targeted, default to all Linear layers
    if target_modules is None:
        target_modules = [torch.nn.Linear]

    # Get parameters to prune based on target modules and submodules
    params_to_prune = []
    for name, module in model.named_modules():
        # Check if module is of target type
        if any(isinstance(module, m) for m in target_modules):
            # Check if it belongs to target submodule (if specified)
            if target_submodules is None or any(
                submodule in name for submodule in target_submodules
            ):
                params_to_prune.append((module, "weight"))

    if not params_to_prune:
        print("Warning: No parameters found to prune! Check your target modules and submodules.")
        return model

    print(f"Found {len(params_to_prune)} modules to prune")

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


# Function to calculate sparsity of the model
def calculate_sparsity(model):
    """
    Calculate the sparsity percentage in the model.

    Args:
        model: The PyTorch model

    Returns:
        float: Percentage of zero weights in the model
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:  # Only consider weight parameters
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    if total_params == 0:
        return 0.0

    sparsity = 100.0 * zero_params / total_params
    return sparsity


def load_whisper_model(model_name, device, pruning_config=None, use_fp16=False):
    """
    Load Whisper model and optionally apply pruning.

    Args:
        model_name: The Whisper model name
        device: Device to load the model to
        pruning_config: Dictionary with pruning settings or None
        use_fp16: Whether to use FP16 precision (only if not pruned and on CUDA)

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        # Load model without device_map
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_config:
            print(f"Applying pruning with config: {pruning_config}")
            model = apply_pruning(
                model,
                method=pruning_config.get("method", "l1_unstructured"),
                amount=pruning_config.get("amount", 0.3),
                target_modules=pruning_config.get("target_modules", None),
                target_submodules=pruning_config.get("target_submodules", None),
                make_permanent=pruning_config.get("make_permanent", False),
            )

            # Calculate and print sparsity
            sparsity = calculate_sparsity(model)
            print(f"Model sparsity after pruning: {sparsity:.2f}%")

        # Move model to device before potential FP16 conversion
        model = model.to(device)

        # Apply FP16 only if specified, not pruned, and on CUDA
        if use_fp16 and pruning_config is None and torch.cuda.is_available():
            model = model.half()
            print("Converted model to FP16.")

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


def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=2):
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


def get_non_zero_params(model):
    """Get the number of non-zero parameters in the model."""
    non_zero = 0
    total = 0
    for param in model.parameters():
        non_zero += torch.sum(param != 0).item()
        total += param.numel()
    return non_zero, total, 100.0 * non_zero / total


def main():
    original_model_name = "openai/whisper-small"
    batch_size = 16
    save_path = "results"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define pruning configurations
    pruning_configs = {
        "baseline": {
            "pruning_config": None,  # Unpruned baseline
            "use_fp16": False,
        },
        "baseline_fp16": {
            "pruning_config": None,  # Unpruned baseline with FP16
            "use_fp16": True,
        },
        "l1_unstructured_30": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.3,  # 30% pruning
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        "l1_unstructured_50": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.5,  # 50% pruning
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        "l1_unstructured_70": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.7,  # 70% pruning
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        "random_unstructured_50": {
            "pruning_config": {
                "method": "random_unstructured",
                "amount": 0.5,  # 50% random pruning
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        "ln_structured_30": {
            "pruning_config": {
                "method": "ln_structured",
                "amount": 0.3,  # 30% structured pruning
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        # Selective pruning configurations
        "encoder_only_50": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.5,
                "target_modules": [torch.nn.Linear],
                "target_submodules": ["encoder"],
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        "decoder_only_50": {
            "pruning_config": {
                "method": "l1_unstructured",
                "amount": 0.5,
                "target_modules": [torch.nn.Linear],
                "target_submodules": ["decoder"],
                "make_permanent": True,
            },
            "use_fp16": False,
        },
        # Combine pruning with FP16
        "l1_unstructured_50_fp16": {
            "pruning_config": {"method": "l1_unstructured", "amount": 0.5, "make_permanent": True},
            "use_fp16": True,
        },
    }

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load full datasets
    dataset_clean = load_librispeech(num_samples=2620, split="test.clean")
    dataset_other = load_librispeech(num_samples=2939, split="test.other")

    # Process datasets
    processed_test_data_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))
    processed_test_data_other = dataset_other.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Process configurations in specific order to ensure baseline runs first
    # Create ordered list of configurations to process
    ordered_configs = ["baseline"]  # Baseline always first
    # Add remaining configs
    ordered_configs.extend([name for name in pruning_configs if name != "baseline"])

    print(f"\nProcessing configurations in this order: {ordered_configs}")

    # Evaluate each pruning configuration in order
    for model_name in ordered_configs:
        config = pruning_configs[model_name]
        try:
            print(f"\nEvaluating {model_name} configuration...")

            # Clear memory before loading new model
            clear_gpu_memory()

            # Load model with current configuration
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                pruning_config=config["pruning_config"],
                use_fp16=config.get("use_fp16", False),
            )

            if model_name != "baseline" and config["pruning_config"] is not None:
                # Get non-zero parameters count and percentage
                non_zero, total, density = get_non_zero_params(model)
                print(f"Model parameters: {non_zero:,}/{total:,} non-zero ({density:.2f}% density)")
                print(f"Model sparsity: {100.0 - density:.2f}%")

            model.eval()

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
                        model_size = get_model_disk_size_in_mb(model)

                        # Calculate additional pruning metrics if applicable
                        additional_metrics = {}
                        if model_name != "baseline" and config["pruning_config"] is not None:
                            non_zero, total, density = get_non_zero_params(model)
                            additional_metrics = {
                                "non_zero_params": non_zero,
                                "total_params": total,
                                "density_percent": density,
                                "sparsity_percent": 100.0 - density,
                                "pruning_method": config["pruning_config"]["method"],
                                "pruning_amount": config["pruning_config"]["amount"],
                            }

                        results[f"{model_name}_{split}"] = {
                            "metrics": scores,
                            "model_size_mb": model_size,
                            "model_type": model_name,
                            "model_name": original_model_name,
                            **additional_metrics,
                        }

                        # Save metrics
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_metrics.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[f"{model_name}_{split}"], f, indent=2)

                        # Save transcriptions (optional, can use a lot of disk space)
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
                    clear_gpu_memory()

            # Save sparse model version if this was a pruned model
            if config["pruning_config"] is not None:
                sparse_model_path = os.path.join(save_path, f"{model_name}_sparse.pt")
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

    # Print final summary with baseline comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY (COMPARED TO BASELINE)")
    print("=" * 60)

    # Get baseline metrics for both splits
    baseline_results = {}
    for split in ["clean", "other"]:
        baseline_key = f"baseline_{split}"
        if baseline_key in results:
            baseline_results[split] = results[baseline_key]
        else:
            print(f"Warning: No baseline results found for {split} split")

    # First print baseline results
    print("\nBASELINE PERFORMANCE:")
    print("-" * 30)
    for split, baseline in baseline_results.items():
        print(f"\n{split.upper()} Split:")
        print(f"Model Size: {baseline['model_size_mb']:.2f} MB")
        print("Metrics:")
        for metric_name, value in baseline["metrics"].items():
            if metric_name in ["WER", "CER", "RTF"]:
                print(f"  {metric_name}: {value:.4f}")
            elif metric_name in ["total_processing_time", "total_audio_duration"]:
                print(f"  {metric_name}: {value:.2f} seconds")

    print("\n" + "=" * 60)
    print("PRUNED MODEL RESULTS (with relative changes)")
    print("=" * 60 + "\n")

    # Then print all non-baseline results with relative changes
    for run_name, run_results in results.items():
        # Skip baseline results as they were already printed
        if run_name.startswith("baseline"):
            continue

        split = "clean" if "clean" in run_name else "other"
        if split in baseline_results:
            baseline = baseline_results[split]

            print(f"\n{run_name}:")
            print("-" * len(run_name))

            # Calculate dense model size change
            size_mb = run_results["model_size_mb"]
            baseline_size_mb = baseline["model_size_mb"]
            size_change_pct = 0
            if baseline_size_mb > 0:
                size_change_pct = (size_mb - baseline_size_mb) / baseline_size_mb * 100

            print(f"Model Size (Dense): {size_mb:.2f} MB ({size_change_pct:+.1f}% vs baseline)")

        # Print sparse model size if available
        if "sparse_model_size_mb" in run_results:
            print(f"Model Size (Sparse): {run_results['sparse_model_size_mb']:.2f} MB")
            print(f"Size Reduction: {run_results['size_reduction_percent']:.1f}%")

        # Print pruning metrics if available
        if "sparsity_percent" in run_results:
            print(f"Sparsity: {run_results['sparsity_percent']:.2f}%")
            print(f"Pruning Method: {run_results['pruning_method']}")
            print(f"Pruning Amount: {run_results['pruning_amount']}")

        print("Metrics:")
        for metric_name, value in run_results["metrics"].items():
            if metric_name in ["WER", "CER", "RTF"]:
                print(f"  {metric_name}: {value:.4f}")

        # Add more detailed performance metrics
        if "total_processing_time" in run_results["metrics"]:
            print(
                f"  Total Processing Time: {run_results['metrics']['total_processing_time']:.2f} s"
            )
            print(f"  Total Audio Duration: {run_results['metrics']['total_audio_duration']:.2f} s")
            print(f"  Avg Batch Latency: {run_results['metrics']['avg_latency']:.4f} s")

        # Add CPU metrics if they exist
        if "avg_cpu_percent" in run_results["metrics"]:
            print(f"  Avg CPU: {run_results['metrics']['avg_cpu_percent']:.1f}%")
            print(f"  Peak CPU: {run_results['metrics']['peak_cpu_percent']:.1f}%")


if __name__ == "__main__":
    main()
