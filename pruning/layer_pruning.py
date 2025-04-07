import gc
import io
import json
import os
import time
from collections import deque

import datasets
import numpy as np
import psutil
import torch
import torch.nn.utils.prune as prune
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Create results directory
RESULTS_DIR = "pruning/whisper_targeted_pruning_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def get_component_layers(model, component):
    """
    Get all layers in encoder or decoder with their indices.

    Args:
        model: The WhisperForConditionalGeneration model
        component: 'encoder' or 'decoder'

    Returns:
        tuple: (number of layers, dict mapping layer indices to layer modules)
    """
    # Get all layers in the component
    layer_modules = {}

    # First find the highest layer index to determine total count
    max_layer_idx = -1
    for name, module in model.named_modules():
        if f"{component}.layers." in name:
            parts = name.split(f"{component}.layers.")[-1].split(".")
            if parts[0].isdigit():
                layer_idx = int(parts[0])
                max_layer_idx = max(max_layer_idx, layer_idx)

    # Layer indices are 0-indexed, so add 1 for count
    layer_count = max_layer_idx + 1

    # Print out which component we're analyzing
    print(f"Found {layer_count} layers in the {component}")

    return layer_count


def apply_targeted_l1_pruning(
    model, amount=0.3, target_component="encoder", target_section="early", make_permanent=False
):
    """
    Apply L1 unstructured pruning to specific parts of the Whisper model.

    Args:
        model: The WhisperForConditionalGeneration model
        amount: Amount of weights to prune (0.3 = 30%)
        target_component: 'encoder' or 'decoder'
        target_section: 'early', 'middle', or 'late'
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
    """
    # For Whisper small, we know there are exactly 12 encoder and 12 decoder layers
    # So we'll explicitly define each section as 4 layers

    # Determine layer indices for the target section
    if target_section == "early":
        target_layers = [0, 1, 2, 3]  # First 4 layers
    elif target_section == "middle":
        target_layers = [4, 5, 6, 7]  # Middle 4 layers
    elif target_section == "late":
        target_layers = [8, 9, 10, 11]  # Last 4 layers
    else:
        raise ValueError(f"Unknown target section: {target_section}")

    print(f"Targeting {target_component} {target_section} layers: {target_layers}")

    # Get parameters to prune based on target section
    params_to_prune = []
    prunable_linear_count = 0
    targeted_linear_count = 0

    # First count all linear layers in the model for reference
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            prunable_linear_count += 1

            # Check if this linear layer belongs to our target component and section
            if f"{target_component}.layers." in name:
                parts = name.split(f"{target_component}.layers.")[1].split(".")
                if parts[0].isdigit() and int(parts[0]) in target_layers:
                    params_to_prune.append((module, "weight"))
                    targeted_linear_count += 1

    # Print statistics about what we're pruning
    print(f"Total prunable linear layers in model: {prunable_linear_count}")
    print(f"Linear layers in {target_component} {target_section} section: {targeted_linear_count}")

    if not params_to_prune:
        print(
            f"Warning: No parameters found to prune for {target_component} {target_section} section!"
        )
        return model

    print(
        f"Found {len(params_to_prune)} modules to prune in {target_component} {target_section} section with L1 unstructured pruning, amount={amount}"
    )

    # Apply L1 unstructured pruning
    success_count = 0
    try:
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=amount
        )
        success_count = len(params_to_prune)
    except Exception as e:
        print(f"Error during global unstructured pruning: {e}")

        # Try pruning modules individually if global pruning fails
        for i, (module, param_name) in enumerate(params_to_prune):
            try:
                prune.l1_unstructured(module, param_name, amount=amount)
                success_count += 1
            except Exception as e2:
                print(f"Error pruning module {i}: {e2}")

    print(f"Successfully applied pruning to {success_count}/{len(params_to_prune)} modules")

    # Make pruning permanent if requested
    if make_permanent:
        print("Making pruning permanent...")
        permanent_count = 0
        for module, param_name in params_to_prune:
            try:
                # Only make permanent if the module has a weight_mask attribute
                if hasattr(module, f"{param_name}_mask"):
                    prune.remove(module, param_name)
                    permanent_count += 1
            except Exception as e:
                print(f"Could not make pruning permanent for {module}: {e}")
        print(f"Made pruning permanent for {permanent_count}/{len(params_to_prune)} modules")

    return model


def load_whisper_model(model_name, device, pruning_config=None, make_permanent=True):
    """
    Load Whisper model and optionally apply targeted pruning.

    Args:
        model_name: The Whisper model name
        device: Device to load the model to
        pruning_config: Dict with keys 'amount', 'component', 'section' or None for no pruning
        make_permanent: Whether to make pruning permanent

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_config is not None and pruning_config.get("amount", 0) > 0:
            print(
                f"Applying targeted pruning to {pruning_config['component']} {pruning_config['section']} "
                f"section with amount={pruning_config['amount']}"
            )

            model = apply_targeted_l1_pruning(
                model,
                amount=pruning_config["amount"],
                target_component=pruning_config["component"],
                target_section=pruning_config["section"],
                make_permanent=make_permanent,
            )

            # Calculate and print sparsity
            sparsity, total_params, non_zero_params = calculate_sparsity(model)
            print(f"Model sparsity after pruning: {sparsity:.2f}%")
            print(f"Total parameters: {total_params:,}")
            print(f"Non-zero parameters: {non_zero_params:,}")

        # Move model to device
        model = model.to(device)
        model.config.forced_decoder_ids = None
        return model

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


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
    # Configuration to match the quantization code
    original_model_name = "openai/whisper-small"
    batch_size = 16  # Match the quantization code batch size
    save_path = RESULTS_DIR
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if (
        not torch.cuda.is_available()
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")  # Use MPS for Apple Silicon if available
    print(f"Using {device}")

    # Define pruning configurations to test
    pruning_configs = [
        # Baseline (no pruning)
        {"name": "baseline", "config": None},
        # 30% pruning configurations
        {
            "name": "encoder_early_30",
            "config": {"amount": 0.3, "component": "encoder", "section": "early"},
        },
        {
            "name": "encoder_middle_30",
            "config": {"amount": 0.3, "component": "encoder", "section": "middle"},
        },
        {
            "name": "encoder_late_30",
            "config": {"amount": 0.3, "component": "encoder", "section": "late"},
        },
        {
            "name": "decoder_early_30",
            "config": {"amount": 0.3, "component": "decoder", "section": "early"},
        },
        {
            "name": "decoder_middle_30",
            "config": {"amount": 0.3, "component": "decoder", "section": "middle"},
        },
        {
            "name": "decoder_late_30",
            "config": {"amount": 0.3, "component": "decoder", "section": "late"},
        },
        # 40% pruning configurations
        {
            "name": "encoder_early_40",
            "config": {"amount": 0.4, "component": "encoder", "section": "early"},
        },
        {
            "name": "encoder_middle_40",
            "config": {"amount": 0.4, "component": "encoder", "section": "middle"},
        },
        {
            "name": "encoder_late_40",
            "config": {"amount": 0.4, "component": "encoder", "section": "late"},
        },
        {
            "name": "decoder_early_40",
            "config": {"amount": 0.4, "component": "decoder", "section": "early"},
        },
        {
            "name": "decoder_middle_40",
            "config": {"amount": 0.4, "component": "decoder", "section": "middle"},
        },
        {
            "name": "decoder_late_40",
            "config": {"amount": 0.4, "component": "decoder", "section": "late"},
        },
    ]

    # Print the test plan
    print("\nTEST PLAN:")
    print("1. Baseline: No pruning")
    print("2. Targeted pruning at 30% sparsity:")
    print("   - Early encoder layers (layers 0-3)")
    print("   - Middle encoder layers (layers 4-7)")
    print("   - Late encoder layers (layers 8-11)")
    print("   - Early decoder layers (layers 0-3)")
    print("   - Middle decoder layers (layers 4-7)")
    print("   - Late decoder layers (layers 8-11)")
    print("3. Targeted pruning at 40% sparsity for the same sections")

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load a smaller subset of test data for faster evaluation
    print("\nLoading datasets...")
    dataset_clean = load_librispeech(num_samples=500, split="test.clean")

    print(f"Clean dataset: {len(dataset_clean)} samples")

    # Process datasets
    print("\nProcessing datasets...")
    processed_test_data_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Evaluate each configuration
    for pruning_config in pruning_configs:
        config_name = pruning_config["name"]
        print("\n" + "=" * 50)
        print(f"Evaluating {config_name}")
        print("=" * 50)

        # Clear memory before loading new model
        clear_gpu_memory()

        try:
            # Load and prune model
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                pruning_config=pruning_config["config"],
                make_permanent=True,
            )

            # Calculate actual sparsity and parameter counts
            sparsity, total_params, non_zero_params = calculate_sparsity(model)
            print(f"Actual model sparsity: {sparsity:.2f}%")
            print(f"Total parameters: {total_params:,}")
            print(f"Non-zero parameters: {non_zero_params:,}")

            # Get model size
            model_size = get_model_disk_size_in_mb(model)

            # Evaluate on clean split
            print("\nEvaluating on clean split...")

            # Initialize memory tracker for this run
            tracker = WhisperMemoryTracker(f"{config_name}_clean", save_path)

            try:
                # Evaluate the model
                scores, result = evaluate_model(
                    model=model,
                    processor=processor,
                    dataset=processed_test_data_clean,
                    metrics=metrics,
                    memory_tracker=tracker,
                    batch_size=batch_size,
                    split="clean",
                )

                # Store results
                if isinstance(scores, dict) and "error" not in scores:
                    # Build results dictionary
                    results[config_name] = {
                        "metrics": scores,
                        "model_size_mb": model_size,
                        "actual_sparsity": sparsity,
                        "total_parameters": total_params,
                        "non_zero_parameters": non_zero_params,
                    }

                    # Save metrics
                    metrics_path = os.path.join(save_path, f"{config_name}_summary.json")
                    with open(metrics_path, "w") as f:
                        json.dump(results[config_name], f, indent=2)
                    print(f"Saved metrics to {metrics_path}")

            except Exception as e:
                print(f"Error evaluating {config_name}: {e!s}")
                continue

            finally:
                # Always close tracker and clear memory
                tracker.close()

            # Clear model from memory
            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"Error setting up {config_name}: {e!s}")
            continue

    # Save all results to a single file
    all_results_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {all_results_path}")

    # Generate results table
    print("\n" + "=" * 80)
    print("TARGETED PRUNING EXPERIMENT RESULTS")
    print("=" * 80)

    # Define the metrics to include in the table
    table_metrics = ["WER", "CER", "RTF", "actual_sparsity"]

    # Print table header
    header = "| Configuration | WER | CER | RTF | Sparsity | Model Size (MB) |"
    separator = (
        "|"
        + "-" * 15
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 8
        + "|"
        + "-" * 10
        + "|"
        + "-" * 16
        + "|"
    )

    print(header)
    print(separator)

    # Print baseline first
    if "baseline" in results:
        baseline = results["baseline"]
        baseline_row = "| baseline | "
        for metric in table_metrics:
            if metric in baseline["metrics"]:
                baseline_row += f"{baseline['metrics'][metric]:.4f} | "
            elif metric == "actual_sparsity":
                baseline_row += f"{baseline[metric]:.2f}% | "
            else:
                baseline_row += "N/A | "
        baseline_row += f"{baseline['model_size_mb']:.2f} |"
        print(baseline_row)

    # Print 30% pruning results
    print("\n| 30% Pruning | WER | CER | RTF | Sparsity | Model Size (MB) |")
    print(separator)

    for config_name in [c["name"] for c in pruning_configs if "30" in c["name"]]:
        if config_name in results:
            config_result = results[config_name]
            row = f"| {config_name} | "
            for metric in table_metrics:
                if metric in config_result["metrics"]:
                    row += f"{config_result['metrics'][metric]:.4f} | "
                elif metric == "actual_sparsity":
                    row += f"{config_result[metric]:.2f}% | "
                else:
                    row += "N/A | "
            row += f"{config_result['model_size_mb']:.2f} |"
            print(row)

    # Print 40% pruning results
    print("\n| 40% Pruning | WER | CER | RTF | Sparsity | Model Size (MB) |")
    print(separator)

    for config_name in [c["name"] for c in pruning_configs if "40" in c["name"]]:
        if config_name in results:
            config_result = results[config_name]
            row = f"| {config_name} | "
            for metric in table_metrics:
                if metric in config_result["metrics"]:
                    row += f"{config_result['metrics'][metric]:.4f} | "
                elif metric == "actual_sparsity":
                    row += f"{config_result[metric]:.2f}% | "
                else:
                    row += "N/A | "
            row += f"{config_result['model_size_mb']:.2f} |"
            print(row)

    print("\nDetailed metrics saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
