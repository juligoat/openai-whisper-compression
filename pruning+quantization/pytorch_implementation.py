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
RESULTS_DIR = "pruning_pytorch_quantization_results"
PRUNED_MODEL_DIR = os.path.join(RESULTS_DIR, "pruned_model")
PYTORCH_QUANT_DIR = os.path.join(RESULTS_DIR, "pytorch_quantized")

for directory in [RESULTS_DIR, PRUNED_MODEL_DIR, PYTORCH_QUANT_DIR]:
    os.makedirs(directory, exist_ok=True)


class WhisperMemoryTracker:
    def __init__(self, model_name: str, save_path: str):
        self.model_name = model_name
        self.save_path = save_path
        self.peak_cpu_percent = 0
        self.peak_ram_gb = 0
        self.memory_measurements = deque(maxlen=10)
        self.start_time = time.time()
        self.process = psutil.Process()

        self.process.cpu_percent(interval=None)  # First call returns 0, discard it
        self.initial_cpu_percent = np.mean(
            [self.process.cpu_percent(interval=0.1) for _ in range(5)]
        )
        self.initial_ram_usage = self.process.memory_info().rss / (1024**3)
        self.peak_ram_gb = self.initial_ram_usage

    def log_memory(self, split, batch_idx, batch_size, audio_duration):
        current_time = time.time()
        cpu_percent = np.mean([self.process.cpu_percent(interval=0.1) for _ in range(3)])
        current_ram = self.process.memory_info().rss / (1024**3)
        self.peak_ram_gb = max(self.peak_ram_gb, current_ram)

        memory_data = {
            "timestamp": float(current_time - self.start_time),
            "cpu_percent": float(cpu_percent),
            "ram_gb": float(current_ram),
            "batch_info": {
                "split": split,
                "batch_idx": int(batch_idx),
                "batch_size": int(batch_size),
                "audio_duration": float(audio_duration),
            },
        }

        # Append the memory measurement
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

    def close(self):
        """Cleanup and save final metrics."""
        self.print_summary()
        self.save_metrics()


def calculate_pruned_dense_size(model, pruning_threshold=0.0):
    """
    Calculate the theoretical size of a dense model with pruned weights and biases removed.
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


def calculate_theoretical_quantized_size(model, pruning_threshold=0.0, int8_quantization=True):
    """
    Calculate the theoretical size of a model that is both pruned and quantized with PyTorch.

    Args:
        model: The pruned model
        pruning_threshold: Values with absolute value below this threshold are considered pruned
        int8_quantization: Whether INT8 quantization is applied (PyTorch dynamic)

    Returns:
        float: Theoretical size in MB after pruning + quantization
    """
    print("\n=== Calculating theoretical size with pruning + PyTorch INT8 quantization ===")

    # PyTorch dynamic quantization uses INT8 for weights
    bits_per_param = 8 if int8_quantization else 32
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

        # Calculate size with quantization (only quantize Linear module weights)
        if "weight" in name and any(
            layer_type in name for layer_type in ["linear", "Linear", "layer"]
        ):
            param_quantized_bytes = non_zero_params * bytes_per_param
        else:
            # Non-weight params stay FP32
            param_quantized_bytes = non_zero_params * 4

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
        print(f"Size after PyTorch quantization: {quantized_size_mb:.2f} MB")
        print(f"Overall size reduction: {size_reduction:.1f}%")

    return quantized_size_mb


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


def load_and_prune_whisper_model(model_name, device, pruning_config=None, make_permanent=True):
    """
    Load Whisper model and optionally apply pruning.
    """
    try:
        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_config is not None:
            print("Applying custom L1 unstructured pruning with config:")
            for component, percentage in pruning_config.items():
                print(f"  {component}: {percentage}%")

            model = apply_custom_l1_pruning(model, pruning_config, make_permanent=make_permanent)

            # Calculate and print sparsity
            sparsity_metrics = calculate_sparsity(model)
            overall_sparsity = sparsity_metrics[0]
            total_params = sparsity_metrics[1]
            overall_non_zero_params = sparsity_metrics[2]
            weight_sparsity = sparsity_metrics[3]
            bias_sparsity = sparsity_metrics[6]

            print(f"Overall model sparsity after pruning: {overall_sparsity:.2f}%")
            print(f"Total parameters: {total_params:,}")
            print(f"Non-zero parameters: {overall_non_zero_params:,}")
            print(f"Weight sparsity: {weight_sparsity:.2f}%")
            print(f"Bias sparsity: {bias_sparsity:.2f}%")

        # Move model to device
        model = model.to(device)
        model.config.forced_decoder_ids = None
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def apply_pytorch_quantization(model):
    """
    Apply PyTorch dynamic quantization to a model.
    """
    print("Applying PyTorch dynamic quantization - PyTorch quantization is CPU-only")
    # Move model to CPU
    model = model.to("cpu")
    # Apply PyTorch dynamic quantization
    torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    print("PyTorch quantization applied")
    return model


def clear_memory():
    """Clear cached memory and perform garbage collection."""
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
        features = features.to(model.device)

        # Compute total audio duration for the batch
        audio_durations = [len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]]
        total_audio_duration = sum(audio_durations)

        # Measure processing time
        start_time = time.time()

        # Generate
        try:
            predicted_ids = model.generate(features)
            del features
            gc.collect()
        except Exception as e:
            print(f"Error during generation: {e}")
            try:
                del features
            except NameError:
                pass
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

    # Save per-sample RTF, processing time, and audio duration
    batch["rtf"] = [batch_rtf] * len(batch["audio"])
    batch["processing_time"] = [processing_time] * len(batch["audio"])
    batch["audio_duration"] = [total_audio_duration] * len(batch["audio"])

    return batch


def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=16):
    total_processing_time = 0.0
    total_audio_duration = 0.0
    batch_counter = 0

    # Track batch-specific metrics
    batch_rtfs = []

    print(f"Model is on device: {model.device}")

    def process_batch(batch):
        nonlocal batch_counter, total_processing_time, total_audio_duration

        # Process the batch and update the cumulative totals
        try:
            result = transcribe_batch(batch, model, processor, memory_tracker, split, batch_counter)

            # Get metrics
            batch_processing_time = result["processing_time"][0]
            batch_audio_duration = result["audio_duration"][0]
            batch_rtf = batch_processing_time / batch_audio_duration

            # Store batch metrics
            batch_rtfs.append(batch_rtf)

            print(
                f"Batch {batch_counter}: processing time = {batch_processing_time:.2f}s, "
                f"audio duration = {batch_audio_duration:.2f}s, "
                f"RTF = {batch_rtf:.6f}"
            )

            total_processing_time += batch_processing_time
            total_audio_duration += batch_audio_duration
            batch_counter += 1

        except Exception as e:
            print(f"Error processing batch {batch_counter}: {e}")
            batch["prediction"] = [""] * len(batch["audio"])
            batch["rtf"] = [0.0] * len(batch["audio"])
            batch["processing_time"] = [0.0] * len(batch["audio"])
            batch["audio_duration"] = [0.0] * len(batch["audio"])
            return batch

        # Clear memory after each batch
        clear_memory()

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


def get_model_disk_size_in_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def main():
    # Configuration
    original_model_name = "openai/whisper-small"
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # For PyTorch quantization, which is CPU-only, we'll eventually move to CPU
    print(f"Starting with device: {device}")

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

    # Load processor
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load datasets
    print("\nLoading datasets...")
    dataset_clean = load_librispeech(num_samples=2620, split="test.clean")
    dataset_other = load_librispeech(num_samples=2939, split="test.other")

    print(f"Clean dataset: {len(dataset_clean)} samples")
    print(f"Other dataset: {len(dataset_other)} samples")

    # Process datasets
    print("\nProcessing datasets...")
    processed_test_data_clean = dataset_clean.map(lambda x: map_to_feats(x, processor))
    processed_test_data_other = dataset_other.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Results storage
    results = {}

    # Step 1: Create and evaluate pruned base model
    print("\n" + "=" * 50)
    print("Creating pruned base model")
    print("=" * 50)

    clear_memory()

    # Create the pruned base model
    pruned_model = load_and_prune_whisper_model(
        model_name=original_model_name,
        device=device,
        pruning_config=pruning_config,
        make_permanent=True,
    )

    # Calculate metrics for the pruned model
    sparsity_metrics = calculate_sparsity(pruned_model)
    overall_sparsity = sparsity_metrics[0]
    total_params = sparsity_metrics[1]
    overall_non_zero_params = sparsity_metrics[2]
    weight_sparsity = sparsity_metrics[3]
    bias_sparsity = sparsity_metrics[6]

    # Calculate theoretical dense pruned size
    theoretical_dense_pruned_size = calculate_pruned_dense_size(pruned_model)

    # Save original pruned model size
    pruned_model_size = get_model_disk_size_in_mb(pruned_model)

    # Evaluate pruned model on both splits
    pruned_results = {}

    for split, dataset in [
        ("clean", processed_test_data_clean),
        ("other", processed_test_data_other),
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
                "theoretical_dense_pruned_size_mb": theoretical_dense_pruned_size,
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

    # Save the pruned model
    pruned_model_path = os.path.join(PRUNED_MODEL_DIR, "pruned_model.pt")
    torch.save(pruned_model.state_dict(), pruned_model_path)
    print(f"Saved pruned model to {pruned_model_path}")

    # Step 2: Apply PyTorch quantization to pruned model
    print("\n" + "=" * 50)
    print("Applying PyTorch quantization to pruned model")
    print("=" * 50)

    clear_memory()

    # Apply PyTorch quantization to a fresh copy of the pruned model
    pytorch_quantized_model = WhisperForConditionalGeneration.from_pretrained(original_model_name)
    pytorch_quantized_model.load_state_dict(torch.load(pruned_model_path))

    # Calculate theoretical size with pruning + quantization
    theoretical_pruned_quantized_size = calculate_theoretical_quantized_size(
        pytorch_quantized_model, pruning_threshold=0.0, int8_quantization=True
    )

    # Apply PyTorch quantization (CPU-only)
    pytorch_quantized_model = apply_pytorch_quantization(pytorch_quantized_model)

    # Get size of quantized model
    pytorch_quant_size = get_model_disk_size_in_mb(pytorch_quantized_model)

    # Evaluate on both splits
    for split, dataset in [
        ("clean", processed_test_data_clean),
        ("other", processed_test_data_other),
    ]:
        print(f"\nEvaluating PyTorch quantized model on {split} split...")

        # Initialize memory tracker
        tracker = WhisperMemoryTracker(f"pruned_pytorch_quant_{split}", PYTORCH_QUANT_DIR)

        try:
            # Run evaluation
            scores, transcriptions = evaluate_model(
                model=pytorch_quantized_model,
                processor=processor,
                dataset=dataset,
                metrics=metrics,
                memory_tracker=tracker,
                batch_size=batch_size,
                split=split,
            )

            # Store results
            results[f"pruned_pytorch_quant_{split}"] = {
                "metrics": scores,
                "model_size_mb": pytorch_quant_size,
                "theoretical_pruned_quantized_size_mb": theoretical_pruned_quantized_size,
                "model_type": "pruned_pytorch_quantization",
                "pruning_config": pruning_config,
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(
                PYTORCH_QUANT_DIR, f"pruned_pytorch_quant_{split}_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(results[f"pruned_pytorch_quant_{split}"], f, indent=2)

            # Save transcriptions
            transcriptions_path = os.path.join(
                PYTORCH_QUANT_DIR, f"pruned_pytorch_quant_{split}_transcriptions.json"
            )
            with open(transcriptions_path, "w") as f:
                json.dump(transcriptions, f, indent=2)

        except Exception as e:
            print(f"Error evaluating PyTorch quantized model on {split} split: {e}")
            continue

        finally:
            tracker.close()

    # Save combined results
    all_results_path = os.path.join(RESULTS_DIR, "pruned_pytorch_quantized_results.json")
    with open(all_results_path, "w") as f:
        results_to_save = {"pruned_baseline": pruned_results, "pruned_pytorch_quantized": results}
        json.dump(results_to_save, f, indent=2)

    # Save pytorch quantized model
    pytorch_quant_model_path = os.path.join(PYTORCH_QUANT_DIR, "pruned_pytorch_quantized_model.pt")
    torch.save(pytorch_quantized_model.state_dict(), pytorch_quant_model_path)
    print(f"Saved pytorch quantized model to {pytorch_quant_model_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PRUNED AND PYTORCH QUANTIZED EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nPruned Baseline:")
    if "clean" in pruned_results:
        baseline = pruned_results["clean"]
        print(f"  WER: {baseline['metrics']['WER']:.4f}")
        print(f"  CER: {baseline['metrics'].get('CER', 0):.4f}")
        print(f"  RTF: {baseline['metrics']['RTF']:.4f}")
        print(f"  Overall Sparsity: {baseline['overall_sparsity']:.2f}%")
        print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
        print(
            f"  Theoretical Dense Pruned Size: {baseline['theoretical_dense_pruned_size_mb']:.2f} MB"
        )

    print("\nPruned + PyTorch Quantized:")
    key = "pruned_pytorch_quant_clean"
    if key in results:
        result = results[key]

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

    print("\nResults saved to:", RESULTS_DIR)
    print("  Pruned model info:", PRUNED_MODEL_DIR)
    print("  PyTorch quantized info:", PYTORCH_QUANT_DIR)


if __name__ == "__main__":
    main()
