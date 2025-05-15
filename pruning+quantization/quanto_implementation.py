import gc
import io
import json
import os
import time
from collections import defaultdict, deque

import datasets
import numpy as np
import psutil
import torch
import torch.nn.utils.prune as prune
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.quanto import freeze, qint4, qint8, quantize

# Create results directory
RESULTS_DIR = "global_pruning_quanto_quantization_results"
PRUNED_MODEL_DIR = os.path.join(RESULTS_DIR, "global_pruned_model")
QUANTO_QINT4_DIR = os.path.join(RESULTS_DIR, "global_pruned_quanto_qint4")
QUANTO_QINT8_DIR = os.path.join(RESULTS_DIR, "global_pruned_quanto_qint8")

for directory in [RESULTS_DIR, PRUNED_MODEL_DIR, QUANTO_QINT4_DIR, QUANTO_QINT8_DIR]:
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


def calculate_theoretical_quantized_size(model, bit_width=8):
    """
    Calculate the theoretical size of a model after Quanto quantization.
    Takes into account that Quanto quantization keeps the dense format, meaning zeros still take up space.
    
    Args:
        model: The model to quantize (can be pruned or not)
        bit_width: Quantization bit width (4 for INT4, 8 for INT8)
    
    Returns:
        float: Theoretical size in MB after quantization
    """
    print(f"\n=== Calculating theoretical size with Quanto INT{bit_width} quantization ===")
    
    total_bytes = 0
    weight_bytes = 0
    bias_bytes = 0
    
    # Count parameters by type
    total_weight_params = 0
    total_bias_params = 0
    
    # For each parameter in the model
    for name, param in model.named_parameters():
        # Weights are quantized to reduced precision
        if "weight" in name:
            # Each weight uses bit_width bits regardless of whether it's zero or not
            param_bytes = param.numel() * (bit_width / 8) 
            # Add quantization overhead (scales, zero points) - approximately 0.1% overhead
            param_bytes += param.numel() * (bit_width / 8) * 0.001
            weight_bytes += param_bytes
            total_weight_params += param.numel()
        # Biases typically remain in FP32
        else:
            param_bytes = param.numel() * 4  # 4 bytes per float32
            bias_bytes += param_bytes
            total_bias_params += param.numel()
            
        total_bytes += param_bytes
    
    # Convert to MB
    total_size_mb = total_bytes / (1024 * 1024)
    weight_size_mb = weight_bytes / (1024 * 1024)
    bias_size_mb = bias_bytes / (1024 * 1024)
    
    # Original FP32 size for comparison
    original_size_mb = (total_weight_params * 4 + total_bias_params * 4) / (1024 * 1024)
    size_reduction = 100.0 * (original_size_mb - total_size_mb) / original_size_mb
    
    # Print size breakdown
    print(f"Original model size (FP32): {original_size_mb:.2f} MB")
    print(f"Theoretical quantized model size: {total_size_mb:.2f} MB")
    print(f"  - Quantized weights ({bit_width}-bit): {weight_size_mb:.2f} MB")
    print(f"  - FP32 biases: {bias_size_mb:.2f} MB")
    print(f"Size reduction: {size_reduction:.1f}%")
    
    return total_size_mb


def calculate_model_gflops(model):
    """
    Calculate approximate GFLOPs for Whisper model accounting for pruning.
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


def determine_component_type(name, param_name):
    """
    Determine which component a parameter belongs to for global pruning.
    """
    # For bias parameters
    if param_name == "bias":
        return "all_bias"

    # For weight parameters - determine component type
    if "encoder" in name and ("fc1" in name or "fc2" in name):
        return "encoder_ffn"

    elif "decoder" in name and ("fc1" in name or "fc2" in name):
        # Extract layer number
        layer_num = extract_layer_num(name)
        if layer_num is not None:
            if layer_num < 4:  # First 4 layers (0-3)
                return "decoder_ffn_first"
            elif layer_num < 8:  # Middle 4 layers (4-7)
                return "decoder_ffn_middle"
            else:  # Last 4 layers (8-11)
                return "decoder_ffn_last"

    # Encoder Self Attention
    elif (
        "encoder" in name
        and "self_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        return "encoder_self_attn"

    # Decoder Self Attention
    elif (
        "decoder" in name
        and "self_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        return "decoder_self_attn"

    # Decoder Cross Attention
    elif (
        "decoder" in name
        and "encoder_attn" in name
        and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name)
    ):
        return "decoder_cross_attn"

    # LayerNorm
    elif "layer_norm" in name.lower() or "layernorm" in name.lower():
        return "layer_norm"

    # Token embeddings
    elif "embed_tokens" in name:
        return "token_embeddings"

    # Positional embeddings
    elif "embed_positions" in name:
        return "positional_embeddings"

    # Convolutional layers
    elif "conv" in name.lower():
        return "conv_layers"

    # Final output projection
    elif "proj_out" in name:
        return "output_projection"

    # Not subject to pruning
    return None


def apply_global_l1_pruning(model, pruning_config, make_permanent=False):
    """
    Apply global L1 unstructured pruning to a Whisper model with custom percentages for different components.
    This implementation treats each component type as a group and applies global pruning within that group.
    """
    print("\n=== Applying Global L1 Unstructured Pruning ===")

    # Track components to be pruned
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

    # Dictionary to collect parameters by component type
    component_params = defaultdict(list)
    component_modules = defaultdict(list)

    # Step 1: Collect all parameters by component type
    for name, module in model.named_modules():
        # Handle weight parameters
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            comp_type = determine_component_type(name, "weight")
            if comp_type and pruning_config.get(comp_type, 0) > 0:
                component_params[comp_type].append(module.weight.data.float().abs().flatten())
                component_modules[comp_type].append((module, "weight"))
                components_pruned[comp_type] += 1

        # Handle bias parameters separately
        if hasattr(module, "bias") and module.bias is not None:
            if pruning_config.get("all_bias", 0) > 0:
                component_params["all_bias"].append(module.bias.data.float().abs().flatten())
                component_modules["all_bias"].append((module, "bias"))
                components_pruned["all_bias"] += 1

    # Print summary of components to be pruned
    print("\nComponents to be pruned (global method):")
    for component, count in components_pruned.items():
        if count > 0:
            amount = pruning_config.get(component, 0)
            print(f"  {component}: {count} modules at {amount}% global pruning")

    # Step 2: Calculate global thresholds for each component type
    total_pruned_modules = 0
    for comp_type, params_list in component_params.items():
        if not params_list:
            continue

        pruning_amount = pruning_config.get(comp_type, 0) / 100.0
        if pruning_amount <= 0:
            continue

        # Concatenate all parameters for this component type
        all_weights = torch.cat(params_list)

        # Calculate threshold for global pruning
        k = int(all_weights.numel() * pruning_amount)
        if k > 0:
            threshold = torch.kthvalue(all_weights, k).values.item()

            total_pruned_modules += len(component_modules[comp_type])
            print(
                f"Component {comp_type}: {pruning_amount*100:.1f}% global pruning, threshold = {threshold:.6f}"
            )
            print(
                f"  Affects {len(component_modules[comp_type])} modules with {all_weights.numel():,} parameters"
            )

            # Apply custom L1 unstructured pruning with the global threshold to each module
            for module, param_name in component_modules[comp_type]:
                param = getattr(module, param_name)

                # Custom pruning: Create a mask based on the global threshold
                mask = param.data.float().abs() > threshold

                # Use PyTorch's pruning mechanism to maintain compatibility
                prune.CustomFromMask.apply(module, param_name, mask)

    print(f"Successfully applied global pruning to {total_pruned_modules} modules")

    # Make pruning permanent if requested
    if make_permanent:
        print("\nMaking pruning permanent...")
        permanent_count = 0

        # Make pruning permanent for all modules
        for comp_type, modules_list in component_modules.items():
            for module, param_name in modules_list:
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
            print("Applying global L1 unstructured pruning with config:")
            for component, percentage in pruning_config.items():
                print(f"  {component}: {percentage}%")

            model = apply_global_l1_pruning(model, pruning_config, make_permanent=make_permanent)

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

            # Calculate model GFLOPs
            gflops = calculate_model_gflops(model)
            print(f"Estimated model complexity: {gflops:.4f} GFLOPs")

        # Move model to device
        model = model.to(device)
        model.config.forced_decoder_ids = None
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise


def apply_quanto_quantization(model, is_int4=False):
    """
    Apply Quanto quantization to a model.
    
    Args:
        model: The model to quantize
        is_int4: If True, use INT4 quantization, otherwise use INT8
    
    Returns:
        Quantized model
    """
    if is_int4:
        print("Applying Quanto INT4 quantization")
        quantize(model, weights=qint4)
    else:
        print("Applying Quanto INT8 quantization")
        quantize(model, weights=qint8)
    
    # Freeze the model
    freeze(model)
    
    print(f"Quanto {'INT4' if is_int4 else 'INT8'} quantization applied")
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


def save_sparse_model(model, output_path):
    """
    Convert pruned model to sparse format for storage efficiency.
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


def main():
    # Configuration
    original_model_name = "openai/whisper-small"
    batch_size = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Starting with device: {device}")

    # Define custom pruning configuration with the specified percentages from the second script
    pruning_config = {
        "encoder_ffn": 55,
        "decoder_ffn_first": 25,
        "decoder_ffn_middle": 45,
        "decoder_ffn_last": 30,
        "encoder_self_attn": 40,
        "decoder_self_attn": 50,
        "decoder_cross_attn": 45,
        "layer_norm": 0,
        "token_embeddings": 25,
        "positional_embeddings": 0,
        "conv_layers": 20,
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
    print("Creating globally pruned base model")
    print("=" * 50)

    clear_memory()

    # Create the pruned base model using global pruning
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

    # Save sparse model
    sparse_model_path = os.path.join(PRUNED_MODEL_DIR, "global_pruned_sparse_model.pt")
    sparse_model_size = save_sparse_model(pruned_model, sparse_model_path)

    # Save original pruned model size
    pruned_model_size = get_model_disk_size_in_mb(pruned_model)

    # Evaluate pruned model on both splits
    pruned_results = {}

    for split, dataset in [
        ("clean", processed_test_data_clean),
        ("other", processed_test_data_other),
    ]:
        print(f"\nEvaluating globally pruned base model on {split} split...")

        # Initialize memory tracker
        tracker = WhisperMemoryTracker(f"global_pruned_baseline_{split}", PRUNED_MODEL_DIR)

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
                "sparse_model_size_mb": sparse_model_size,
                "theoretical_dense_pruned_size_mb": theoretical_dense_pruned_size,
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(PRUNED_MODEL_DIR, f"global_pruned_baseline_{split}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(pruned_results[split], f, indent=2)

        except Exception as e:
            print(f"Error evaluating pruned base model on {split} split: {e}")

        finally:
            tracker.close()

    # Save the pruned model
    pruned_model_path = os.path.join(PRUNED_MODEL_DIR, "global_pruned_model.pt")
    torch.save(pruned_model.state_dict(), pruned_model_path)
    print(f"Saved pruned model to {pruned_model_path}")

    # Step 2: Apply Quanto INT4 quantization to pruned model
    print("\n" + "=" * 50)
    print("Applying Quanto INT4 quantization to globally pruned model")
    print("=" * 50)

    clear_memory()

    # Apply Quanto INT4 quantization to a fresh copy of the pruned model
    qint4_model = WhisperForConditionalGeneration.from_pretrained(original_model_name)
    qint4_model.load_state_dict(torch.load(pruned_model_path))
    qint4_model = qint4_model.to(device)
    qint4_model.config.forced_decoder_ids = None

    # Calculate theoretical size with pruning + INT4 quantization using the fixed function
    theoretical_pruned_int4_size = calculate_theoretical_quantized_size(
        qint4_model, bit_width=4
    )

    # Apply Quanto INT4 quantization
    qint4_model = apply_quanto_quantization(qint4_model, is_int4=True)

    # Get size of quantized model
    qint4_model_size = get_model_disk_size_in_mb(qint4_model)

    # Evaluate INT4 model on both splits
    qint4_results = {}

    for split, dataset in [
        ("clean", processed_test_data_clean),
        ("other", processed_test_data_other),
    ]:
        print(f"\nEvaluating Quanto INT4 quantized model on {split} split...")

        # Initialize memory tracker
        tracker = WhisperMemoryTracker(f"global_pruned_quanto_int4_{split}", QUANTO_QINT4_DIR)

        try:
            # Run evaluation
            scores, transcriptions = evaluate_model(
                model=qint4_model,
                processor=processor,
                dataset=dataset,
                metrics=metrics,
                memory_tracker=tracker,
                batch_size=batch_size,
                split=split,
            )

            # Store results
            qint4_results[split] = {
                "metrics": scores,
                "model_size_mb": qint4_model_size,
                "theoretical_pruned_int4_size_mb": theoretical_pruned_int4_size,
                "model_type": "global_pruned_quanto_int4",
                "pruning_config": pruning_config,
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(
                QUANTO_QINT4_DIR, f"global_pruned_quanto_int4_{split}_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(qint4_results[split], f, indent=2)

            # Save transcriptions
            transcriptions_path = os.path.join(
                QUANTO_QINT4_DIR, f"global_pruned_quanto_int4_{split}_transcriptions.json"
            )
            with open(transcriptions_path, "w") as f:
                json.dump(transcriptions, f, indent=2)

        except Exception as e:
            print(f"Error evaluating Quanto INT4 quantized model on {split} split: {e}")
            continue

        finally:
            tracker.close()

    # Save INT4 model
    qint4_model_path = os.path.join(QUANTO_QINT4_DIR, "global_pruned_quanto_int4_model.pt")
    torch.save(qint4_model.state_dict(), qint4_model_path)
    print(f"Saved Quanto INT4 model to {qint4_model_path}")

    # Step 3: Apply Quanto INT8 quantization to pruned model
    print("\n" + "=" * 50)
    print("Applying Quanto INT8 quantization to globally pruned model")
    print("=" * 50)

    clear_memory()

    # Apply Quanto INT8 quantization to a fresh copy of the pruned model
    qint8_model = WhisperForConditionalGeneration.from_pretrained(original_model_name)
    qint8_model.load_state_dict(torch.load(pruned_model_path))
    qint8_model = qint8_model.to(device)
    qint8_model.config.forced_decoder_ids = None

    # Calculate theoretical size with pruning + INT8 quantization using the fixed function
    theoretical_pruned_int8_size = calculate_theoretical_quantized_size(
        qint8_model, bit_width=8
    )

    # Apply Quanto INT8 quantization
    qint8_model = apply_quanto_quantization(qint8_model, is_int4=False)  # INT8

    # Get size of quantized model
    qint8_model_size = get_model_disk_size_in_mb(qint8_model)

    # Evaluate INT8 model on both splits
    qint8_results = {}

    for split, dataset in [
        ("clean", processed_test_data_clean),
        ("other", processed_test_data_other),
    ]:
        print(f"\nEvaluating Quanto INT8 quantized model on {split} split...")

        # Initialize memory tracker
        tracker = WhisperMemoryTracker(f"global_pruned_quanto_int8_{split}", QUANTO_QINT8_DIR)

        try:
            # Run evaluation
            scores, transcriptions = evaluate_model(
                model=qint8_model,
                processor=processor,
                dataset=dataset,
                metrics=metrics,
                memory_tracker=tracker,
                batch_size=batch_size,
                split=split,
            )

            # Store results
            qint8_results[split] = {
                "metrics": scores,
                "model_size_mb": qint8_model_size,
                "theoretical_pruned_int8_size_mb": theoretical_pruned_int8_size,
                "model_type": "global_pruned_quanto_int8",
                "pruning_config": pruning_config,
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(
                QUANTO_QINT8_DIR, f"global_pruned_quanto_int8_{split}_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(qint8_results[split], f, indent=2)

            # Save transcriptions
            transcriptions_path = os.path.join(
                QUANTO_QINT8_DIR, f"global_pruned_quanto_int8_{split}_transcriptions.json"
            )
            with open(transcriptions_path, "w") as f:
                json.dump(transcriptions, f, indent=2)

        except Exception as e:
            print(f"Error evaluating Quanto INT8 quantized model on {split} split: {e}")
            continue

        finally:
            tracker.close()

    # Save INT8 model
    qint8_model_path = os.path.join(QUANTO_QINT8_DIR, "global_pruned_quanto_int8_model.pt")
    torch.save(qint8_model.state_dict(), qint8_model_path)
    print(f"Saved Quanto INT8 model to {qint8_model_path}")

    # Save combined results
    all_results_path = os.path.join(RESULTS_DIR, "global_pruned_quanto_quantized_results.json")
    with open(all_results_path, "w") as f:
        results_to_save = {
            "global_pruned_baseline": pruned_results,
            "global_pruned_quanto_int4": qint4_results,
            "global_pruned_quanto_int8": qint8_results,
        }
        json.dump(results_to_save, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GLOBAL PRUNED AND QUANTO QUANTIZED EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nGlobal Pruned Baseline:")
    if "clean" in pruned_results:
        baseline = pruned_results["clean"]
        print(f"  WER: {baseline['metrics']['WER']:.4f}")
        print(f"  CER: {baseline['metrics'].get('CER', 0):.4f}")
        print(f"  RTF: {baseline['metrics']['RTF']:.4f}")
        print(f"  Overall Sparsity: {baseline['overall_sparsity']:.2f}%")
        print(f"  Model Size: {baseline['model_size_mb']:.2f} MB")
        print(f"  Sparse Model Size: {baseline['sparse_model_size_mb']:.2f} MB")
        print(
            f"  Theoretical Dense Pruned Size: {baseline['theoretical_dense_pruned_size_mb']:.2f} MB"
        )

    print("\nGlobal Pruned + Quanto INT4:")
    key = "clean"
    if key in qint4_results:
        result = qint4_results[key]

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

        if "theoretical_pruned_int4_size_mb" in result:
            theoretical_change = f"{(result['theoretical_pruned_int4_size_mb'] - baseline['model_size_mb']):.2f} MB ({(result['theoretical_pruned_int4_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%)"
            print(
                f"  Theoretical Pruned+INT4 Size: {result['theoretical_pruned_int4_size_mb']:.2f} MB (Δ {theoretical_change})"
            )

    print("\nGlobal Pruned + Quanto INT8:")
    key = "clean"
    if key in qint8_results:
        result = qint8_results[key]

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

        if "theoretical_pruned_int8_size_mb" in result:
            theoretical_change = f"{(result['theoretical_pruned_int8_size_mb'] - baseline['model_size_mb']):.2f} MB ({(result['theoretical_pruned_int8_size_mb'] - baseline['model_size_mb']) / baseline['model_size_mb'] * 100:+.2f}%)"
            print(
                f"  Theoretical Pruned+INT8 Size: {result['theoretical_pruned_int8_size_mb']:.2f} MB (Δ {theoretical_change})"
            )

    print("\nResults saved to:", RESULTS_DIR)
    print("  Global Pruned model info:", PRUNED_MODEL_DIR)
    print("  Quanto INT4 info:", QUANTO_QINT4_DIR)
    print("  Quanto INT8 info:", QUANTO_QINT8_DIR)


if __name__ == "__main__":
    main()