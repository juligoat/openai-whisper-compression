import gc
import gzip
import io
import json
import os
from collections import defaultdict

import torch
import torch.nn.utils.prune as prune
from transformers import WhisperForConditionalGeneration

# Create results directory
RESULTS_DIR = "pruning/whisper_pruning_results"
GLOBAL_PRUNING_DIR = os.path.join(RESULTS_DIR, "global_l1_pruning")
PLOTS_DIR = os.path.join(GLOBAL_PRUNING_DIR, "plots")
MODELS_DIR = os.path.join(GLOBAL_PRUNING_DIR, "models")

for directory in [RESULTS_DIR, GLOBAL_PRUNING_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


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

    Args:
        name: Module name
        param_name: Parameter name (weight or bias)

    Returns:
        str: Component type identifier or None if not subject to pruning
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
    This implementation treats each component type as a group and applies global pruning within that group,
    while using PyTorch's pruning utilities for consistency with the original code.

    Args:
        model: The WhisperForConditionalGeneration model
        pruning_config: Dictionary mapping component types to pruning percentages
        make_permanent: Whether to make pruning permanent

    Returns:
        Pruned model
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

                # Use PyTorch's pruning mechanism to maintain compatibility with original code
                # Instead of using percentage-based pruning, we'll use a custom mask
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
        return 0.0, 0, 0, 0.0, 0, 0

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


def load_whisper_model(model_name, device, pruning_config=None, make_permanent=True):
    """
    Load Whisper model and optionally apply pruning.

    Args:
        model_name: The Whisper model name
        device: Device to load the model to
        pruning_config: Dictionary mapping component types to pruning percentages, or None for no pruning
        make_permanent: Whether to make pruning permanent

    Returns:
        WhisperForConditionalGeneration model
    """
    try:
        # Load model without device_map
        model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map=None)

        # Apply pruning if specified
        if pruning_config is not None:
            print("Applying global L1 unstructured pruning with config:")
            for component, percentage in pruning_config.items():
                print(f"  {component}: {percentage}%")

            model = apply_global_l1_pruning(model, pruning_config, make_permanent=make_permanent)

            # Calculate and print sparsity
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
            ) = calculate_sparsity(model)

            print(f"Overall model sparsity after pruning: {overall_sparsity:.2f}%")
            print(f"Total parameters: {total_params:,}")
            print(f"Non-zero parameters: {overall_non_zero_params:,}")

            print(f"\nWeight sparsity: {weight_sparsity:.2f}%")
            print(f"Total weight parameters: {weight_total_params:,}")
            print(f"Non-zero weight parameters: {weight_non_zero_params:,}")

            print(f"\nBias sparsity: {bias_sparsity:.2f}%")
            print(f"Total bias parameters: {bias_total_params:,}")
            print(f"Non-zero bias parameters: {bias_non_zero_params:,}")

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


def save_compressed_model(model, path):
    """
    Save a model using gzip compression.

    Args:
        model: The PyTorch model to save
        path: Path where to save the compressed model

    Returns:
        float: Size of the compressed model in MB
    """
    print(f"\nSaving compressed model to {path}")

    # Get the state dict
    state_dict = model.state_dict()

    # Convert state dict to bytes
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    state_dict_bytes = buffer.getvalue()

    # Save with gzip compression
    compressed_path = path.replace(".pth", ".gz") if path.endswith(".pth") else path + ".gz"

    with gzip.open(compressed_path, "wb") as f:
        f.write(state_dict_bytes)

    # Get the size of the compressed file
    compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
    original_size_mb = len(state_dict_bytes) / (1024 * 1024)

    print(f"Original model size: {original_size_mb:.2f} MB")
    print(f"Compressed model size: {compressed_size_mb:.2f} MB")
    print(f"Compression ratio: {original_size_mb/compressed_size_mb:.2f}x")

    return compressed_size_mb


def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer, _use_new_zipfile_serialization=True
    )  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)


def main():
    # Configuration to match the quantization code
    original_model_name = "openai/whisper-small"
    save_path = GLOBAL_PRUNING_DIR
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

    # Store results
    results = {}

    # Define model configurations
    model_configs = {
        "baseline": {
            "pruning_config": None  # No pruning for baseline
        },
        "global_pruning": {
            "pruning_config": pruning_config  # Global pruning configuration
        },
    }

    # Process each configuration
    for model_name, config in model_configs.items():
        print("\n" + "=" * 50)
        print(f"Processing {model_name}")
        print("=" * 50)

        # Clear memory before loading new model
        clear_gpu_memory()

        try:
            # Load and prune model
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                pruning_config=config["pruning_config"],
                make_permanent=True,
            )

            # Calculate actual sparsity and parameter counts
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
            ) = calculate_sparsity(model)

            # Get model size
            model_size = get_model_disk_size_in_mb(model)

            # Build results dictionary
            results[model_name] = {
                "model_size_mb": model_size,
                "model_type": model_name,
                "pruning_type": "none" if config["pruning_config"] is None else "global",
                "overall_sparsity": overall_sparsity,
                "weight_sparsity": weight_sparsity,
                "bias_sparsity": bias_sparsity,
                "total_parameters": total_params,
                "non_zero_parameters": overall_non_zero_params,
            }

            # Save metrics
            metrics_path = os.path.join(save_path, f"{model_name}_summary.json")
            with open(metrics_path, "w") as f:
                json.dump(results[model_name], f, indent=2)
            print(f"Saved metrics to {metrics_path}")

            # Save compressed model - this is the main focus now
            compressed_model_path = os.path.join(MODELS_DIR, f"{model_name}_compressed")
            compressed_size = save_compressed_model(model, compressed_model_path)
            results[model_name]["compressed_model_size_mb"] = compressed_size

            if compressed_size > 0 and model_size > 0:
                results[model_name]["compressed_size_reduction_percent"] = (
                    100.0 * (model_size - compressed_size) / model_size
                )

            # Update the saved metrics file
            with open(metrics_path, "w") as f:
                json.dump(results[model_name], f, indent=2)

            # Clear model from memory
            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"Error setting up {model_name}: {e!s}")
            continue

    # Save all results to a single file
    all_results_path = os.path.join(GLOBAL_PRUNING_DIR, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {all_results_path}")

    # Print summary of compression results
    print("\n" + "=" * 60)
    print("MODEL COMPRESSION SUMMARY")
    print("=" * 60)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Model Size: {result['model_size_mb']:.2f} MB")
        print(f"  Compressed Size: {result.get('compressed_model_size_mb', 0):.2f} MB")
        if "compressed_size_reduction_percent" in result:
            print(
                f"  Compression Reduction: {result['compressed_size_reduction_percent']:.1f}% from original"
            )
        print(f"  Overall Sparsity: {result['overall_sparsity']:.2f}%")
        print(f"  Total Parameters: {result['total_parameters']:,}")
        print(f"  Non-zero Parameters: {result['non_zero_parameters']:,}")

    print("\nAll compressed models saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
