#!/usr/bin/env python
# Script to prune Whisper Small using global pruning and save using only numpy compression

import argparse
import io
import os
import time
import zipfile
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.utils.prune as prune
from transformers import WhisperForConditionalGeneration


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
    while using PyTorch's pruning utilities.

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


def save_whisper_optimized_numpy_only(model, output_path):
    """
    Save pruned Whisper model using only numpy compression format.

    Args:
        model: Pruned PyTorch model
        output_path: Path to save the model

    Returns:
        float: Size of saved model in MB
    """
    print("\n=== Converting model to optimized format (numpy-only) ===")

    # Track statistics
    total_params = 0
    total_zeros = 0
    dense_bytes = 0
    compressed_bytes = 0

    # Create a buffer to save compressed data
    buffer = io.BytesIO()

    # Create a zip file for storage
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Store model metadata
        metadata = {"model_type": "whisper", "format_version": "1.0", "compression": "numpy_only"}
        zf.writestr("metadata.txt", str(metadata))

        # Process each parameter
        for name, param in model.state_dict().items():
            # Get parameter as numpy array
            param_np = param.detach().cpu().numpy()

            # Calculate statistics
            param_size = param_np.size
            param_zeros = np.sum(param_np == 0)
            sparsity = 100.0 * param_zeros / param_size if param_size > 0 else 0
            total_params += param_size
            total_zeros += param_zeros

            # Original storage size (assuming float32)
            original_bytes = param_size * 4
            dense_bytes += original_bytes

            # Always use numpy's compressed format
            compressed_buf = io.BytesIO()
            np.savez_compressed(compressed_buf, data=param_np)
            compressed_data = compressed_buf.getvalue()
            compressed_bytes += len(compressed_data)

            # Store as compressed numpy - simpler file structure
            zf.writestr(f"{name}.npz", compressed_data)

            # Print info for large or highly sparse layers
            if param_size > 1000000 or (param_size > 100000 and sparsity > 80):
                print(
                    f"  {name}: {sparsity:.1f}% sparse, compressed size: {len(compressed_data)/1024/1024:.2f} MB"
                )

    # Write the zip file to disk
    buffer.seek(0)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())

    # Get actual file size
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    dense_size_mb = dense_bytes / (1024 * 1024)
    overall_sparsity = 100.0 * total_zeros / total_params if total_params > 0 else 0

    print("\nCompression summary:")
    print(f"  Overall sparsity: {overall_sparsity:.2f}%")
    print(f"  Original dense size: {dense_size_mb:.2f} MB")
    print(f"  Compressed size on disk: {actual_size_mb:.2f} MB")
    print(f"  Compression ratio: {dense_size_mb/actual_size_mb:.2f}x")

    return actual_size_mb


def load_whisper_optimized_numpy_only(model_path, device="cpu"):
    """
    Load a Whisper model that was saved in numpy-only optimized format.

    Args:
        model_path: Path to the saved model
        device: PyTorch device to load the model to

    Returns:
        OrderedDict: PyTorch state dictionary for loading into a model
    """
    print(f"Loading optimized model from {model_path}")

    # Create state dict to populate
    state_dict = OrderedDict()

    # Open the zip file
    with zipfile.ZipFile(model_path, "r") as zf:
        # Get all file entries
        file_list = zf.namelist()

        # Process each parameter file (exclude metadata)
        for file_path in file_list:
            if file_path.endswith(".npz"):
                # Extract parameter name from file path
                param_name = file_path.replace(".npz", "")

                # Load compressed numpy array
                with zf.open(file_path) as f:
                    npz_data = np.load(io.BytesIO(f.read()))
                    dense_array = npz_data["data"]

                # Convert to PyTorch tensor
                param_tensor = torch.tensor(dense_array, dtype=torch.float32, device=device)

                # Add to state dict
                state_dict[param_name] = param_tensor

    return state_dict


def load_pruned_model(model_path, device="cpu"):
    """
    Load a pruned Whisper model from optimized file.

    Args:
        model_path: Path to the saved model
        device: PyTorch device to load the model to

    Returns:
        model: Loaded WhisperForConditionalGeneration model
    """
    # Get the base model architecture first
    print("Loading base Whisper Small model architecture...")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # Load our optimized state dict
    state_dict = load_whisper_optimized_numpy_only(model_path, device)

    # Load state dict into the model
    print("Loading optimized weights into model...")
    base_model.load_state_dict(state_dict)

    return base_model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Prune Whisper Small model using global pruning and save using NumPy-only compressed storage"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pruned_models", help="Directory to save pruned model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="whisper_small_numpy_only",
        help="Name for the pruned model file",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")
    parser.add_argument(
        "--test_loading", action="store_true", help="Test loading the pruned model after saving"
    )
    parser.add_argument(
        "--increase_pruning",
        action="store_true",
        help="Increase pruning ratios for better compression",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        args.device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Define pruning configuration with the global pruning percentages from the second file
    base_pruning_config = {
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

    # Option to increase pruning for better compression
    if args.increase_pruning:
        pruning_config = {
            "encoder_ffn": 80,  # Increased from 55%
            "decoder_ffn_first": 60,  # Increased from 25%
            "decoder_ffn_middle": 75,  # Increased from 45%
            "decoder_ffn_last": 70,  # Increased from 30%
            "encoder_self_attn": 70,  # Increased from 40%
            "decoder_self_attn": 75,  # Increased from 50%
            "decoder_cross_attn": 75,  # Increased from 45%
            "layer_norm": 0,  # Keep unchanged (sensitive)
            "token_embeddings": 60,  # Increased from 25%
            "positional_embeddings": 0,  # Keep unchanged (sensitive)
            "conv_layers": 60,  # Increased from 20%
            "output_projection": 60,  # Increased from 25%
            "all_bias": 50,  # Pruning bias now (was 0)
        }
        print("Using increased pruning configuration for better compression")
    else:
        pruning_config = base_pruning_config
        print("Using original pruning configuration")

    # Step 1: Load Whisper Small model
    print("Loading Whisper Small model...")
    start_time = time.time()
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Print initial model size
    original_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (
        1024 * 1024
    )  # 4 bytes per float32
    print(f"Original model size: {original_size_mb:.2f} MB")

    # Step 2: Apply global pruning (instead of local pruning)
    print("\nApplying global pruning...")
    start_time = time.time()
    model = apply_global_l1_pruning(model, pruning_config, make_permanent=True)
    print(f"Pruning completed in {time.time() - start_time:.2f} seconds")

    # Step 3: Calculate and report sparsity
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

    print("\nSparsity report:")
    print(f"Overall sparsity: {overall_sparsity:.2f}%")
    print(f"Weight sparsity: {weight_sparsity:.2f}%")
    print(f"Bias sparsity: {bias_sparsity:.2f}%")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {overall_non_zero_params:,}")

    # Step 4: Save model using numpy-only optimized storage
    output_path = os.path.join(args.output_dir, f"{args.model_name}.zip")
    print("\nSaving model...")
    start_time = time.time()
    sparse_size_mb = save_whisper_optimized_numpy_only(model, output_path)
    print(f"Model saved in {time.time() - start_time:.2f} seconds")

    print("\nResults summary:")
    print(f"Original model size: {original_size_mb:.2f} MB")
    print(f"Optimized model size: {sparse_size_mb:.2f} MB")
    print(f"Compression ratio: {original_size_mb/sparse_size_mb:.2f}x")
    print(f"Size reduction: {100.0 * (original_size_mb - sparse_size_mb) / original_size_mb:.1f}%")

    # Step 5: Test loading the model if requested
    if args.test_loading:
        print("\nTesting model loading...")
        start_time = time.time()
        loaded_model = load_pruned_model(output_path, device)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

        # Verify the loaded model
        print("\nVerifying loaded model...")
        (
            loaded_sparsity,
            loaded_total_params,
            loaded_non_zero_params,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = calculate_sparsity(loaded_model)

        print(f"Loaded model sparsity: {loaded_sparsity:.2f}%")
        print(f"Loaded model parameters: {loaded_total_params:,}")
        print(f"Loaded model non-zero parameters: {loaded_non_zero_params:,}")

        if abs(loaded_sparsity - overall_sparsity) < 0.01 and loaded_total_params == total_params:
            print("✅ Loaded model verification successful!")
        else:
            print("❌ Loaded model does not match the original pruned model")

    print(f"\nPruned model saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
