#!/usr/bin/env python
# Script to prune Whisper Small and save using optimized compressed storage

import argparse
import io
import os
import time
import zipfile
from collections import OrderedDict

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


def determine_pruning_amount(name, param_name, pruning_config):
    """
    Determine pruning percentage for a specific parameter based on its name and type.

    Args:
        name: Module name
        param_name: Parameter name (weight or bias)
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
    Apply L1 unstructured pruning to a Whisper model with custom percentages.

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
            weight_amount = determine_pruning_amount(name, "weight", pruning_config)

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
            bias_amount = determine_pruning_amount(name, "bias", pruning_config)

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


def save_whisper_optimized(model, output_path):
    """
    Save pruned Whisper model in an optimized format that actually reduces size.

    Args:
        model: Pruned PyTorch model
        output_path: Path to save the model

    Returns:
        float: Size of saved model in MB
    """
    print("\n=== Converting model to optimized sparse format ===")

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
        metadata = {"model_type": "whisper", "format_version": "1.0", "compression": "zip_deflate"}
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

            # Determine storage method based on sparsity
            if sparsity > 70:  # Only use sparse for highly sparse tensors
                # For very sparse tensors, store only non-zero values and their indices
                non_zero_mask = param_np != 0
                non_zero_values = param_np[non_zero_mask]
                non_zero_indices = np.where(non_zero_mask.reshape(-1))[0]

                # Store format, shape, indices and values
                param_data = {
                    "format": "sparse",
                    "shape": param_np.shape,
                    "dtype": str(param_np.dtype),
                    "indices": non_zero_indices,
                    "values": non_zero_values,
                }

                # Calculate storage size
                indices_bytes = len(non_zero_indices) * 4  # int32 indices
                values_bytes = len(non_zero_values) * 4  # float32 values
                param_bytes = indices_bytes + values_bytes

                if param_bytes < original_bytes:  # Only use sparse if smaller
                    # Save as sparse
                    compressed_bytes += param_bytes

                    # Save indices and values to the zip file
                    zf.writestr(f"{name}/format.txt", "sparse")
                    zf.writestr(f"{name}/shape.txt", str(param_np.shape))
                    zf.writestr(f"{name}/dtype.txt", str(param_np.dtype))

                    # Save with np.save which is more efficient than pickle
                    indices_buf = io.BytesIO()
                    np.save(indices_buf, non_zero_indices)
                    zf.writestr(f"{name}/indices.npy", indices_buf.getvalue())

                    values_buf = io.BytesIO()
                    np.save(values_buf, non_zero_values)
                    zf.writestr(f"{name}/values.npy", values_buf.getvalue())

                    # Print info for very sparse or large layers
                    if param_size > 100000 and sparsity > 80:
                        print(f"  {name}: {sparsity:.1f}% sparse, saved as sparse")
                        print(
                            f"    Original: {original_bytes/1024/1024:.2f} MB, "
                            f"Sparse: {param_bytes/1024/1024:.2f} MB"
                        )
                    continue

            # For parameters that aren't sparse enough or where sparse isn't smaller,
            # we'll use numpy's compressed format which works well with repeated zeros
            compressed_buf = io.BytesIO()
            np.savez_compressed(compressed_buf, data=param_np)
            compressed_data = compressed_buf.getvalue()
            compressed_bytes += len(compressed_data)

            # Store as compressed numpy
            zf.writestr(f"{name}/format.txt", "compressed")
            zf.writestr(f"{name}/data.npz", compressed_data)

            # Print info for large layers
            if param_size > 1000000:
                print(f"  {name}: {sparsity:.1f}% sparse, saved as compressed")
                print(
                    f"    Original: {original_bytes/1024/1024:.2f} MB, "
                    f"Compressed: {len(compressed_data)/1024/1024:.2f} MB"
                )

    # Write the zip file to disk
    buffer.seek(0)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())

    # Get actual file size
    actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    dense_size_mb = dense_bytes / (1024 * 1024)

    return actual_size_mb


def load_whisper_optimized(model_path, device="cpu"):
    """
    Load a Whisper model that was saved in optimized format.

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

        # Get all parameter names by parsing file paths
        param_names = set()
        for file_path in file_list:
            if "/" in file_path:
                param_name = file_path.split("/")[0]
                param_names.add(param_name)

        # Process each parameter
        for param_name in param_names:
            # Determine the storage format
            if f"{param_name}/format.txt" in file_list:
                with zf.open(f"{param_name}/format.txt") as f:
                    format_type = f.read().decode("utf-8").strip()

                if format_type == "sparse":
                    # Load shape
                    with zf.open(f"{param_name}/shape.txt") as f:
                        shape_str = f.read().decode("utf-8").strip()
                        # Parse the shape tuple from string, e.g., "(10, 20)" -> (10, 20)
                        shape = eval(shape_str)

                    # Load dtype
                    with zf.open(f"{param_name}/dtype.txt") as f:
                        dtype_str = f.read().decode("utf-8").strip()
                        # Get numpy dtype from string
                        dtype = np.dtype(dtype_str)

                    # Load indices and values
                    with zf.open(f"{param_name}/indices.npy") as f:
                        indices = np.load(io.BytesIO(f.read()))

                    with zf.open(f"{param_name}/values.npy") as f:
                        values = np.load(io.BytesIO(f.read()))

                    # Reconstruct the dense array
                    dense_array = np.zeros(np.prod(shape), dtype=dtype)
                    dense_array[indices] = values
                    dense_array = dense_array.reshape(shape)

                    # Convert to PyTorch tensor
                    param_tensor = torch.tensor(dense_array, dtype=torch.float32, device=device)

                elif format_type == "compressed":
                    # Load compressed numpy array
                    with zf.open(f"{param_name}/data.npz") as f:
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
    state_dict = load_whisper_optimized(model_path, device)

    # Load state dict into the model
    print("Loading optimized weights into model...")
    base_model.load_state_dict(state_dict)

    return base_model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Prune Whisper Small model and save using optimized storage"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./pruned_models", help="Directory to save pruned model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="whisper_small_pruned",
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

    # Define pruning configuration with the specified percentages
    base_pruning_config = {
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

    # Option to increase pruning for better compression
    if args.increase_pruning:
        pruning_config = {
            "encoder_ffn": 80,  # Increased from 50%
            "decoder_ffn_first": 60,  # Increased from 25%
            "decoder_ffn_middle": 75,  # Increased from 45%
            "decoder_ffn_last": 70,  # Increased from 30%
            "encoder_self_attn": 70,  # Increased from 40%
            "decoder_self_attn": 75,  # Increased from 50%
            "decoder_cross_attn": 75,  # Increased from 45%
            "layer_norm": 0,  # Keep unchanged (sensitive)
            "token_embeddings": 60,  # Increased from 25%
            "positional_embeddings": 0,  # Keep unchanged (sensitive)
            "conv_layers": 60,  # Increased from 30%
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

    # Step 2: Apply pruning
    print("\nApplying pruning...")
    start_time = time.time()
    model = apply_custom_l1_pruning(model, pruning_config, make_permanent=True)
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

    # Step 4: Save model using optimized storage
    output_path = os.path.join(args.output_dir, f"{args.model_name}.zip")
    print("\nSaving model...")
    start_time = time.time()
    sparse_size_mb = save_whisper_optimized(model, output_path)
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
