import re

import pandas as pd
from tabulate import tabulate
from transformers import WhisperModel


def analyze_whisper_model(model_name="openai/whisper-small", verbose=True):
    """
    Comprehensive analysis of a Whisper model architecture.

    Args:
        model_name: The Whisper model name (default: "openai/whisper-small")
        verbose: Whether to print detailed information during analysis

    Returns:
        tuple: (model, component_info, summary_tables, stats)
    """
    # Load the Whisper model
    print(f"Loading {model_name}...")
    model = WhisperModel.from_pretrained(model_name)

    # Get basic model info
    encoder_layers = len(model.encoder.layers)
    decoder_layers = len(model.decoder.layers)
    print(f"Model loaded: Encoder has {encoder_layers} layers, Decoder has {decoder_layers} layers")

    # Get total parameters count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Step 1: Create initial maps of all parameters and modules
    param_info = {}  # Maps parameter names to info about them
    module_info = {}  # Maps module names to info about them

    # First, track all parameters
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        param_name = name.rsplit(".", 1)[1] if "." in name else name

        # Store parameter info
        param_info[name] = {
            "shape": list(param.shape),
            "size": param.numel(),
            "module": module_name,
            "is_bias": "bias" in param_name,
            "is_encoder": "encoder" in module_name and "encoder_attn" not in module_name,
            "is_decoder": "decoder" in module_name or "encoder_attn" in module_name,
        }

    # Then, map all modules
    for name, module in model.named_modules():
        if name == "":
            continue  # Skip root module

        # Store basic module info
        module_info[name] = {
            "type": module.__class__.__name__,
            "params": {},
            "param_count": 0,
            "component_type": "",
            "layer_num": -1,
            "is_encoder": "encoder" in name and "encoder_attn" not in name,
            "is_decoder": "decoder" in name or "encoder_attn" in name,
        }

    # Step 2: Assign parameters to modules
    for param_name, info in param_info.items():
        module_name = info["module"]
        if module_name in module_info:
            module_info[module_name]["params"][param_name] = info
            module_info[module_name]["param_count"] += info["size"]

    # Look for specific modules that might not be captured normally
    all_param_names = list(param_info.keys())

    # Specifically look for proj_out
    proj_out_params = [name for name in all_param_names if "proj_out" in name]
    if proj_out_params:
        if verbose:
            print(f"Found proj_out parameters: {proj_out_params}")

        # Add a special module if necessary
        for param_name in proj_out_params:
            module_name = param_name.rsplit(".", 1)[0]
            if module_name not in module_info:
                module_info[module_name] = {
                    "type": "OutputProjection",
                    "params": {param_name: param_info[param_name]},
                    "param_count": param_info[param_name]["size"],
                    "component_type": "Final Output Projection",
                    "layer_num": -1,
                    "is_encoder": False,
                    "is_decoder": True,
                }
            else:
                # Update existing module
                module_info[module_name]["component_type"] = "Final Output Projection"

    # Step 3: Identify components and categorize modules
    for name, info in module_info.items():
        # Extract layer number if present
        layer_match = re.search(r"\.layers\.(\d+)", name)
        if layer_match:
            info["layer_num"] = int(layer_match.group(1))

        # Determine component type based on patterns in the name
        if "self_attn" in name:
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                info["component_type"] = "Self-Attention QKV Projection"
            elif "out_proj" in name:
                info["component_type"] = "Self-Attention Output Projection"
            elif "layer_norm" in name:
                info["component_type"] = "Layer Normalization"
            else:
                info["component_type"] = "Self-Attention"

        elif "cross_attn" in name or "encoder_attn" in name:
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                info["component_type"] = "Cross-Attention QKV Projection"
            elif "out_proj" in name:
                info["component_type"] = "Cross-Attention Output Projection"
            elif "layer_norm" in name:
                info["component_type"] = "Layer Normalization"
            else:
                info["component_type"] = "Cross-Attention"

        elif "feed_forward" in name or "fc" in name:
            if "fc1" in name:
                info["component_type"] = "FFN Input Projection"
            elif "fc2" in name:
                info["component_type"] = "FFN Output Projection"
            elif "layer_norm" in name:
                info["component_type"] = "Layer Normalization"
            else:
                info["component_type"] = "FFN"

        elif "layer_norm" in name or "layernorm" in name:
            info["component_type"] = "Layer Normalization"

        elif "conv" in name:
            info["component_type"] = "Convolutional Layer"

        elif "embed_positions" in name:
            info["component_type"] = "Positional Embedding"

        elif "embed_tokens" in name:
            info["component_type"] = "Token Embedding"

        elif "proj_out" in name:
            info["component_type"] = "Final Output Projection"

        elif "proj" in name and not any(
            x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]
        ):
            if "decoder" in name:
                info["component_type"] = "Final Output Projection"
            else:
                info["component_type"] = "Output Projection"

        elif not info["component_type"]:  # If still not assigned
            info["component_type"] = "Other"

    # Step 4: Calculate statistics and build summary tables

    # Remove modules with no parameters for cleaner analysis
    module_info = {k: v for k, v in module_info.items() if v["param_count"] > 0}

    # Create a flattened list of all modules with their details
    all_modules = []
    for name, info in module_info.items():
        # Calculate bias and weight params
        bias_params = sum(p["size"] for p in info["params"].values() if p["is_bias"])
        weight_params = info["param_count"] - bias_params

        all_modules.append(
            {
                "name": name,
                "type": info["type"],
                "component_type": info["component_type"],
                "layer_num": info["layer_num"],
                "total_params": info["param_count"],
                "weight_params": weight_params,
                "bias_params": bias_params,
                "is_encoder": info["is_encoder"],
                "is_decoder": info["is_decoder"],
            }
        )

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_modules)

    # Create summary tables
    summary_tables = {}

    # 1. Encoder vs Decoder
    encoder_params = df[df["is_encoder"]]["total_params"].sum()
    decoder_params = df[df["is_decoder"]]["total_params"].sum()
    # Handle any overlapping or uncategorized parameters
    other_params = total_params - encoder_params - decoder_params

    encoder_decoder_table = [
        ["Component", "Parameters", "Percentage"],
        ["Encoder", f"{encoder_params:,}", f"{encoder_params/total_params*100:.2f}%"],
        ["Decoder", f"{decoder_params:,}", f"{decoder_params/total_params*100:.2f}%"],
    ]
    if other_params > 0:
        encoder_decoder_table.append(
            ["Other", f"{other_params:,}", f"{other_params/total_params*100:.2f}%"]
        )
    encoder_decoder_table.append(["Total", f"{total_params:,}", "100.00%"])

    summary_tables["encoder_decoder"] = encoder_decoder_table

    # 2. Component Types
    component_summary = (
        df.groupby("component_type")
        .agg({"total_params": "sum", "weight_params": "sum", "bias_params": "sum", "name": "count"})
        .sort_values("total_params", ascending=False)
    )
    component_summary["percentage"] = component_summary["total_params"] / total_params * 100

    component_table = [["Component Type", "Parameters", "Count", "Percentage"]]
    for idx, row in component_summary.iterrows():
        component_table.append(
            [idx, f"{row['total_params']:,}", row["name"], f"{row['percentage']:.2f}%"]
        )
    component_table.append(
        ["Total", f"{total_params:,}", sum(component_summary["name"]), "100.00%"]
    )

    summary_tables["component_types"] = component_table

    # 3. Aggregated Component Types
    # Create a mapping for aggregated component types
    component_mapping = {
        "Self-Attention QKV Projection": "Attention Projections",
        "Cross-Attention QKV Projection": "Attention Projections",
        "Self-Attention Output Projection": "Attention Projections",
        "Cross-Attention Output Projection": "Attention Projections",
        "FFN Input Projection": "FFN Projections",
        "FFN Output Projection": "FFN Projections",
        "Self-Attention": "Attention Core Components",
        "Cross-Attention": "Attention Core Components",
        "Final Output Projection": "Final Output Projection",
        "Layer Normalization": "Normalization",
        "Convolutional Layer": "Convolutional Frontend",
        "Token Embedding": "Embeddings",
        "Positional Embedding": "Embeddings",
        "Output Projection": "Other Projections",
        "Other": "Other",
    }

    # Apply the mapping
    df["aggregated_type"] = df["component_type"].map(component_mapping)

    agg_summary = (
        df.groupby("aggregated_type")
        .agg({"total_params": "sum", "name": "count"})
        .sort_values("total_params", ascending=False)
    )
    agg_summary["percentage"] = agg_summary["total_params"] / total_params * 100

    agg_table = [["Aggregated Component Type", "Parameters", "Count", "Percentage"]]
    for idx, row in agg_summary.iterrows():
        agg_table.append(
            [idx, f"{row['total_params']:,}", row["name"], f"{row['percentage']:.2f}%"]
        )
    agg_table.append(["Total", f"{total_params:,}", sum(agg_summary["name"]), "100.00%"])

    summary_tables["aggregated_components"] = agg_table

    # 4. Encoder Component Types
    encoder_df = df[df["is_encoder"]]
    encoder_summary = (
        encoder_df.groupby("component_type")
        .agg({"total_params": "sum", "name": "count"})
        .sort_values("total_params", ascending=False)
    )
    encoder_summary["percentage"] = encoder_summary["total_params"] / encoder_params * 100

    encoder_table = [["Encoder Component Type", "Parameters", "Count", "Percentage"]]
    for idx, row in encoder_summary.iterrows():
        encoder_table.append(
            [idx, f"{row['total_params']:,}", row["name"], f"{row['percentage']:.2f}%"]
        )
    encoder_table.append(["Total", f"{encoder_params:,}", sum(encoder_summary["name"]), "100.00%"])

    summary_tables["encoder_components"] = encoder_table

    # 5. Decoder Component Types
    decoder_df = df[df["is_decoder"]]
    decoder_summary = (
        decoder_df.groupby("component_type")
        .agg({"total_params": "sum", "name": "count"})
        .sort_values("total_params", ascending=False)
    )
    decoder_summary["percentage"] = decoder_summary["total_params"] / decoder_params * 100

    decoder_table = [["Decoder Component Type", "Parameters", "Count", "Percentage"]]
    for idx, row in decoder_summary.iterrows():
        decoder_table.append(
            [idx, f"{row['total_params']:,}", row["name"], f"{row['percentage']:.2f}%"]
        )
    decoder_table.append(["Total", f"{decoder_params:,}", sum(decoder_summary["name"]), "100.00%"])

    summary_tables["decoder_components"] = decoder_table

    # 6. Weights vs Biases
    bias_params = df["bias_params"].sum()
    weight_params = df["weight_params"].sum()

    weight_bias_table = [
        ["Parameter Type", "Count", "Percentage"],
        ["Weights", f"{weight_params:,}", f"{weight_params/total_params*100:.2f}%"],
        ["Biases", f"{bias_params:,}", f"{bias_params/total_params*100:.2f}%"],
        ["Total", f"{total_params:,}", "100.00%"],
    ]

    summary_tables["weight_bias"] = weight_bias_table

    # 7. Layer Position
    layer_df = df[df["layer_num"] >= 0]

    # Define layer positions based on quartiles
    total_layers = max(encoder_layers, decoder_layers)
    quartile = total_layers // 4

    def get_position(layer_num):
        if layer_num < quartile:
            return "Early Layers"
        elif layer_num < 2 * quartile:
            return "Middle-Early Layers"
        elif layer_num < 3 * quartile:
            return "Middle-Late Layers"
        else:
            return "Late Layers"

    layer_df["position"] = layer_df["layer_num"].apply(get_position)

    position_summary = (
        layer_df.groupby("position")
        .agg({"total_params": "sum", "name": "count"})
        .sort_values("position")
    )
    position_summary["percentage"] = (
        position_summary["total_params"] / layer_df["total_params"].sum() * 100
    )

    position_table = [["Layer Position", "Parameters", "Count", "Percentage"]]
    for idx, row in position_summary.iterrows():
        position_table.append(
            [idx, f"{row['total_params']:,}", row["name"], f"{row['percentage']:.2f}%"]
        )
    position_table.append(
        ["Total", f"{layer_df['total_params'].sum():,}", sum(position_summary["name"]), "100.00%"]
    )

    summary_tables["layer_position"] = position_table

    # Step 5: Create a comprehensive stats dictionary
    stats = {
        "total_params": total_params,
        "encoder_params": encoder_params,
        "encoder_percentage": encoder_params / total_params * 100,
        "decoder_params": decoder_params,
        "decoder_percentage": decoder_params / total_params * 100,
        "other_params": other_params,
        "other_percentage": other_params / total_params * 100 if total_params > 0 else 0,
        "weight_params": weight_params,
        "weight_percentage": weight_params / total_params * 100,
        "bias_params": bias_params,
        "bias_percentage": bias_params / total_params * 100,
        "component_counts": {
            comp: count for comp, count in zip(component_summary.index, component_summary["name"])
        },
        "component_params": {
            comp: params
            for comp, params in zip(component_summary.index, component_summary["total_params"])
        },
        "component_percentages": {
            comp: pct for comp, pct in zip(component_summary.index, component_summary["percentage"])
        },
        "data_frame": df,
        "module_info": module_info,
        "param_info": param_info,
    }

    # Print tabular summaries if requested
    if verbose:
        print("\n" + "=" * 80)
        print("ENCODER VS DECODER")
        print(tabulate(encoder_decoder_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("COMPONENT TYPE SUMMARY")
        print(tabulate(component_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("AGGREGATED COMPONENT SUMMARY")
        print(tabulate(agg_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("ENCODER COMPONENTS")
        print(tabulate(encoder_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("DECODER COMPONENTS")
        print(tabulate(decoder_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("WEIGHTS VS BIASES")
        print(tabulate(weight_bias_table, headers="firstrow", tablefmt="grid"))

        print("\n" + "=" * 80)
        print("LAYER POSITION")
        print(tabulate(position_table, headers="firstrow", tablefmt="grid"))

    return model, module_info, summary_tables, stats


def save_model_analysis(stats, tables, filename="whisper_analysis_results.csv"):
    """Save the analysis results to CSV files"""

    # Create DataFrame for component summary
    components_df = pd.DataFrame(
        {
            "Component": list(stats["component_params"].keys()),
            "Parameters": list(stats["component_params"].values()),
            "Percentage": [
                stats["component_percentages"][comp] for comp in stats["component_params"].keys()
            ],
            "Count": [stats["component_counts"][comp] for comp in stats["component_params"].keys()],
        }
    )

    # Save to CSV
    components_df.to_csv(f"components_{filename}", index=False)
    print(f"Component analysis saved to components_{filename}")

    # Save main stats to CSV
    main_stats = pd.DataFrame(
        [
            {
                "Total_Parameters": stats["total_params"],
                "Encoder_Parameters": stats["encoder_params"],
                "Encoder_Percentage": stats["encoder_percentage"],
                "Decoder_Parameters": stats["decoder_params"],
                "Decoder_Percentage": stats["decoder_percentage"],
                "Weight_Parameters": stats["weight_params"],
                "Weight_Percentage": stats["weight_percentage"],
                "Bias_Parameters": stats["bias_params"],
                "Bias_Percentage": stats["bias_percentage"],
            }
        ]
    )

    main_stats.to_csv(f"summary_{filename}", index=False)
    print(f"Summary statistics saved to summary_{filename}")

    return True


if __name__ == "__main__":
    try:
        # You can change this to any Whisper model variant
        model_name = "openai/whisper-small"

        # Run the analysis
        model, module_info, tables, stats = analyze_whisper_model(model_name, verbose=True)

        # Save results to CSV
        save_model_analysis(stats, tables)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error during analysis: {e}")
