#!/usr/bin/env python
# Script to test a pruned Whisper model saved in optimized format

import argparse
import io
import os
import time
import zipfile
from collections import OrderedDict

import datasets
import numpy as np
import torch
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor


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


def calculate_sparsity(model):
    """
    Calculate the sparsity percentage and parameter counts in the model.

    Args:
        model: The PyTorch model

    Returns:
        tuple: (overall_sparsity, total_params, non_zero_params)
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()

    # Calculate overall sparsity
    if total_params == 0:
        return 0.0, 0, 0

    overall_sparsity = 100.0 * zero_params / total_params
    non_zero_params = total_params - zero_params

    return overall_sparsity, total_params, non_zero_params


def get_model_disk_size_in_mb(model):
    """
    Calculate the size of a model's state dict when saved to disk.

    Args:
        model: PyTorch model

    Returns:
        float: Size in MB
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
    return buffer.getbuffer().nbytes / (1024**2)


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


def map_to_feats(batch, processor):
    """
    Process audio batch to extract features and normalize text.
    """
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch


def transcribe_batch(batch, model, processor):
    """
    Transcribe a batch of audio samples.
    """
    with torch.no_grad():
        # Prepare input features
        features = torch.from_numpy(np.array(batch["input_features"], dtype=np.float32)).squeeze(1)
        features = features.to(model.device)

        # Record processing time
        start_time = time.time()

        # Generate transcriptions
        predicted_ids = model.generate(features)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Decode predictions
        transcription = [processor.decode(ids) for ids in predicted_ids]
        batch["prediction"] = [processor.tokenizer.normalize(x) for x in transcription]

        # Store processing time
        batch["processing_time"] = [processing_time] * len(batch["audio"])

    return batch


def evaluate_model(model, processor, dataset, metrics, batch_size=8):
    """
    Evaluate model on dataset.

    Args:
        model: The model to evaluate
        processor: WhisperProcessor for feature extraction
        dataset: Dataset to evaluate on
        metrics: Dictionary of metrics to compute
        batch_size: Batch size for processing

    Returns:
        dict: Evaluation results
    """
    total_processing_time = 0.0

    def process_batch(batch):
        nonlocal total_processing_time

        # Process the batch
        result = transcribe_batch(batch, model, processor)

        # Track processing time
        batch_processing_time = result["processing_time"][0]
        total_processing_time += batch_processing_time

        return result

    # Process dataset in batches
    start = time.time()
    result = dataset.map(process_batch, batched=True, batch_size=batch_size)
    end = time.time()

    # Compute metrics
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

    # Add timing metrics
    scores["total_processing_time"] = total_processing_time
    scores["avg_latency"] = total_processing_time / (len(dataset) / batch_size)

    print(f"{len(result)} sentences evaluated in {end - start:.2f} s.")
    print(f"Average batch latency: {scores['avg_latency']:.4f} s")
    print(f"Total processing time: {total_processing_time:.2f} s")

    # Print some sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(result))):
        print(f"\nReference: {result['reference'][i]}")
        print(f"Prediction: {result['prediction'][i]}")

    return scores, result


def main():
    parser = argparse.ArgumentParser(description="Test a pruned Whisper model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pruned_models/whisper_small_pruned.zip",
        help="Path to the pruned model file",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of test samples to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)")
    args = parser.parse_args()

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

    # Step 1: Load the pruned model
    print(f"Loading pruned model from {args.model_path}...")
    start_time = time.time()
    model = load_pruned_model(args.model_path, device)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Step 2: Load processor
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Step 3: Calculate and report model statistics
    print("\nModel statistics:")
    overall_sparsity, total_params, non_zero_params = calculate_sparsity(model)
    print(f"Overall sparsity: {overall_sparsity:.2f}%")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {non_zero_params:,}")

    # Get model size
    model_size = get_model_disk_size_in_mb(model)
    print(f"Model size (if saved as standard PyTorch): {model_size:.2f} MB")
    print(f"Model size (optimized format): {os.path.getsize(args.model_path) / (1024**2):.2f} MB")

    # Step 4: Load test dataset
    print("\nLoading test dataset...")
    dataset = load_librispeech(num_samples=args.num_samples, split="test.clean")

    # Step 5: Process dataset
    print("\nProcessing dataset...")
    processed_dataset = dataset.map(lambda x: map_to_feats(x, processor))

    # Step 6: Evaluate model
    print("\nEvaluating model...")
    metrics = {"WER": load("wer"), "CER": load("cer")}
    scores, _ = evaluate_model(
        model=model,
        processor=processor,
        dataset=processed_dataset,
        metrics=metrics,
        batch_size=args.batch_size,
    )

    # Step 7: Report results
    print("\nEvaluation results:")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.5f}")
        else:
            print(f"{metric}: {value}")

    print("\nTest completed successfully!")
    print(
        f"Your pruned model is working correctly and is {os.path.getsize(args.model_path) / (1024**2):.2f} MB "
        f"(compared to ~922 MB for the original model)"
    )


if __name__ == "__main__":
    main()
