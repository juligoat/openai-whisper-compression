import argparse
import io
import time
from functools import partial

import datasets
import numpy as np
import torch
from evaluate import load
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def map_to_feats(batch, processor):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch


def transcribe_batch(batch, model, processor):
    with torch.no_grad():
        features = torch.from_numpy(np.array(batch["input_features"], dtype=np.float32)).squeeze(1)

        # Get total audio duration (in seconds)
        audio_durations = [len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]]
        total_audio_duration = sum(audio_durations)

        # Measure processing time
        start_time = time.time()
        predicted_ids = model.generate(features.to(model.device))
        processing_time = time.time() - start_time

        # Calculate RTF
        rtf = processing_time / total_audio_duration

    transcription = [processor.decode(ids) for ids in predicted_ids]
    batch["prediction"] = [processor.tokenizer.normalize(x) for x in transcription]
    batch["rtf"] = [rtf] * len(batch["audio"])
    return batch


def evaluate_model(model, processor, dataset, metrics, batch_size=2):
    map_fn = partial(transcribe_batch, model=model, processor=processor)
    start = time.time()
    result = dataset.map(map_fn, batched=True, batch_size=batch_size)
    end = time.time()

    # Compute scores for each metric
    scores = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["WER", "CER"]:
            score = 100 * metric.compute(
                references=result["reference"], predictions=result["prediction"]
            )
            scores[metric_name] = score
            print(f"{metric_name}: {score:.2f}")

    # Calculate average RTF
    avg_rtf = sum(result["rtf"]) / len(result["rtf"])
    scores["RTF"] = avg_rtf
    print(f"RTF: {avg_rtf:.3f}")

    print(f"{len(result)} sentences evaluated in {end - start:.2f} s.")

    # Also return transcriptions
    transcriptions = {"references": result["reference"], "predictions": result["prediction"]}

    return scores, transcriptions


def load_librispeech(num_samples=None, split="test.clean"):
    """
    Load LibriSpeech clean/other data.
    """
    if num_samples:
        # Stream partial dataset
        stream_dataset = datasets.load_dataset("librispeech_asr", split=split, streaming=True)
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
    print(f"Total audio duration: {total_hours:.2f} hours")
    return dataset


def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_in_bytes = buffer.getbuffer().nbytes
    size_in_mb = size_in_bytes / (1024**2)
    return size_in_mb


def main():
    parser = argparse.ArgumentParser(description="Baseline Whisper Evaluation")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="The name of the trained Model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="The batch size to use for evaluation."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="The device to use for evaluation."
    )
    parser.add_argument("--save_path", type=str, default="results", help="Path to save results")
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    import json
    import os

    os.makedirs(args.save_path, exist_ok=True)

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("USING CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("USING MPS")
        else:
            device = torch.device("cpu")
            print("USING CPU")
    else:
        device = torch.device(args.device)

    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.config.forced_decoder_ids = None
    processor = WhisperProcessor.from_pretrained(args.model)

    # Load and process test data
    dataset = load_librispeech(
        num_samples=2939, split="test.other"
    )  # Adjust number of samples as needed
    processed_test_data = dataset.map(lambda x: map_to_feats(x, processor))

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    print("Evaluating baseline float model:")
    float_scores, float_transcriptions = evaluate_model(
        model, processor, processed_test_data, metrics, args.batch_size
    )
    float_model_size = get_model_disk_size_in_mb(model)
    print(f"Float model size: {float_model_size:.2f} MB")

    # Save metrics and transcriptions
    float_metrics = {
        "metrics": float_scores,
        "model_size_mb": float_model_size,
        "model_type": "float",
        "model_name": args.model,
    }

    with open(os.path.join(args.save_path, "float_metrics.json"), "w") as f:
        json.dump(float_metrics, f, indent=2)

    with open(os.path.join(args.save_path, "float_transcriptions.json"), "w") as f:
        json.dump(float_transcriptions, f, indent=2)


if __name__ == "__main__":
    main()
