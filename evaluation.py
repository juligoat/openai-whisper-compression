import json
import logging
import os
import time
from typing import Any, Dict, Tuple
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=2, num_warmup=2):
    """
    Evaluate a Whisper model on a dataset.

    Args:
        model: The model to evaluate
        processor: Whisper processor for tokenization
        dataset: Dataset to evaluate on
        metrics: Dictionary of metrics to compute
        memory_tracker: Tracker for memory usage
        split: Dataset split name (for logging)
        batch_size: Batch size for evaluation
        num_warmup: Number of warmup batches to run before measuring

    Returns:
        Tuple of (metrics dict, transcriptions dict)
    """
    from data_utils import transcribe_batch

    # Perform warmup runs if specified
    if num_warmup > 0:
        logger.info(f"Performing {num_warmup} warmup runs...")
        
        # Create a small subset for warmup
        warmup_size = min(num_warmup * batch_size, len(dataset))
        warmup_dataset = dataset.select(range(warmup_size))
        
        # Create a dummy memory tracker for warmup to avoid skewing measurement
        class DummyTracker:
            def log_memory(self, *args, **kwargs):
                pass
        
        dummy_tracker = DummyTracker()
        
        # Run warmup batches with progress bar
        warmup_progress = tqdm(total=warmup_size, desc=f"Warmup on {split}")
        
        def warmup_batch(batch):
            # Process the batch without collecting results
            _ = transcribe_batch(batch, model, processor, dummy_tracker, "warmup", 0)
            warmup_progress.update(len(batch))
            return batch
        
        # Execute warmup
        _ = warmup_dataset.map(warmup_batch, batched=True, batch_size=batch_size)
        warmup_progress.close()
        
        # Make sure GPU operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        logger.info("Warmup complete")

    total_processing_time = 0.0
    total_audio_duration = 0.0
    batch_counter = 0

    # Create progress bar for the actual evaluation
    evaluation_progress = tqdm(total=len(dataset), desc=f"Evaluating on {split}")

    def process_batch(batch):
        nonlocal batch_counter, total_processing_time, total_audio_duration
        # Process the batch and update the cumulative totals
        result = transcribe_batch(batch, model, processor, memory_tracker, split, batch_counter)
        # Each sample in the batch has the same processing time and audio duration
        batch_processing_time = result["processing_time"][0]
        batch_audio_duration = result["audio_duration"][0]
        total_processing_time += batch_processing_time
        total_audio_duration += batch_audio_duration
        batch_counter += 1
        
        # Update progress bar
        evaluation_progress.update(len(batch))
        
        # Optionally update progress bar with current metrics
        if batch_counter % 5 == 0 and total_audio_duration > 0:
            current_rtf = total_processing_time / total_audio_duration
            evaluation_progress.set_description(
                f"Evaluating on {split} - RTF: {current_rtf:.4f}"
            )
            
        return result

    start = time.time()
    result = dataset.map(process_batch, batched=True, batch_size=batch_size)
    end = time.time()
    
    # Close progress bar
    evaluation_progress.close()

    # Calculate overall RTF from the accumulated totals
    overall_rtf = (
        total_processing_time / total_audio_duration if total_audio_duration > 0 else float("inf")
    )
    logger.info(f"Overall RTF: {overall_rtf:.6f}")

    # Compute metrics (e.g., WER, CER)
    scores = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["WER", "CER"]:
            score = 100 * metric.compute(
                references=result["reference"], predictions=result["prediction"]
            )
            scores[metric_name] = score
            logger.info(f"{metric_name}: {score:.5f}")

    scores["RTF"] = overall_rtf

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

    logger.info(f"{len(result)} sentences evaluated in {end - start:.2f} s.")
    return scores, {"references": result["reference"], "predictions": result["prediction"]}


def save_evaluation_results(model_name, split, results, transcriptions, save_path):
    """
    Save evaluation results to disk.

    Args:
        model_name: Name of the model
        split: Dataset split
        results: Evaluation results dictionary
        transcriptions: Transcriptions dictionary
        save_path: Path to save results
    """
    # Save metrics
    metrics_path = os.path.join(save_path, f"{model_name}_{split}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save transcriptions
    transcriptions_path = os.path.join(save_path, f"{model_name}_{split}_transcriptions.json")
    with open(transcriptions_path, "w") as f:
        json.dump(transcriptions, f, indent=2)

    logger.info(f"Results saved to {metrics_path} and {transcriptions_path}")


def print_evaluation_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a summary of evaluation results.

    Args:
        results: Dictionary of evaluation results
    """
    logger.info("\nFinal Results Summary:")
    for run_name, run_results in results.items():
        logger.info(f"\n{run_name}:")
        logger.info(f"Model Size: {run_results['model_size_mb']:.2f} MB")
        logger.info("Metrics:")
        for metric_name, value in run_results["metrics"].items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Add CPU metrics if they exist
        if "avg_cpu_percent" in run_results["metrics"]:
            logger.info(f"  Avg CPU: {run_results['metrics']['avg_cpu_percent']:.4f}%")
            logger.info(f"  Peak CPU: {run_results['metrics']['peak_cpu_percent']:.4f}%")