import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
import torch
from jiwer import cer, wer
from tqdm.notebook import tqdm
from whisper_quantization_pkg.config import ProjectConfig


class ModelProfiler:
    """Class to handle model profiling and metrics"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = ProjectConfig()
        self.metrics_history: List[Dict[str, float]] = []
        self.size_details: Dict[str, Any] = {}

    def measure_model_size(self, model: torch.nn.Module) -> float:
        """Measure model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def get_detailed_model_size(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get detailed size information about the model components"""
        param_size = 0
        buffer_size = 0
        layer_info = {}

        # Parameters
        total_params = 0
        for name, param in model.named_parameters():
            num_params = param.nelement()
            total_params += num_params
            size = num_params * param.element_size()
            param_size += size
            layer_info[name] = {
                "size_mb": size / 1024**2,
                "num_params": num_params,
                "type": "parameter",
            }

        # Buffers
        for name, buffer in model.named_buffers():
            size = buffer.nelement() * buffer.element_size()
            buffer_size += size
            layer_info[name] = {
                "size_mb": size / 1024**2,
                "num_elements": buffer.nelement(),
                "type": "buffer",
            }

        total_size = (param_size + buffer_size) / 1024**2

        self.size_details = {
            "total_size_mb": total_size,
            "param_size_mb": param_size / 1024**2,
            "buffer_size_mb": buffer_size / 1024**2,
            "total_params": total_params,
            "layer_info": layer_info,
        }

        return self.size_details

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics during evaluation

        Args:
            metrics: Dictionary of metric names and values
        """
        self.metrics_history.append(metrics)

    def save_metrics(self, filename: str) -> None:
        """Save logged metrics

        Args:
            filename: Name of the file to save metrics to
        """
        if not self.metrics_history:
            print("No metrics to save")
            return

        full_path = self.config.dirs["results"] / filename
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(full_path, index=False)
        print(f"Metrics saved to {full_path}")

    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024**2
        return memory_mb

    def print_size_analysis(self):
        """Print detailed size analysis"""
        if not self.size_details:
            print("No size analysis available. Run get_detailed_model_size first.")
            return

        print(f"Model Size Analysis for {self.model_name}")
        print("-" * 50)
        print(f"Total Model Size: {self.size_details['total_size_mb']:.2f} MB")
        print(f"Parameter Size: {self.size_details['param_size_mb']:.2f} MB")
        print(f"Buffer Size: {self.size_details['buffer_size_mb']:.2f} MB")
        print(f"Total Parameters: {self.size_details['total_params']:,}")
        print("\nLayer-by-Layer Breakdown:")
        print("-" * 50)

        # Sort layers by size
        sorted_layers = sorted(
            self.size_details["layer_info"].items(), key=lambda x: x[1]["size_mb"], reverse=True
        )

        for name, info in sorted_layers:
            if info["type"] == "parameter":
                print(f"{name}:")
                print(f"  Size: {info['size_mb']:.2f} MB")
                print(f"  Parameters: {info['num_params']:,}")
            else:
                print(f"Buffer {name}:")
                print(f"  Size: {info['size_mb']:.2f} MB")
            print("-" * 30)

    @staticmethod
    def calculate_error_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
        """Calculate WER and CER metrics with comprehensive normalization

        Args:
            reference: Reference text
            hypothesis: Hypothesis text

        Returns:
            Tuple of (WER, CER) scores
        """

        def normalize_text(text: str) -> str:
            # Convert to lowercase
            text = text.lower()
            # Remove all punctuation
            text = "".join(c for c in text if c.isalnum() or c.isspace())
            # Normalize whitespace (replace multiple spaces with single space)
            text = " ".join(text.split())
            return text

        reference_norm = normalize_text(reference)
        hypothesis_norm = normalize_text(hypothesis)

        return wer(reference_norm, hypothesis_norm), cer(reference_norm, hypothesis_norm)


class WhisperEvaluator:
    """Class to handle Whisper model evaluation"""

    def __init__(self, model, processor, device: torch.device, profiler: ModelProfiler):
        self.model = model
        self.processor = processor
        self.device = device
        self.profiler = profiler
        self.config = ProjectConfig()
        self.model.to(device)
        self.model.eval()

    def process_audio(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process audio input for model

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary with processed features and attention mask
        """
        processed = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,  # Explicitly request attention mask
        )
        return {
            "input_features": processed.input_features.to(self.device),
            "attention_mask": processed.attention_mask.to(self.device)
            if processed.attention_mask is not None
            else None,
        }

    def evaluate_sample(self, audio: torch.Tensor, reference: str) -> Dict[str, Any]:
        """Evaluate a single audio sample"""
        inputs = self.process_audio(audio)

        # Measure inference
        mem_before = self.profiler.get_memory_usage()
        start_time = time.time()

        with torch.no_grad():
            # Generate with proper handling of attention mask
            predicted_ids = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs["attention_mask"],
                language="en",  # Specify English
                task="transcribe",  # Specify transcription task
            )

            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Calculate metrics
        inference_time = time.time() - start_time
        mem_used = self.profiler.get_memory_usage() - mem_before
        wer, cer = self.profiler.calculate_error_metrics(reference, transcription)

        return {
            "reference": reference,
            "hypothesis": transcription,
            "wer": wer,
            "cer": cer,
            "inference_time": inference_time,
            "memory_used": mem_used,
        }

    def evaluate_dataset(self, dataset) -> pd.DataFrame:
        """Evaluate entire dataset

        Args:
            dataset: Dataset to evaluate

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            result = self.evaluate_sample(item["audio"]["array"], item["text"])
            result["sample_id"] = idx
            results.append(result)

            # Log metrics
            self.profiler.log_metrics(
                {
                    "wer": result["wer"],
                    "cer": result["cer"],
                    "inference_time": result["inference_time"],
                    "memory_used": result["memory_used"],
                }
            )

        return pd.DataFrame(results)

    def save_model(self, model_dir: str) -> None:
        """Save model and processor

        Args:
            model_dir: Directory name for the model
        """
        save_path = self.config.dirs["models"] / model_dir
        save_path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(str(save_path))
        self.processor.save_pretrained(str(save_path))
        print(f"Model and processor saved to '{save_path}'")


class VisualizationUtils:
    """Class for visualization utilities"""

    def __init__(self):
        self.config = ProjectConfig()

    def plot_error_distributions(self, results_df: pd.DataFrame, filename: str):
        """Plot WER and CER distributions"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(data=results_df, x="wer", bins=20)
        plt.title("Distribution of Word Error Rate")
        plt.xlabel("WER")

        plt.subplot(1, 2, 2)
        sns.histplot(data=results_df, x="cer", bins=20)
        plt.title("Distribution of Character Error Rate")
        plt.xlabel("CER")

        plt.tight_layout()
        save_path = self.config.dirs["plots"] / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        print(f"Plot saved to {save_path}")

    def plot_performance_metrics(self, results_df: pd.DataFrame, filename: str):
        """Plot performance metrics distributions"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(data=results_df, x="inference_time", bins=20)
        plt.title("Distribution of Inference Time")
        plt.xlabel("Time (seconds)")

        plt.subplot(1, 2, 2)
        sns.histplot(data=results_df, x="memory_used", bins=20)
        plt.title("Distribution of Memory Usage")
        plt.xlabel("Memory (MB)")

        plt.tight_layout()
        save_path = self.config.dirs["plots"] / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        print(f"Plot saved to {save_path}")


class TranscriptionMetrics:
    """Class to handle transcription-specific metrics"""

    @staticmethod
    def calculate_summary_metrics(
        results_df: pd.DataFrame, model_size: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate summary metrics for transcription results

        Args:
            results_df (pd.DataFrame): DataFrame with evaluation results
            model_size (float, optional): Size of the model in MB

        Returns:
            Dict[str, float]: Dictionary of summary metrics
        """
        metrics = {
            "avg_wer": results_df["wer"].mean(),
            "median_wer": results_df["wer"].median(),
            "std_wer": results_df["wer"].std(),
            "avg_cer": results_df["cer"].mean(),
            "median_cer": results_df["cer"].median(),
            "std_cer": results_df["cer"].std(),
            "avg_inference_time": results_df["inference_time"].mean(),
            "avg_memory_used": results_df["memory_used"].mean(),
        }

        # Add model size if provided
        if model_size is not None:
            metrics["model_size_mb"] = model_size

        return metrics

    @staticmethod
    def save_summary(metrics: Dict[str, float], save_path: Path) -> None:
        """Save summary metrics to CSV

        Args:
            metrics: Dictionary of metrics
            save_path: Path to save the CSV file
        """
        # Convert to DataFrame and save
        pd.DataFrame([metrics]).to_csv(save_path, index=False)
        print(f"Summary metrics saved to {save_path}")

    @staticmethod
    def print_summary(metrics: Dict[str, float]) -> None:
        """Print summary metrics in a formatted way

        Args:
            metrics: Dictionary of metrics
        """
        print("\nSummary Metrics:")
        print("-" * 50)

        # Define metric groups for better organization
        groups = {
            "Error Rates": ["avg_wer", "median_wer", "std_wer", "avg_cer", "median_cer", "std_cer"],
            "Performance": ["avg_inference_time", "avg_memory_used", "model_size_mb"],
        }

        for group_name, metric_names in groups.items():
            print(f"\n{group_name}:")
            for metric in metric_names:
                if metric in metrics:
                    print(f"  {metric:.<30} {metrics[metric]:.4f}")

    @staticmethod
    def print_sample_comparisons(results_df: pd.DataFrame, n_samples: int = 3):
        """Print sample transcription comparisons"""
        print("\nSample Transcriptions:")
        for _, row in results_df.head(n_samples).iterrows():
            print("\nReference:")
            print(row["reference"])
            print("\nHypothesis:")
            print(row["hypothesis"])
            print(f"WER: {row['wer']:.4f}, CER: {row['cer']:.4f}")
            print("-" * 80)


def setup_device() -> torch.device:
    """Configure the device (CUDA if available, MPS for M1/M2, CPU otherwise)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
        print(f"GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    return device


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"


def ensure_dirs_exist() -> None:
    """Ensure all necessary directories exist"""
    dirs = ["data", "models", "results", "results/plots"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True, parents=True)
