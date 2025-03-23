import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("whisper_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WhisperMemoryTracker:
    """Tracks memory usage during model evaluation."""
    
    def __init__(self, model_name: str, save_path: str):
        """
        Initialize memory tracker.
        
        Args:
            model_name: Name of the model being tracked
            save_path: Directory to save metrics
        """
        self.model_name = model_name
        self.save_path = Path(save_path)
        self.peak_gpu_memory = 0
        self.peak_cpu_percent = 0
        self.memory_measurements = deque(maxlen=500)  # Store last 500 measurements
        self.start_time = time.time()
        self.process = psutil.Process()

        # Initialize GPU memory attributes even if running on CPU
        self.initial_gpu_memory = 0
        self.initial_gpu_cached = 0

        self.process.cpu_percent(interval=None)  # First call returns 0, discard it
        self.initial_cpu_percent = np.mean([self.process.cpu_percent(interval=0.1) for _ in range(5)])  # Stable avg
        self.initial_ram_usage = self.process.memory_info().rss / (1024 ** 3)

        # Initialize GPU memory metrics if available
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            self.initial_gpu_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            
        # Ensure save path exists
        self.save_path.mkdir(parents=True, exist_ok=True)

    def log_memory(self, split: str, batch_idx: int, batch_size: int, audio_duration: float) -> None:
        """
        Log current memory usage.
        
        Args:
            split: Dataset split name
            batch_idx: Current batch index
            batch_size: Batch size
            audio_duration: Total audio duration in seconds
        """
        current_time = time.time()
        cpu_percent = np.mean([self.process.cpu_percent(interval=0.1) for _ in range(3)])  # Avg over 3 readings

        memory_data = {
            "timestamp": float(current_time - self.start_time),
            "cpu_percent": float(cpu_percent),
            "ram_gb": float(self.process.memory_info().rss / (1024 ** 3)),
            "batch_info": {
                "split": split,
                "batch_idx": int(batch_idx),
                "batch_size": int(batch_size),
                "audio_duration": float(audio_duration)
            }
        }

        if torch.cuda.is_available():
            gpu_allocated = float(torch.cuda.memory_allocated() / (1024 ** 3))
            gpu_cached = float(torch.cuda.memory_reserved() / (1024 ** 3)) 
            gpu_peak = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            
            memory_data.update({
                "gpu_allocated_gb": gpu_allocated,
                "gpu_cached_gb": gpu_cached,
                "gpu_peak_gb": gpu_peak
            })
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_peak)

        # Append the memory measurement
        self.memory_measurements.append(dict(memory_data))
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics.
        
        Returns:
            Dictionary with memory usage summary
        """
        if not self.memory_measurements:
            return {"error": "No measurements recorded"}
            
        summary = {
            "duration_seconds": time.time() - self.start_time,
            "cpu": {
                "initial_percent": self.initial_cpu_percent,
                "peak_percent": self.peak_cpu_percent,
                "initial_ram_gb": self.initial_ram_usage,
                "current_ram_gb": self.process.memory_info().rss / (1024 ** 3)
            }
        }
        
        if torch.cuda.is_available():
            gpu_measurements = [m.get("gpu_allocated_gb", 0) for m in self.memory_measurements if "gpu_allocated_gb" in m]
            if gpu_measurements:
                summary["gpu"] = {
                    "initial_allocated_gb": self.initial_gpu_memory,
                    "initial_cached_gb": self.initial_gpu_cached,
                    "peak_allocated_gb": self.peak_gpu_memory,
                    "average_allocated_gb": sum(gpu_measurements) / len(gpu_measurements) if gpu_measurements else 0,
                    "current_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                    "current_cached_gb": torch.cuda.memory_reserved() / (1024 ** 3)
                }
        
        return summary
    
    def save_metrics(self) -> None:
        """Save memory metrics to a JSON file."""
        metrics_path = self.save_path / f"{self.model_name}_memory_metrics.json"
        summary = self.get_memory_summary()
        
        # Convert deque to list for JSON serialization
        measurements_list = []
        for m in self.memory_measurements:
            measurement_copy = m.copy() if isinstance(m, dict) else m
            if isinstance(measurement_copy, dict):
                if "timestamp" in measurement_copy and isinstance(measurement_copy["timestamp"], datetime):
                    measurement_copy["timestamp"] = measurement_copy["timestamp"].isoformat()
            measurements_list.append(measurement_copy)
        
        # Create the output dictionary with serializable data
        output_data = {
            "summary": summary,
            "detailed_measurements": measurements_list
        }
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                logger.info(f"Memory metrics saved to {metrics_path}")
        except TypeError as e:
            logger.warning(f"JSON serialization error: {e}")
            simplified_output = {
                "summary": {
                    "duration_seconds": summary["duration_seconds"] if isinstance(summary, dict) else 0,
                    "cpu": {
                        "peak_percent": self.peak_cpu_percent,
                        "current_ram_gb": self.process.memory_info().rss / (1024 ** 3)
                    }
                },
                "error": "Full data couldn't be serialized to JSON"
            }
            with open(metrics_path, 'w') as f:
                json.dump(simplified_output, f, indent=2)
                logger.warning(f"Simplified metrics saved to {metrics_path} due to serialization error")
    
    def print_summary(self) -> None:
        """Print detailed memory usage summary."""
        summary = self.get_memory_summary()
        
        logger.info(f"\n=== Memory Usage Summary for {self.model_name} ===")
        logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
        logger.info(f"\nCPU Usage:")
        logger.info(f"  Initial CPU: {summary['cpu']['initial_percent']:.3f}%")
        logger.info(f"  Peak CPU: {summary['cpu']['peak_percent']:.3f}%")
        logger.info(f"  Initial RAM: {summary['cpu']['initial_ram_gb']:.4f} GB")
        logger.info(f"  Current RAM: {summary['cpu']['current_ram_gb']:.4f} GB")
        
        if 'gpu' in summary:
            logger.info(f"\nGPU Usage:")
            logger.info(f"  Initial Allocated: {summary['gpu']['initial_allocated_gb']:.4f} GB")
            logger.info(f"  Peak Allocated: {summary['gpu']['peak_allocated_gb']:.4f} GB")
            logger.info(f"  Average Allocated: {summary['gpu']['average_allocated_gb']:.4f} GB")
            logger.info(f"  Current Allocated: {summary['gpu']['current_allocated_gb']:.4f} GB")
            logger.info(f"  Current Cached: {summary['gpu']['current_cached_gb']:.4f} GB")
    
    def close(self) -> None:
        """Cleanup and save final metrics."""
        self.print_summary()
        self.save_metrics()