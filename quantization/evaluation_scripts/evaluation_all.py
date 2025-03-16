import argparse
import io
import time
from functools import partial
import psutil
import datasets
import numpy as np
import torch
from evaluate import load
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    BitsAndBytesConfig, 
    HqqConfig
)
import os
from datetime import datetime
import json
from optimum.quanto import (
    freeze,
    qint4,
    qint8,
    qfloat8,
    quantize,
    Calibration
)
from collections import deque

class WhisperMemoryTracker:
    def __init__(self, model_name: str, save_path: str):
        self.model_name = model_name
        self.save_path = save_path
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

    def log_memory(self, split, batch_idx, batch_size, audio_duration):
        current_time = time.time()
        cpu_percent = np.mean([self.process.cpu_percent(interval=0.1) for _ in range(3)])  # Avg over 3 readings

        memory_data = {
            "timestamp": float(current_time - self.start_time),  # Ensure it's a native float
            "cpu_percent": float(cpu_percent),  # Ensure it's a native float
            "ram_gb": float(self.process.memory_info().rss / (1024 ** 3)),  # Ensure it's a native float
            "batch_info": {
                "split": split,
                "batch_idx": int(batch_idx),  # Ensure it's a native int
                "batch_size": int(batch_size),  # Ensure it's a native int
                "audio_duration": float(audio_duration)  # Ensure it's a native float
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

        # Append the memory measurement and explicitly make it a dict
        self.memory_measurements.append(dict(memory_data))
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
    
    def get_memory_summary(self):
        """Get comprehensive memory usage statistics."""
        if not self.memory_measurements:
            return "No measurements recorded"
            
        summary = {
            "duration_seconds": time.time() - self.start_time,
            "cpu": {
                "initial_percent": self.initial_cpu_percent,
                "peak_percent": self.peak_cpu_percent,
                "initial_ram_gb": self.initial_ram_usage,
                "current_ram_gb": psutil.Process().memory_info().rss / (1024 ** 3)
            }
        }
        
        if torch.cuda.is_available():
            gpu_measurements = [m["gpu_allocated_gb"] for m in self.memory_measurements]
            summary["gpu"] = {
                "initial_allocated_gb": self.initial_gpu_memory,
                "initial_cached_gb": self.initial_gpu_cached,
                "peak_allocated_gb": self.peak_gpu_memory,
                "average_allocated_gb": sum(gpu_measurements) / len(gpu_measurements),
                "current_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                "current_cached_gb": torch.cuda.memory_reserved() / (1024 ** 3)
            }
        
        return summary
    
    def save_metrics(self):
        """Save memory metrics to a JSON file."""
        metrics_path = os.path.join(self.save_path, f"{self.model_name}_memory_metrics.json")
        summary = self.get_memory_summary()
        
        # Convert deque to list for JSON serialization
        measurements_list = []
        for m in self.memory_measurements:
            # Create a copy of each measurement to avoid modifying the original
            measurement_copy = m.copy() if isinstance(m, dict) else m
            # Convert any non-serializable types
            if isinstance(measurement_copy, dict):
                # Convert timestamps to strings if they're datetime objects
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
        except TypeError as e:
            # If we still have serialization issues, let's create a simpler output
            print(f"Warning: JSON serialization error: {e}")
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
    
    def print_summary(self):
        """Print detailed memory usage summary."""
        summary = self.get_memory_summary()
        
        print(f"\n=== Memory Usage Summary for {self.model_name} ===")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"\nCPU Usage:")
        print(f"  Initial CPU: {summary['cpu']['initial_percent']:.3f}%")
        print(f"  Peak CPU: {summary['cpu']['peak_percent']:.3f}%")
        print(f"  Initial RAM: {summary['cpu']['initial_ram_gb']:.4f} GB")
        print(f"  Current RAM: {summary['cpu']['current_ram_gb']:.4f} GB")
        
        if 'gpu' in summary:
            print(f"\nGPU Usage:")
            print(f"  Initial Allocated: {summary['gpu']['initial_allocated_gb']:.4f} GB")
            print(f"  Peak Allocated: {summary['gpu']['peak_allocated_gb']:.4f} GB")
            print(f"  Average Allocated: {summary['gpu']['average_allocated_gb']:.4f} GB")
            print(f"  Current Allocated: {summary['gpu']['current_allocated_gb']:.4f} GB")
            print(f"  Current Cached: {summary['gpu']['current_cached_gb']:.4f} GB")
    
    def close(self):
        """Cleanup and save final metrics."""
        self.print_summary()
        self.save_metrics()

def load_whisper_model(model_name, device, quantization=None, use_fp16=False):
    try:
        quant_config = None
        # Track models that can't be moved after loading
        is_immovable_quantization = False

        if quantization:
            print(f'Applying {quantization} quantization')
            if quantization.startswith("bnb_"):
                is_immovable_quantization = True
                if quantization == "bnb_fp4_32":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.float32)
                elif quantization == "bnb_fp4_16":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.float16)
                elif quantization == "bnb_nf4_32":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32)
                elif quantization == "bnb_nf4_16":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
                elif quantization == "bnb_fp4_32_double":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)
                elif quantization == "bnb_fp4_16_double":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='fp4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                elif quantization == "bnb_nf4_32_double":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)
                elif quantization == "bnb_nf4_16_double":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            elif quantization.startswith("hqq_int"):
                is_immovable_quantization = True
                if quantization == "hqq_int3":
                    quant_config = HqqConfig(nbits=3)                
                if quantization == "hqq_int4":
                    quant_config = HqqConfig(nbits=4)
                elif quantization == "hqq_int8":
                    quant_config = HqqConfig(nbits=8)
        
        # For quantization methods that can't be moved, use device_map="auto"
        if is_immovable_quantization:
            print(f"Loading with device_map='auto' as {quantization} doesn't support moving models")
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map="cpu"
            )
        else:
            # Load without device_map for other methods
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map=None
            )
            
            # Apply other quantization methods if needed
            if quantization in ["quanto_int4", "quanto_int8"]:
                quantize(model, weights=qint4 if quantization == "quanto_int4" else qint8)
                freeze(model)
            
            if quantization == "pytorch":
                torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )

            # Only move to device if quantization allows it
            model = model.to(device)

        # Apply FP16 only if not quantized and on CUDA
        if use_fp16 and quantization is None and torch.cuda.is_available():
            model = model.half()
            print(f"Converted model to FP16.")

        model.config.forced_decoder_ids = None
        return model

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

def clear_gpu_memory():
    """Clear cached GPU memory and reset peak memory stats if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        print("Running on CPU - no GPU memory to clear")

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
        if next(model.parameters()).dtype == torch.float16:
            features = features.half()
        features = features.to(model.device)

        # Compute total audio duration for the batch
        audio_durations = [len(audio["array"]) / audio["sampling_rate"] for audio in batch["audio"]]
        total_audio_duration = sum(audio_durations)

        # Measure processing time with GPU synchronization
        start_time = time.time()
        predicted_ids = model.generate(features)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all GPU ops complete before stopping the timer
        processing_time = time.time() - start_time

        # Calculate batch RTF
        batch_rtf = processing_time / total_audio_duration
        
        # Log memory usage for this batch
        memory_tracker.log_memory(
            split=split,
            batch_idx=batch_idx,
            batch_size=len(batch["audio"]),
            audio_duration=total_audio_duration
        )

    # Decode predictions
    transcription = [processor.decode(ids) for ids in predicted_ids]
    batch["prediction"] = [processor.tokenizer.normalize(x) for x in transcription]
    # Save per-sample RTF, processing time, and audio duration (same value repeated for all samples in the batch)
    batch["rtf"] = [batch_rtf] * len(batch["audio"])
    batch["processing_time"] = [processing_time] * len(batch["audio"])
    batch["audio_duration"] = [total_audio_duration] * len(batch["audio"])
    return batch

def evaluate_model(model, processor, dataset, metrics, memory_tracker, split, batch_size=2):
    total_processing_time = 0.0
    total_audio_duration = 0.0
    batch_counter = 0
    
    def process_batch(batch):
        nonlocal batch_counter, total_processing_time, total_audio_duration
        # Process the batch and update the cumulative totals
        result = transcribe_batch(batch, model, processor, memory_tracker, split, batch_counter)
        # Each sample in the batch has the same processing time and audio duration;
        # take the value from the first sample as representative.
        batch_processing_time = result["processing_time"][0]
        batch_audio_duration = result["audio_duration"][0]
        total_processing_time += batch_processing_time
        total_audio_duration += batch_audio_duration
        batch_counter += 1
        return result
    
    start = time.time()
    result = dataset.map(process_batch, batched=True, batch_size=batch_size)
    end = time.time()
    
    # Calculate overall RTF from the accumulated totals
    overall_rtf = total_processing_time / total_audio_duration
    print(f"Overall RTF: {overall_rtf:.6f}")
    
    # Compute metrics (e.g., WER, CER)
    scores = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["WER", "CER"]:
            score = 100 * metric.compute(
                references=result["reference"], predictions=result["prediction"]
            )
            scores[metric_name] = score
            print(f"{metric_name}: {score:.5f}")
    
    scores["RTF"] = overall_rtf
    
    # Record CPU metrics
    average_cpu_usage = (sum([m['cpu_percent'] for m in memory_tracker.memory_measurements]) /
                         len(memory_tracker.memory_measurements)) if memory_tracker.memory_measurements else 0
    peak_cpu_usage = memory_tracker.peak_cpu_percent
    scores["avg_cpu_percent"] = average_cpu_usage
    scores["peak_cpu_percent"] = peak_cpu_usage
    
    print(f"{len(result)} sentences evaluated in {end - start:.2f} s.")
    return scores, {"references": result["reference"], "predictions": result["prediction"]}



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

def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)  # Use new serialization
    return buffer.getbuffer().nbytes / (1024**2)

def main():
    original_model_name = "openai/whisper-small"
    batch_size = 16
    save_path = "results"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using {device}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define model configurations
    model_configs = {


        "baseline_fp32": {
            "quantization": None
        },
        "pytorch": {
            "quantization": "pytorch"
        },
        "quanto_int4": {
            "quantization": "quanto_int4",
        },
        "quanto_int8": {
            "quantization": "quanto_int8",
        },
        "hqq_int8": {
            "quantization": "hqq_int8",
        },
        "static_quanto_int4_int8": {
            "quantization": None,
        },
        "static_quanto_int8_int8": {
            "quantization": None,
        },
        "static_quanto_int8_float8": {
            "quantization": None,
        },
        "hqq_int4": {
            "quantization": "hqq_int4",
        },
        "static_quanto_int4_float8": {
            "quantization": None,
        },
        "static_quanto_float8_int8": {
            "quantization": None,
        },
        "static_quanto_float8_float8": {
            "quantization": None,
        },
        "hqq_int3": {
            "quantization": "hqq_int3",
        }
    }

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)
    
    # Load full datasets
    dataset_clean = load_librispeech(num_samples=2620, split="test.clean")
    dataset_other = load_librispeech(num_samples=2939, split="test.other")

    # Calculate 10% for calibration (approximately 300 samples)
    n_calibration_clean = len(dataset_clean) // 10  # ~262 samples
    n_calibration_other = len(dataset_other) // 10  # ~294 samples

    # Split the data into calibration and test sets
    calibration_data_clean = dataset_clean.select(range(n_calibration_clean))
    test_data_clean = dataset_clean.select(range(n_calibration_clean, len(dataset_clean)))
    calibration_data_other = dataset_other.select(range(n_calibration_other))
    test_data_other = dataset_other.select(range(n_calibration_other, len(dataset_other)))

    # Print dataset sizes
    print(f"Clean dataset splits:")
    print(f"  Calibration: {len(calibration_data_clean)} samples")
    print(f"  Test: {len(test_data_clean)} samples")
    print(f"Other dataset splits:")
    print(f"  Calibration: {len(calibration_data_other)} samples")
    print(f"  Test: {len(test_data_other)} samples")

    processed_calibration_data_clean = calibration_data_clean.map(lambda x: map_to_feats(x, processor))
    processed_test_data_clean = test_data_clean.map(lambda x: map_to_feats(x, processor))
    processed_calibration_data_other = calibration_data_other.map(lambda x: map_to_feats(x, processor))
    processed_test_data_other = test_data_other.map(lambda x: map_to_feats(x, processor))
    
    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}
    
    # Store results
    results = {}
    
    # Evaluate each model configuration
    for model_name, config in model_configs.items():
        try:
            print(f"\nEvaluating {model_name} configuration...")
            
            # Clear memory before loading new model
            clear_gpu_memory()
            
            # Load model with current configuration
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                **config
            )

            if "static_quanto_int4_int8" in model_name:

                # Quantize the model
                quantize(model, weights=qint4, activations=qint8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            elif "static_quanto_int8_int8" in model_name:

                # Quantize the model
                quantize(model, weights=qint8, activations=qint8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            elif "static_quanto_int4_float8" in model_name:

                # Quantize the model
                quantize(model, weights=qint4, activations=qfloat8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            elif "static_quanto_int8_float8" in model_name:

                # Quantize the model
                quantize(model, weights=qint8, activations=qfloat8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            elif "static_quanto_float8_int8" in model_name:

                # Quantize the model
                quantize(model, weights=qfloat8, activations=qint8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            elif "static_quanto_float8_float8" in model_name:

                # Quantize the model
                quantize(model, weights=qfloat8, activations=qfloat8)
                calibration_memory_tracker = WhisperMemoryTracker(f"{model_name}", save_path)

                print("Calibrating on dataset...")
                with Calibration():
                    calibration_scores, _ = evaluate_model(
                        model, 
                        processor, 
                        processed_calibration_data_clean, 
                        metrics, 
                        memory_tracker=calibration_memory_tracker,
                        batch_size=batch_size,
                        split='other'
                    )

                freeze(model)

            model.eval()
            
            # Evaluate on both splits
            for split, dataset in [("clean", processed_test_data_clean), 
                                 ("other", processed_test_data_other)]:
                print(f"\nEvaluating on {split} split...")
                
                # Initialize memory tracker for this run
                tracker = WhisperMemoryTracker(f"{model_name}_{split}", save_path)
                
                try:
                    # Run evaluation
                    scores, transcriptions = evaluate_model(
                        model=model,
                        processor=processor,
                        dataset=dataset,
                        metrics=metrics,
                        memory_tracker=tracker,
                        batch_size=batch_size,
                        split=split
                    )
                    
                    # Store and save results
                    if scores is not None:
                        model_size = get_model_disk_size_in_mb(model)
                        results[f"{model_name}_{split}"] = {
                            "metrics": scores,
                            "model_size_mb": model_size,
                            "model_type": model_name,
                            "model_name": original_model_name,
                        }
                        
                        # Save metrics
                        metrics_path = os.path.join(save_path, f"{model_name}_{split}_metrics.json")
                        with open(metrics_path, "w") as f:
                            json.dump(results[f"{model_name}_{split}"], f, indent=2)
                        
                        # Save transcriptions
                        transcriptions_path = os.path.join(save_path, f"{model_name}_{split}_transcriptions.json")
                        with open(transcriptions_path, "w") as f:
                            json.dump(transcriptions, f, indent=2)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} on {split} split: {str(e)}")
                    continue
                    
                finally:
                    # Always close tracker and clear memory
                    tracker.close()
                    clear_gpu_memory()
            
            # Clear model from memory
            del model
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error setting up {model_name}: {str(e)}")
            continue
    
    # Print final summary
    print("\nFinal Results Summary:")
    for run_name, run_results in results.items():
        print(f"\n{run_name}:")
        print(f"Model Size: {run_results['model_size_mb']:.2f} MB")
        print("Metrics:")
        for metric_name, value in run_results['metrics'].items():
            print(f"  {metric_name}: {value:.4f}")

        # Add CPU metrics if they exist
        if "avg_cpu_percent" in run_results['metrics']:
            print(f"  Avg CPU: {run_results['metrics']['avg_cpu_percent']:.4f}%")
            print(f"  Peak CPU: {run_results['metrics']['peak_cpu_percent']:.4f}%")

if __name__ == "__main__":
    main()
