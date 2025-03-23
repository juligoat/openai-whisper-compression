import logging
import datasets
from typing import Dict, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)

def load_librispeech(num_samples=None, split="test.clean"):
    """
    Load LibriSpeech clean/other data.
    
    Args:
        num_samples: Number of samples to load (None for all)
        split: Dataset split to load
        
    Returns:
        Loaded dataset
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

    logger.info(f"Loaded {len(dataset)} test samples")
    logger.info(f"Total audio duration: {total_hours:.4f} hours")
    return dataset

def map_to_feats(batch, processor):
    """
    Process audio data into model input features.
    
    Args:
        batch: Batch of audio data
        processor: Whisper processor
        
    Returns:
        Processed batch with input features
    """
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    batch["input_features"] = input_features
    batch["reference"] = processor.tokenizer.normalize(batch["text"])
    return batch

def prepare_datasets(processor, num_samples_clean=2620, num_samples_other=2939, calibration_percentage=0.1):
    """
    Load and prepare datasets for evaluation.
    
    Args:
        processor: Whisper processor
        num_samples_clean: Number of clean samples to load
        num_samples_other: Number of other samples to load
        calibration_percentage: Percentage to use for calibration
        
    Returns:
        Dictionary containing all processed datasets
    """
    # Load full datasets
    dataset_clean = load_librispeech(num_samples=num_samples_clean, split="test.clean")
    dataset_other = load_librispeech(num_samples=num_samples_other, split="test.other")

    # Calculate samples for calibration
    n_calibration_clean = int(len(dataset_clean) * calibration_percentage)
    n_calibration_other = int(len(dataset_other) * calibration_percentage)

    # Split the data into calibration and test sets
    calibration_data_clean = dataset_clean.select(range(n_calibration_clean))
    test_data_clean = dataset_clean.select(range(n_calibration_clean, len(dataset_clean)))
    calibration_data_other = dataset_other.select(range(n_calibration_other))
    test_data_other = dataset_other.select(range(n_calibration_other, len(dataset_other)))

    # Print dataset sizes
    logger.info(f"Clean dataset splits:")
    logger.info(f"  Calibration: {len(calibration_data_clean)} samples")
    logger.info(f"  Test: {len(test_data_clean)} samples")
    logger.info(f"Other dataset splits:")
    logger.info(f"  Calibration: {len(calibration_data_other)} samples")
    logger.info(f"  Test: {len(test_data_other)} samples")

    # Process each split
    processed_calibration_data_clean = calibration_data_clean.map(
        lambda x: map_to_feats(x, processor)
    )
    processed_test_data_clean = test_data_clean.map(
        lambda x: map_to_feats(x, processor)
    )
    processed_calibration_data_other = calibration_data_other.map(
        lambda x: map_to_feats(x, processor)
    )
    processed_test_data_other = test_data_other.map(
        lambda x: map_to_feats(x, processor)
    )
    
    return {
        "calibration_clean": processed_calibration_data_clean,
        "test_clean": processed_test_data_clean,
        "calibration_other": processed_calibration_data_other,
        "test_other": processed_test_data_other
    }

def transcribe_batch(batch, model, processor, memory_tracker, split, batch_idx):
    """
    Transcribe a batch of audio data.
    
    Args:
        batch: Batch of audio data
        model: Whisper model
        processor: Whisper processor
        memory_tracker: Memory tracking object
        split: Dataset split name
        batch_idx: Batch index
        
    Returns:
        Processed batch with predictions
    """
    import torch
    import numpy as np
    import time
    
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
    # Save per-sample RTF, processing time, and audio duration
    batch["rtf"] = [batch_rtf] * len(batch["audio"])
    batch["processing_time"] = [processing_time] * len(batch["audio"])
    batch["audio_duration"] = [total_audio_duration] * len(batch["audio"])
    return batch