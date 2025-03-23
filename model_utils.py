import io
import logging
import torch
from typing import Dict, Optional, Union, Any

from transformers import (
    WhisperForConditionalGeneration, 
    BitsAndBytesConfig, 
    HqqConfig
)
from optimum.quanto import (
    freeze,
    qint4,
    qint8,
    qfloat8,
    quantize,
    Calibration
)

logger = logging.getLogger(__name__)

def clear_gpu_memory() -> None:
    """
    Clear cached GPU memory and reset peak memory stats if CUDA is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU memory cleared")
    else:
        logger.info("Running on CPU - no GPU memory to clear")

def _create_bnb_config(quantization_type: str) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytesConfig for 4-bit quantization.
    
    Args:
        quantization_type: String identifier for the quantization type
            (e.g., 'bnb_fp4_32', 'bnb_nf4_16_double')
            
    Returns:
        BitsAndBytesConfig object with appropriate settings
    """
    # Determine compute dtype (float16 or float32)
    compute_dtype = torch.float16 if "16" in quantization_type else torch.float32
    
    # Determine quantization type (fp4 or nf4)
    quant_type = "fp4" if "fp4" in quantization_type else "nf4"
    
    # Determine if double quantization should be used
    use_double_quant = "double" in quantization_type
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant
    )

def _create_hqq_config(quantization_type: str) -> HqqConfig:
    """
    Create an HqqConfig for n-bit quantization.
    
    Args:
        quantization_type: String identifier for the HQQ quantization type
            (e.g., 'hqq_int3', 'hqq_int4', 'hqq_int8')
            
    Returns:
        HqqConfig object with appropriate settings
    """
    # Extract number of bits from the quantization type
    if "int3" in quantization_type:
        nbits = 3
    elif "int4" in quantization_type:
        nbits = 4
    elif "int8" in quantization_type:
        nbits = 8
    else:
        raise ValueError(f"Unsupported HQQ quantization type: {quantization_type}")
    
    return HqqConfig(nbits=nbits)

def load_whisper_model(
    model_name: str, 
    device: torch.device, 
    quantization: Optional[str] = None, 
    use_fp16: bool = False
) -> WhisperForConditionalGeneration:
    """
    Load a Whisper model with specified quantization settings.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model onto
        quantization: Quantization method to apply (e.g., 'bnb_fp4_32', 'quanto_int8')
        use_fp16: Whether to use FP16 precision
        
    Returns:
        Loaded and configured WhisperForConditionalGeneration model
    """
    try:
        quant_config = None
        is_immovable_quantization = False

        if quantization:
            logger.info(f'Applying {quantization} quantization')
            
            # Handle BitsAndBytes (bnb) quantization
            if quantization.startswith("bnb_"):
                is_immovable_quantization = True
                quant_config = _create_bnb_config(quantization)
            
            # Handle HQQ quantization
            elif quantization.startswith("hqq_int"):
                is_immovable_quantization = True
                quant_config = _create_hqq_config(quantization)
        
        # For quantization methods that can't be moved, use device_map="auto"
        if is_immovable_quantization:
            logger.info(f"Loading with device_map='auto' as {quantization} doesn't support moving models")
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map="auto"
            )
        else:
            # Load without device_map for other methods
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map=None
            )
            
            # Apply Quanto quantization if needed
            if quantization in ["quanto_int4", "quanto_int8"]:
                quantize(model, weights=qint4 if quantization == "quanto_int4" else qint8)
                freeze(model)
            
            # Apply PyTorch dynamic quantization
            elif quantization == "pytorch":
                torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )

            # Only move to device if quantization allows it
            model = model.to(device)

        # Apply FP16 only if not quantized and on CUDA
        if use_fp16 and quantization is None and torch.cuda.is_available():
            model = model.half()
            logger.info(f"Converted model to FP16")

        model.config.forced_decoder_ids = None
        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise

def apply_static_quantization(
    model: WhisperForConditionalGeneration,
    weight_type: str,
    activation_type: str,
    processor,
    dataset,
    metrics,
    model_name: str,
    save_path: str,
    batch_size: int,
    memory_tracker_cls,
    evaluate_fn
) -> WhisperForConditionalGeneration:
    """
    Apply static quantization with calibration.
    
    Args:
        model: The model to quantize
        weight_type: Weight quantization type ('int4', 'int8', 'float8')
        activation_type: Activation quantization type ('int4', 'int8', 'float8')
        processor: Whisper processor for tokenization
        dataset: Dataset for calibration
        metrics: Metrics to compute during calibration
        model_name: Name of the model for logging
        save_path: Path to save calibration results
        batch_size: Batch size for calibration
        memory_tracker_cls: Class for tracking memory usage
        evaluate_fn: Function to use for evaluation during calibration
        
    Returns:
        Quantized model
    """
    # Map weight and activation types to quanto types
    type_mapping = {
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8
    }
    
    if weight_type not in type_mapping or activation_type not in type_mapping:
        raise ValueError(f"Unsupported quantization types: weight_type={weight_type}, activation_type={activation_type}")
    
    # Quantize the model with specified types
    quantize(model, weights=type_mapping[weight_type], activations=type_mapping[activation_type])
    
    # Setup memory tracker for calibration
    calibration_memory_tracker = memory_tracker_cls(f"{model_name}_calibration", save_path)
    
    logger.info(f"Calibrating on dataset with {weight_type}/{activation_type}...")
    with Calibration():
        calibration_scores, _ = evaluate_fn(
            model, 
            processor, 
            dataset, 
            metrics, 
            memory_tracker=calibration_memory_tracker,
            batch_size=batch_size,
            split='calibration'
        )
    
    # Freeze the model after calibration
    freeze(model)
    calibration_memory_tracker.close()
    
    return model

def get_model_disk_size_in_mb(model: torch.nn.Module) -> float:
    """
    Calculate the disk size of a model in megabytes.
    
    Args:
        model: PyTorch model to measure
        
    Returns:
        Size of the model in MB
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
    size_mb = buffer.getbuffer().nbytes / (1024**2)
    return size_mb

def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    Get memory usage statistics for a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with memory usage statistics
    """
    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters())
    
    # Calculate parameter size in MB
    param_size_mb = param_count * 4 / (1024**2)  # Assuming float32 (4 bytes per parameter)
    
    # Get GPU memory usage if available
    gpu_allocated_mb = 0
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        gpu_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    
    return {
        "param_count": param_count,
        "param_size_mb": param_size_mb,
        "gpu_allocated_mb": gpu_allocated_mb,
        "model_disk_size_mb": get_model_disk_size_in_mb(model)
    }

def save_model_metrics(
    model: torch.nn.Module, 
    model_name: str, 
    metrics: Dict[str, Any], 
    save_path: str
) -> None:
    """
    Save model metrics and metadata to disk.
    
    Args:
        model: The model to analyze
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the metrics
    """
    import os
    import json
    
    # Get model memory usage
    memory_stats = get_model_memory_usage(model)
    
    # Combine with metrics
    combined_metrics = {
        "model_name": model_name,
        "memory_stats": memory_stats,
        "evaluation_metrics": metrics
    }
    
    # Save to file
    os.makedirs(save_path, exist_ok=True)
    metrics_path = os.path.join(save_path, f"{model_name}_model_metrics.json")
    
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    
    logger.info(f"Model metrics saved to {metrics_path}")