import logging
import os

import torch
from evaluate import load
from transformers import WhisperProcessor

from data_utils import prepare_datasets
from evaluation import evaluate_model, print_evaluation_summary, save_evaluation_results

# Import from our custom modules
from memory_tracker import WhisperMemoryTracker
from model_utils import (
    apply_static_quantization,
    clear_gpu_memory,
    get_model_disk_size_in_mb,
    load_whisper_model,
    save_model_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("whisper_eval.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    # Configuration parameters
    original_model_name = "openai/whisper-small"
    batch_size = 16
    save_path = "results"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using {device}")

    # Ensure results directory exists
    os.makedirs(save_path, exist_ok=True)

    # Define model configurations
    model_configs = {
        "baseline_fp32": {"quantization": None},
        "pytorch": {"quantization": "pytorch"},
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
            "weight_type": "int4",
            "activation_type": "int8",
        },
        "static_quanto_int8_int8": {
            "quantization": None,
            "weight_type": "int8",
            "activation_type": "int8",
        },
        "static_quanto_int8_float8": {
            "quantization": None,
            "weight_type": "int8",
            "activation_type": "float8",
        },
        "hqq_int4": {
            "quantization": "hqq_int4",
        },
        "static_quanto_int4_float8": {
            "quantization": None,
            "weight_type": "int4",
            "activation_type": "float8",
        },
        "static_quanto_float8_int8": {
            "quantization": None,
            "weight_type": "float8",
            "activation_type": "int8",
        },
        "static_quanto_float8_float8": {
            "quantization": None,
            "weight_type": "float8",
            "activation_type": "float8",
        },
        "hqq_int3": {
            "quantization": "hqq_int3",
        },
    }

    # Load processor once - can be shared across models
    processor = WhisperProcessor.from_pretrained(original_model_name)

    # Load and prepare datasets
    datasets_dict = prepare_datasets(
        processor=processor,
        num_samples_clean=50,  # Using smaller sample for faster testing
        num_samples_other=50,  # Using smaller sample for faster testing
        calibration_percentage=0.1,
    )

    # Get datasets from the dictionary
    processed_calibration_data_clean = datasets_dict["calibration_clean"]
    processed_test_data_clean = datasets_dict["test_clean"]
    processed_calibration_data_other = datasets_dict["calibration_other"]
    processed_test_data_other = datasets_dict["test_other"]

    # Initialize metrics
    metrics = {"WER": load("wer"), "CER": load("cer")}

    # Store results
    results = {}

    # Evaluate each model configuration
    for model_name, config in model_configs.items():
        try:
            logger.info(f"\nEvaluating {model_name} configuration...")

            # Clear memory before loading new model
            clear_gpu_memory()

            # Load model with current configuration
            model = load_whisper_model(
                model_name=original_model_name,
                device=device,
                **{k: v for k, v in config.items() if k in ["quantization", "use_fp16"]},
            )

            # Handle static quantization configs
            if "weight_type" in config and "activation_type" in config:
                model = apply_static_quantization(
                    model=model,
                    weight_type=config["weight_type"],
                    activation_type=config["activation_type"],
                    processor=processor,
                    dataset=processed_calibration_data_clean,
                    metrics=metrics,
                    model_name=model_name,
                    save_path=save_path,
                    batch_size=batch_size,
                    memory_tracker_cls=WhisperMemoryTracker,
                    evaluate_fn=evaluate_model,
                )

            model.eval()

            # Evaluate on both splits
            for split, dataset in [
                ("clean", processed_test_data_clean),
                ("other", processed_test_data_other),
            ]:
                logger.info(f"\nEvaluating on {split} split...")

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
                        split=split,
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

                        # Save results to disk
                        save_evaluation_results(
                            model_name=model_name,
                            split=split,
                            results=results[f"{model_name}_{split}"],
                            transcriptions=transcriptions,
                            save_path=save_path,
                        )

                        # Save additional model metrics
                        save_model_metrics(
                            model=model,
                            model_name=f"{model_name}_{split}",
                            metrics=scores,
                            save_path=save_path,
                        )

                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {split} split: {e!s}")
                    continue

                finally:
                    # Always close tracker and clear memory
                    tracker.close()
                    clear_gpu_memory()

            # Clear model from memory
            del model
            clear_gpu_memory()

        except Exception as e:
            logger.error(f"Error setting up {model_name}: {e!s}")
            continue

    # Print final summary
    print_evaluation_summary(results)


if __name__ == "__main__":
    main()
