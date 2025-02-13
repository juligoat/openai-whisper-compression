#!/usr/bin/env python3

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig

# Print CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

model_id = "openai/whisper-small"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load processor (tokenizer + feature extractor)
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded successfully")

# Load Whisper model with 4-bit quantization
print("Loading model...")
model_4bit = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
print("Model loaded successfully")

# Try converting to BetterTransformer
try:
    print("Attempting BetterTransformer conversion...")
    model_4bit = model_4bit.to_bettertransformer()
    print("BetterTransformer conversion successful")
except Exception as e:
    print(f"BetterTransformer conversion failed: {e}")

print("All tests completed")
