#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "/leonardo_work/EUHPC_D23_153/antyklimi_data/whisper_models/whisper-large-v3"

CHECKPOINT = "/leonardo_work/EUHPC_D23_153/antyklimi_runs/whisper_large_bimodal/stage_3/checkpoint-1000"
DATASET_PATH = "/leonardo_work/EUHPC_D23_153/antyklimi_data/hf_datasets/cy_normalized"

# =====================================================
# LOAD DATASET
# =====================================================

print(f"Loading dataset from: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)


if isinstance(dataset, dict):
    dataset = dataset["test"] if "test" in dataset else dataset[list(dataset.keys())[0]]

# =====================================================
# LOAD MODEL & PROCESSOR
# =====================================================

print(f"Loading processor from base model: {BASE_MODEL}")
processor = WhisperProcessor.from_pretrained(BASE_MODEL, local_files_only=True)

print(f"Loading fine-tuned model from: {CHECKPOINT}")
model = WhisperForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    local_files_only=True,
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2"
)

model.to(DEVICE)
model.eval()

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    task="transcribe",
    language="el"
)
model.config.suppress_tokens = []

# =====================================================
# EVALUATION LOOP
# =====================================================

predictions = []
references = []

print(f"\n Evaluating on {len(dataset)} samples (Normalized)...")

for sample in tqdm(dataset):
    audio = sample["audio"]["array"]
    
   
    reference = sample.get("text", sample.get("transcription_norm", ""))

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_features.to(dtype=torch.float16), 
            max_length=448
        )

    # 1. Raw Decoding
    prediction_raw = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

  
    norm_prediction = processor.tokenizer._normalize(prediction_raw) 
    norm_reference = processor.tokenizer._normalize(reference)       

    predictions.append(norm_prediction)
    references.append(norm_reference)


wer_score = 100 * wer(references, predictions)
cer_score = 100 * cer(references, predictions)

print(f"\n" + "="*40)
print(f" FINAL NORMALIZED RESULTS")
print(f"   WER: {wer_score:.2f}%")
print(f"   CER: {cer_score:.2f}%")
print("="*40)

# =====================================================
# SAVE RESULTS
# =====================================================

results = {
    "checkpoint": CHECKPOINT,
    "wer": wer_score,
    "cer": cer_score
}

output_file = "evaluation_normalized_final.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

with open("normalized_samples.txt", "w") as f:
    for i in range(min(50, len(predictions))):
        f.write(f"REF:  {references[i]}\n")
        f.write(f"PRED: {predictions[i]}\n")
        f.write("-" * 30 + "\n")

print(f"\nResults saved to {output_file}")
