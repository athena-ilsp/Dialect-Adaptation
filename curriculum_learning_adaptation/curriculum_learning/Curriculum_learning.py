#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ============================================================
# PATHS (LEONARDO)
# ============================================================

MODEL_ROOT = "/leonardo_work/EUHPC_D23_153/antyklimi_data/whisper_models/whisper-large-v3"
CURRICULUM_ROOT = "/leonardo_work/EUHPC_D23_153/antyklimi_data/cypriot_curriculum/exp4_balanced"

OUTPUT_ROOT = "/leonardo_work/EUHPC_D23_153/antyklimi_data/experiments/cypriot_largev3_exp4_balanced"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ============================================================
# HYPERPARAMS
# ============================================================

STAGES = [0, 1, 2, 3]

BATCH_SIZE = 2
GRAD_ACCUM = 8
FP16 = True
MAX_LABEL_LEN = 448

def get_stage_params(stage_idx):
    if stage_idx == 0:
        return {"lr": 2e-5, "epochs": 2}
    elif stage_idx == 1:
        return {"lr": 1.1e-5, "epochs": 3}
    elif stage_idx == 2:
        return {"lr": 1e-5, "epochs": 4}
    else:
        return {"lr": 5e-6, "epochs": 8}

# ============================================================
# LOAD PROCESSOR (LOCAL)
# ============================================================

processor = WhisperProcessor.from_pretrained(
    MODEL_ROOT,
    local_files_only=True,
    attn_implementation="flash_attention_2"
)

# ============================================================
# DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorWhisper:
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feats,
            return_tensors="pt"
        )

        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_feats,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels
        return batch

collator = DataCollatorWhisper(processor)

# ============================================================
# PREPARE FUNCTION (LIKE MEDIUM)
# ============================================================

def prepare(batch):

    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=16000
    ).input_features[0]

    task = batch.get("task", "transcribe")

    if task == "translate":
        prompt = processor.tokenizer.get_decoder_prompt_ids(
            task="translate",
            language="en"
        )
    else:
        prompt = processor.tokenizer.get_decoder_prompt_ids(
            task="transcribe",
            language="el"
        )

    tokens = processor.tokenizer(batch["text"]).input_ids
    labels = [p[1] for p in prompt] + tokens

    batch["labels"] = labels[:MAX_LABEL_LEN]

    return batch

# ============================================================
# CURRICULUM TRAINING LOOP
# ============================================================

current_model_path = MODEL_ROOT

for stage in STAGES:

    params = get_stage_params(stage)

    print(f"\n TRAINING STAGE {stage}")
    print(f" LR={params['lr']} | Epochs={params['epochs']}")

    model = WhisperForConditionalGeneration.from_pretrained(
        current_model_path,
        local_files_only=True
    )

    model.config.use_cache = False

    # Final stage → force Greek ASR
    if stage == 3:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="el",
            task="transcribe"
        )
        model.config.suppress_tokens = []

    model.to("cuda")

    stage_path = os.path.join(CURRICULUM_ROOT, f"stage_{stage}")
    

    dataset = load_from_disk(stage_path)

    dataset = dataset.map(
    prepare,
    remove_columns=[c for c in dataset.column_names if c in ["audio", "text", "task"]],
    num_proc=1,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(OUTPUT_ROOT, f"stage_{stage}"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=params["lr"],
        num_train_epochs=params["epochs"],
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        fp16=FP16,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    final_path = os.path.join(
        OUTPUT_ROOT,
        f"stage_{stage}",
        "final"
    )

    trainer.save_model(final_path)
    current_model_path = final_path

    del model
    torch.cuda.empty_cache()

print("\n Whisper cypriot_largev3_exp4_balanced Curriculum Training COMPLETED!")

