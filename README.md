# Dialect-Adaptation

# Beyond Standard Greek: Adapting Whisper for Greek Dialects through Curriculum Multitask Learning

## Overview

This repository contains the official implementation of:

> **Beyond Standard Greek: Adapting Whisper for Greek Dialects through Curriculum Multitask Learning**

We introduce a staged multitask curriculum that progressively adapts Whisper models from Standard Modern Greek to low-resource Greek dialects through a structured transition across both supervision tasks and linguistic domains.

The proposed framework combines:

* Speech-to-Text Translation (ST)
* Automatic Speech Recognition (ASR)
* Donor-language supervision
* Curriculum Learning

to improve adaptation stability under severe data scarcity.

---

## Abstract

Automatic Speech Recognition systems remain highly sensitive to dialectal variation, particularly in low-resource settings where transcription data are limited. We propose a staged multitask curriculum that gradually transitions from high-resource donor-language translation to dialect-specific transcription. The curriculum operates simultaneously across task space and domain space, reducing distribution mismatch while stabilizing optimization during adaptation.

Experiments on three Greek dialects (Cypriot, Cretan, and Messenian) demonstrate consistent improvements over conventional Whisper fine-tuning across multiple model scales. Extensive ablation studies further reveal the importance of both translation supervision and donor-language pre-alignment.

---

## Method

The proposed curriculum consists of four consecutive stages:

| Stage   | Objective         | Dataset Composition                |
| ------- | ----------------- | ---------------------------------- |
| Stage 0 | Donor Translation | Greek Speech → English Translation |
| Stage 1 | Mixed Domain      | Greek ASR + Dialect Translation    |
| Stage 2 | Adaptation        | Dialect ASR + Dialect Translation  |
| Stage 3 | Target Domain     | Dialect ASR                        |

The main curriculum configuration used throughout the paper is:

| Stage   | Composition                               |
| ------- | ----------------------------------------- |
| Stage 0 | 100% Greek Translation                    |
| Stage 1 | 40% Greek ASR + 60% Dialect Translation   |
| Stage 2 | 60% Dialect ASR + 40% Dialect Translation |
| Stage 3 | 100% Dialect ASR                          |

---

## Experimental Setup

### Dialects

* Cypriot Greek
* Cretan Greek
* Messenian Greek

### Models

* Whisper Small
* Whisper Medium
* Whisper Large-v3

### Metrics

* Word Error Rate (WER)
* Character Error Rate (CER)

---

## Main Results

| Model                       | Cypriot WER | Cretan WER | Messenian WER |
| --------------------------- | ----------- | ---------- | ------------- |
| Whisper Small FT            | 52.38       | 51.37      | 41.97         |
| Whisper Small Curriculum    | 35.61       | 29.51      | 30.25         |
| Whisper Medium FT           | 37.85       | 41.07      | 22.31         |
| Whisper Medium Curriculum   | 32.47       | 21.17      | 20.42         |
| Whisper Large-v3 FT         | 33.93       | 33.93      | 15.31         |
| Whisper Large-v3 Curriculum | 24.89       | 17.69      | 14.56         |

The proposed curriculum consistently outperforms conventional fine-tuning across all evaluated dialects and model scales.

---

## Repository Structure

```text
Dialect-Adaptation/

├── curriculum_learning_adaptation/
│   ├── curriculum_learning/
│   └── curriculum_learning_without_translation_3stages/
│
├── data_creation/
│
├── fine_tuning/
│
├── evaluation/
│   ├── evaluation_curriculum/
│   └── evaluation_for_fine_tuning/
│
├── figures/
│
├── paper/
│
└── README.md
```

---

## Reproducing Experiments

### Dataset Construction

```bash
python data_creation/build_curriculum_datasets.py
```

### Training

```bash
python fine_tuning/train.py
```

### Evaluation

```bash
python evaluation/evaluation_curriculum/evaluate.py
```
