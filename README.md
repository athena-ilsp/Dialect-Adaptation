# Beyond Standard Greek: Adapting Whisper for Greek Dialects through Curriculum Multitask Learning

## Overview

This repository contains the official implementation of:

> **Beyond Standard Greek: Adapting Whisper for Greek Dialects through Curriculum Multitask Learning**

We introduce a staged multitask curriculum that progressively adapts Whisper models from Standard Modern Greek to low-resource Greek dialects through a structured transition across both supervision tasks and linguistic domains.

The proposed framework combines:

- Speech-to-Text Translation (ST)
- Automatic Speech Recognition (ASR)
- Donor-language supervision
- Curriculum Learning

to improve adaptation stability under severe data scarcity.

**Resources:** [Paper](https://doi.org/10.21437/Interspeech.2026-2567) · [Trained Models and Checkpoints (Hugging Face)](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation) · [Cypriot Greek Dataset Source (Mozilla Data Collective)](https://datacollective.mozillafoundation.org/datasets/cmflnuzz3xq2ok28b4o1ojacv)

---

## Abstract

Automatic Speech Recognition systems remain highly sensitive to dialectal variation, particularly in low-resource settings where transcription data are limited. We propose a staged multitask curriculum that gradually transitions from high-resource donor-language translation to dialect-specific transcription. The curriculum operates simultaneously across task space and domain space, reducing distribution mismatch while stabilizing optimization during adaptation.

Experiments on three Greek dialects (Cypriot, Cretan, and Messenian) demonstrate consistent improvements over conventional Whisper fine-tuning across multiple model scales. Extensive ablation studies further reveal the importance of both translation supervision and donor-language pre-alignment.

---

## Method

The proposed curriculum consists of four consecutive stages:

| Stage | Objective | Dataset Composition |
| --- | --- | --- |
| Stage 0 | Donor Translation | Greek Speech → English Translation |
| Stage 1 | Mixed Domain | Greek ASR + Dialect Translation |
| Stage 2 | Adaptation | Dialect ASR + Dialect Translation |
| Stage 3 | Target Domain | Dialect ASR |

The main curriculum configuration used throughout the paper is:

| Stage | Composition |
| --- | --- |
| Stage 0 | 100% Greek Translation |
| Stage 1 | 40% Greek ASR + 60% Dialect Translation |
| Stage 2 | 60% Dialect ASR + 40% Dialect Translation |
| Stage 3 | 100% Dialect ASR |

---

## Experimental Setup

### Dialects

- Cypriot Greek
- Cretan Greek
- Messenian Greek

### Models

- Whisper Small
- Whisper Medium
- Whisper Large-v3

### Metrics

- Word Error Rate (WER)
- Character Error Rate (CER)

---

## Main Results

| Model | Cypriot WER | Cretan WER | Messenian WER |
| --- | ---: | ---: | ---: |
| Whisper Small FT | 52.38 | 51.37 | 41.97 |
| Whisper Small Curriculum | 35.61 | 29.51 | 30.25 |
| Whisper Medium FT | 37.85 | 41.07 | 22.31 |
| Whisper Medium Curriculum | 32.47 | 21.17 | 20.42 |
| Whisper Large-v3 FT | 33.93 | 33.93 | 15.31 |
| Whisper Large-v3 Curriculum | 24.89 | 17.69 | 14.56 |

*WER values are reported as percentages.*

The proposed curriculum consistently outperforms conventional fine-tuning across all evaluated dialects and model scales.

---

## Models and Checkpoints

The trained Whisper models and experimental checkpoints accompanying the paper are available in the **[ilsp/Dialectal_Adaptation Hugging Face repository](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation)**.

| Experiment family | Released artifacts |
| --- | --- |
| Main curriculum learning | [Curriculum checkpoints](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation/tree/main/experiments/curriculum) |
| Cypriot curriculum-ratio experiments | [Ratio experiments](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation/tree/main/experiments/cy_ratios) |
| Curriculum without translation | [ASR-only curriculum experiments](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation/tree/main/experiments/curriculum_without_translation) |
| Curriculum without Standard Greek supervision | [Donor-language ablations](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation/tree/main/experiments/curriculum_without_GPC) |
| Fine-tuning baselines (Cretan and Messenian) | [Fine-tuned checkpoints](https://huggingface.co/datasets/ilsp/Dialectal_Adaptation/tree/main/experiments/fine_tuned) |
| Fine-tuning baselines (Cypriot) | [Cypriot Greek Whisper models](https://huggingface.co/ilsp/Cypriot_Greek_whisper) |

For detailed checkpoint-level results and comparisons between paper scores and released checkpoints, see the README on Hugging Face.

---

## Dataset Source

The experiments use dialectal speech data alongside Standard Modern Greek donor-language supervision.

**Cypriot Greek — Dataset source:** [**Mozilla Data Collective — Cypriot Greek Spontaneous Speech**](https://datacollective.mozillafoundation.org/datasets/cmflnuzz3xq2ok28b4o1ojacv).

The original Cypriot Greek speech recordings are **not redistributed** in the Hugging Face `ilsp/Dialectal_Adaptation` repository. Please obtain the source data directly from Mozilla Data Collective and follow the provider's access and usage terms.

---

## Repository Structure

```text
Dialect-Adaptation/
├── curriculum_learning_adaptation/
│   ├── curriculum_learning/
│   └── curriculum_learning_without_translation_3stages/
├── data_creation/
├── fine_tuning/
├── evaluation/
│   ├── evaluation_curriculum/
│   └── evaluation_for_fine_tuning/
└── README.md
```
---

## Code Organization and Reproducibility

The trained Whisper models, experimental checkpoints, and supporting artifacts accompanying this work are available on https://huggingface.co/datasets/ilsp/Dialectal_Adaptation .

| Code component | Inputs | Outputs |
| --- | --- | --- |
| [`data_creation/`](data_creation/) | Source speech recordings, transcripts, and translations | Prepared datasets for training and evaluation |
| [`curriculum_learning_adaptation/curriculum_learning/`](curriculum_learning_adaptation/curriculum_learning/) | Pretrained Whisper model and stage-specific ASR/ST datasets | Stage-wise curriculum models and checkpoints |
| [`curriculum_learning_adaptation/curriculum_learning_without_translation_3stages/`](curriculum_learning_adaptation/curriculum_learning_without_translation_3stages/) | Pretrained Whisper model and staged ASR datasets | Checkpoints from the no-translation curriculum variant |
| [`fine_tuning/`](fine_tuning/) | Pretrained Whisper model and dialectal ASR training data | Fine-tuned model checkpoints |
| [`evaluation/`](evaluation/) | Trained checkpoints and corresponding evaluation datasets | WER/CER results and model predictions |

### Curriculum training: inputs and outputs

The curriculum training code loads a pretrained Whisper model and prepared, stage-specific datasets with Hugging Face `load_from_disk()`. The dataset inputs for the four-stage configuration are organized as `stage_0`, `stage_1`, `stage_2`, and `stage_3`. Each stage combines audio inputs and text targets according to its ASR or ST objective. A stage starts from the final model of the preceding stage.

In the provided training-script example, outputs are organized as:

```text
<experiment_output>/
├── stage_0/
│   ├── checkpoint-*/
│   └── final/
├── stage_1/
│   ├── checkpoint-*/
│   └── final/
├── stage_2/
│   ├── checkpoint-*/
│   └── final/
└── stage_3/
    ├── checkpoint-*/
    └── final/
```

The Hugging Face repository contains the selected checkpoints and artifacts released for the experiments; its directory layout may differ from the original training-output paths.

---

## Citation

If you use this code or the released experimental artifacts, please cite:

```bibtex
@inproceedings{klimi26_interspeech,
  title     = {{Beyond Standard Greek: Adapting Whisper for Greek Dialects through Curriculum Multitask Learning}},
  author    = {Antigoni Klimi and Dimitrios Damianos and Stavros Bompolas and Vivian Stamou and Stella Markantonatou and Vassilis Katsouros and Georgios Paraskevopoulos},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {3268--3272},
  doi       = {10.21437/Interspeech.2026-2567},
  issn      = {2958-1796}
}
```

**Paper:** https://doi.org/10.21437/Interspeech.2026-2567

