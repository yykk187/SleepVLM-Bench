<div align="center">
  <h2><b> SleepVLM-Bench: A Systematic Benchmark for Clinical Sleep Staging with Vision-Language Models </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/yourname/SleepVLM-Bench?color=green)
![](https://img.shields.io/github/stars/yourname/SleepVLM-Bench?color=yellow)
![](https://img.shields.io/github/forks/yourname/SleepVLM-Bench?color=lightblue)

</div>

<p align="center">
<img src="./figures/main_framework.png" >
</p>

> 🌟 Please let us know if you find any mistakes or have suggestions!

---

## ⚡ What is SleepVLM-Bench?

**SleepVLM-Bench** is the first systematic benchmark for evaluating **Vision-Language Models (VLMs)** in **clinical sleep staging**.

It studies whether general-purpose multimodal foundation models can understand and reason over physiological sleep signals when PSG data are converted into VLM-friendly forms such as:

- **waveform images**
- **expert textual features**
- **signal sequences**

Our goal is to rigorously test the real capability boundary of VLMs in this clinically important task.

---

## 📑 Benchmark Overview

We evaluate VLMs on **three PSG-based input modalities**:

| Modality | Description |
| :--- | :--- |
| **Image Input** | Stacked EEG / EOG / EMG epoch images |
| **Expert Text Features** | Rule-based natural language descriptions of sleep events |
| **Sequence Input** | Raw / discrete EEG sequence representations |

<img src="./figures/dataset_overview.png" width="700" >

---

## 🗂 Datasets

SleepVLM-Bench is evaluated on three public datasets:

- **DCSM**https://sid.erda.dk/wsgi-bin/ls.py?share_id=fUH3xbOXv8
- **SHHS**https://sleepdata.org/datasets/shhs
- **ISRUC**https://sleeptight.isr.uc.pt

These datasets cover different subject populations, acquisition conditions, and clinical variability, enabling both **in-domain** and **cross-dataset** evaluation.

---

## 🧠 Clinical Rule Guidance

To support interpretable prompting and evaluation, we organize physician-guided sleep staging rules based on the **AASM standard**, including:

- W / N1 / N2 / N3 / REM stage criteria
- waveform event definitions such as:
  - sleep spindles
  - K-complexes
  - slow waves
  - alpha rhythm
  - low-amplitude mixed-frequency activity

<img src="./figures/stage_rules.png" width="700" >

---

## 📊 Main Findings

Our experiments reveal a clear conclusion:

- Current VLMs achieve only **~20%–24% ACC** on 5-class sleep staging
- Performance is often **close to random guessing**
- Expert prompts provide **limited and unstable gains**
- Strong task-specific models still outperform VLMs by a large margin

This means that even powerful models such as **GPT-4o** still cannot reliably replace specialized sleep staging models.

---

## ❗ Key Takeaway

A central contribution of this benchmark is a **reproducible negative result**:

> Current VLMs are still not sufficient for reliable clinical sleep staging.

Rather than being a limitation, this helps define the current boundary of foundation models in:

- physiological signal understanding
- subtle waveform event detection
- clinically trustworthy reasoning

---

## Key Features

| Dataset | W | N1 | N2 | N3 | REM | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DCSM** | 453 | 585 | 606 | 486 | 572 | **2702** |
| **SHHS** | 327 | 494 | 605 | 472 | 492 | **2390** |
| **ISRUC** | 460 | 443 | 499 | 447 | 400 | **2249** |
---

## 📌 Contributions

- **SleepVLM-Bench**: the first benchmark for VLM-based clinical sleep staging
- **Three input paradigms**: image, expert text, and sequence
- **Unified evaluation** on DCSM, SHHS, and ISRUC
- **A reproducible negative finding** that clarifies the capability boundary of current VLMs

---
