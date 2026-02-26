<div align="center">
  <h2><b> SleepVLM-Bench: A Systematic Benchmark for Evaluating Vision-Language Models in Clinical Sleep Staging </b></h2>
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

**SleepVLM-Bench** is the first systematic benchmark designed to evaluate **Vision-Language Models (VLMs)** for **clinical sleep staging**.

Clinical sleep staging is a cornerstone of sleep medicine, but it also poses a fundamental challenge for modern AI systems: models must reason over **long-duration physiological time series**, integrate **multi-channel multimodal PSG signals**, and detect **fine-grained waveform events** under noisy clinical conditions.

Recent multimodal foundation models such as GPT-4o have shown strong general reasoning and multimodal understanding abilities. This raises an important question:

> **Can we transform physiological signals into VLM-friendly modalities (e.g., images or text) and use general-purpose VLMs for reliable clinical sleep staging?**

SleepVLM-Bench is built to answer this question in a rigorous and standardized way.

---

## 🔍 Why is this benchmark needed?

Although end-to-end deep learning models for sleep staging have achieved strong performance, they still suffer from:

- limited generalization across centers and devices,
- heavy dependence on expert annotations,
- weak interpretability in clinical settings.

At the same time, the potential of VLMs in this domain has **not been systematically validated**.

SleepVLM-Bench fills this gap by providing a unified framework to test whether current VLMs can truly understand and reason over sleep-related physiological signals, rather than simply producing plausible language outputs.

---

# 📑 Benchmark Composition

Our benchmark evaluates VLMs on **three VLM-compatible input modalities** constructed from overnight PSG recordings:

<img src="./figures/dataset_overview.png" width="700" >

## Input Modalities

| Modality | Description | Evaluated Ability |
| :--- | :--- | :--- |
| **Image Input** | 30-second stacked waveform images of EEG, EOG, and EMG | Vision-language joint understanding of physiological waveforms |
| **Textual Expert Features** | Natural language descriptions of EEG/EOG/EMG events extracted based on clinical rules | Clinical reasoning and knowledge-guided decision making |
| **Raw / Discrete Sequence Input** | Digitized signal sequence representation for LLM-style modeling | Long-context temporal reasoning over physiological sequences |

These three modalities allow us to systematically analyze the trade-offs among:

- **visual representation**,
- **text-based clinical prompting**,
- **sequence modeling for long physiological contexts**.

---

## 🧠 Physician-Guided Clinical Rules

To support structured prompting and interpretable evaluation, we collaborated with certified sleep specialists to organize sleep staging rules based on the **AASM guideline**.

<img src="./figures/stage_rules.png" width="700" >

The rule system includes:

- **Sleep Stage Classification Rules** for W, N1, N2, N3, and REM,
- **Waveform Identification Rules** for events such as:
  - slow waves,
  - alpha rhythm,
  - low-amplitude mixed-frequency activity,
  - vertex waves,
  - sleep spindles,
  - K-complexes.

These rules are used for:

1. **data quality control**,  
2. **text prompt construction**,  
3. **interpretable error analysis**.

---

# 🗂 Datasets

SleepVLM-Bench evaluates models on three public datasets with different clinical centers, subject populations, and acquisition conditions:

| Dataset | Description |
| :--- | :--- |
| **DCSM** | Clinical sleep dataset with relatively balanced stage distribution |
| **SHHS** | Large-scale public sleep dataset with broader subject diversity |
| **ISRUC** | More challenging dataset with stronger heterogeneity and domain shift |

## Sleep Stage Distribution

| Dataset | W | N1 | N2 | N3 | REM | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DCSM** | 453 | 585 | 606 | 486 | 572 | 2702 |
| **SHHS** | 327 | 494 | 605 | 472 | 492 | 2390 |
| **ISRUC** | 460 | 443 | 499 | 447 | 400 | 2249 |

To reduce severe class imbalance in overnight PSG data, we adopt **class-balanced sampling** during image dataset construction and mini-batch training.

---

# ⚙️ Signal-to-Image Construction

For image-based VLM evaluation, each PSG recording is processed as follows:

- remove power-line interference,
- resample all signals to **100 Hz**,
- apply modality-specific band-pass filters:
  - **EEG:** 0.3–35 Hz
  - **EOG:** 0.1–10 Hz
  - **EMG:** 10–45 Hz
- apply anti-aliasing low-pass filtering before resampling,
- segment into **non-overlapping 30-second epochs**,
- exclude unlabeled or movement epochs,
- render each epoch as a stacked image:
  - **EEG** on top,
  - **EOG** in the middle,
  - **EMG** at the bottom.

<img src="./figures/epoch_example.png" width="700" >

This layout follows the standard clinical reading order and preserves waveform amplitude without inter-record normalization.

---

# 🧪 Experimental Settings

We evaluate current VLMs under unified subject-level splits and challenging cross-dataset settings.

## Compared VLMs

- GPT-4o
- Gemma-3-27b-it
- Qwen2.5VL-7b
- QwenQVQ-72b
- Llama-3.2-11B
- Llava-1.5-7b

## Deep Learning Baselines

- DeepSleep
- AttnSleep

## Sequence LLM Baseline

- ChatTS-14B

---

# 📊 Main Findings

## 1. Current VLMs perform near chance level on clinical sleep staging

Across DCSM, SHHS, and ISRUC, VLMs achieve only:

- **ACC:** ~20%–24%
- **MF1:** ~15%–19%
- **Recall:** ~20%–22%

Since the task is a **5-class classification problem**, these numbers are close to random guessing, suggesting that current VLMs do **not** learn stable and clinically useful representations for this task.

---

## 2. Expert prompts bring limited but unstable gains

Adding structured clinical stage features can provide modest improvements on some datasets and models, especially:

- **Qwen2.5VL** on DCSM,
- partial improvements on **SHHS**.

However, the gains are:

- **small**,
- **model-dependent**,
- and sometimes **negative**.

This suggests that current VLMs cannot reliably align:

- **textual clinical priors**
with
- **low-level physiological waveform evidence**.

---

## 3. Strong task-specific models still outperform VLMs by a large margin

When compared with raw EEG sequence baselines:

| Dataset | Model | ACC | F1 | Rec |
| :--- | :--- | ---: | ---: | ---: |
| DCSM | ChatTS-14B | 21.63 | 17.19 | 20.60 |
| DCSM | DeepSleep | 65.23 | 66.73 | 66.52 |
| DCSM | AttnSleep | **69.34** | **70.52** | **71.22** |
| SHHS | ChatTS-14B | 18.95 | 18.33 | 20.53 |
| SHHS | DeepSleep | 59.01 | 59.66 | 60.57 |
| SHHS | AttnSleep | **63.64** | **64.91** | **64.33** |
| ISRUC | ChatTS-14B | 21.52 | 21.02 | 21.41 |
| ISRUC | DeepSleep | 48.06 | 48.46 | 47.67 |
| ISRUC | AttnSleep | **56.03** | **56.54** | **56.09** |

This shows that **general-purpose multimodal models are still far inferior to task-specific architectures with physiological inductive bias**.

---

## 4. Negative results are important: they define the boundary of current VLM capability

A key contribution of this project is a **reproducible negative finding**:

> Even the strongest current VLMs, including GPT-4o, fail to surpass specialized end-to-end models in clinical sleep staging.

This is not a failure of the benchmark—it is an important empirical result that clearly defines the current limits of VLMs for:

- low-SNR physiological data,
- subtle event detection,
- long temporal reasoning,
- clinically reliable decision making.

---

# 🧩 Failure Modes of VLMs

We observe several common failure patterns:

- relying on **weak non-specific cues** (e.g., low muscle tone, LAMF) instead of key events,
- confusing **N1 and N2** when sleep spindles or K-complexes are subtle,
- producing **well-structured explanations without true evidence grounding**,
- showing **prompt anchoring**, where the model follows the template but not the physiology.

<img src="./figures/discuss.jpg" width="700" >

These findings suggest that current VLMs still struggle to align:

- explicit clinical rules,
- underlying waveform evidence,
- and physiologically valid reasoning.

---

# 🚀 Benchmark Goals

SleepVLM-Bench aims to support future research in:

- physiological signal understanding with foundation models,
- multimodal prompting for clinical decision support,
- interpretable reasoning on biosignals,
- robust cross-center evaluation in sleep medicine,
- identifying the boundary between **general-purpose foundation models** and **task-specific medical AI systems**.

---

# 💻 Code

## Requirements

We recommend:

- Python 3.10+
- PyTorch
- Transformers
- Diffusers
- Accelerate
- PEFT

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running Pipeline

The benchmark includes the following core stages:

1. **Signal preprocessing** – filtering, resampling, epoch segmentation  
2. **Image construction** – stacked EEG/EOG/EMG rendering  
3. **Expert feature generation** – rule-based clinical feature extraction  
4. **Model evaluation** – VLM and sequence baseline testing  
5. **Cross-dataset analysis** – robustness and generalization assessment  

Example usage:

```bash
python preprocess.py
python build_image_dataset.py
python build_text_features.py
python eval_vlm.py
python eval_sequence.py
```

> 📖 More detailed documentation will be released soon.

---

# 📦 Project Structure

```bash
SleepVLM-Bench/
├── figures/
├── data/
├── scripts/
├── preprocess.py
├── build_image_dataset.py
├── build_text_features.py
├── eval_vlm.py
├── eval_sequence.py
├── requirements.txt
└── README.md
```

---

# 📌 Contributions

- **SleepVLM-Bench**: the first benchmark for VLM-based clinical sleep staging
- **Three PSG-to-VLM input paradigms**: image, expert text, and sequence
- **Unified cross-dataset evaluation** on DCSM, SHHS, and ISRUC
- **A reproducible negative result** that clarifies the current capability boundary of VLMs in this task

---

# 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourcitation2025sleepvlmbench,
  title={SleepVLM-Bench: A Systematic Benchmark for Evaluating Vision-Language Models in Clinical Sleep Staging},
  author={Your Name and Coauthors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

# 🙏 Acknowledgements

We thank the clinical sleep experts and collaborators who contributed to rule design, data verification, and interpretation of experimental findings.

---
