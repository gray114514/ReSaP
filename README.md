# ReSaP: Reasoning-enhanced and Scale-aware Prompting for Referring Remote Sensing Image Segmentation

**Official PyTorch implementation of the paper: "ReSaP: Reasoning-enhanced and Scale-aware Prompting for Referring Remote Sensing Image Segmentation" (Submitted to IEEE J-STARS)**

## 📖 Introduction

Referring Remote Sensing Image Segmentation (RRSIS) aims to segment target objects in complex remote sensing imagery based on natural language expressions. Existing Multimodal Large Language Models (MLLMs) often struggle with the**pixel-level perception bottleneck** and **scale granularity mismatch** inherent in satellite imagery.

To address these challenges, we propose **ReSaP**, a novel framework that synergizes high-level semantic reasoning with fine-grained visual perception. ReSaP introduces two core components:

1. **Pixel-Aware GRPO Training:** A reinforcement learning strategy that enforces **Chain-of-Thought (CoT)** reasoning, significantly enhancing spatial localization and pixel discrimination in complex backgrounds.

2. **Scale-Aware Prompting:** A dynamic sampling mechanism that adapts prompt granularity to object scales, resolving the mismatch issues of static prompting methods.

![Architecture](figures/inference.pdf)

## 🛠️ Environment Setup

### 1. Clone the repository

```bash

git clone https://github.com/gray114514/ReSaP.git
cd ReSaP
```

### 2. Create Environment

We recommend using Anaconda/Miniconda to manage dependencies.

```bash

# Create the environment from the yaml file (if provided)
conda env create -f environment.yaml

# Activate the environment
conda activate resap
```

If you install manually, the core dependencies include:

- Python ≥ 3.8

- PyTorch ≥ 2.0 (with CUDA support)

- Transformers (Hugging Face)

- Peft (for LoRA)

- DeepSpeed (for distributed training)

- Segment Anything (SAM)

- Bottle (for the reference server)

```bash

pip install torch transformers peft deepspeed bottle shapely
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 📂 Data Preparation

Please download the**RRSIS-D** and **RIS-LAD** datasets and organize them as follows:

```text

DATA_ROOT/
├── RRSISD/
│   ├── train/
│   ├── val/
│   └── test/
├── RIS_LAD/
│   ├── train/
│   └── test/
```

## 🚀 Training

We support two training stages: **Supervised Fine-Tuning (SFT)** for the baseline and **Pixel-Aware GRPO** for reasoning enhancement.

### 1. Supervised Fine-Tuning (SFT) Baseline

To train the SFT baseline (following the SAM4MLLM paradigm) using LoRA:

```bash

python sft.py
```

*Note: Please modify `sft.py` to set your `model_path` (Base MLLM path) and `DATA_ROOT` before running.*

### 2. Pixel-Aware GRPO Training (Ours)

To perform the reasoning-enhanced training using GRPO, you need to start the Reference Server first to handle reward computation, and then run the distributed training script.

#### Step 1: Start the Reference Server

```bash

python ref_server.py
```

#### Step 2: Start GRPO Training (Distributed)

Run the distributed training script using DeepSpeed (adjust GPU ids as needed):

```bash

deepspeed --include localhost:0,1,2,3 grpo_ref_split.py \
    --model_path <path_to_base_model> \
    --data_root <path_to_dataset> \
    --log_dir <path_to_save_logs> \
    --version_name "grpo_v1"
```

## ⚡ Evaluation

To evaluate the model on benchmarks (RRSIS-D or RIS-LAD) using our inference pipeline:

```bash

python eval.py \
    --option RRSISD \
    --eval_model_path <path_to_trained_model> \
    --data_root <path_to_dataset> \
    --sam_ckpt <path_to_sam_checkpoint> \
    --use_think True \
    --save_img False \
    --log_dir "eval_logs"
```

### Key Arguments:

- `--option`: Dataset option, e.g.,`RRSISD` or `ris_lad`.

- `--use_think`: Set to `True` to enable **Chain-of-Thought (CoT)** inference (ReSaP mode). Set to `False` for SFT baseline.

- `--sam_ckpt`: Path to the Segment Anything Model (SAM) checkpoint (e.g., `sam_vit_l_0b3195.pth`).

- `--save_img`: Set to `True` to save visualization results.

## 📊 Results

Our ReSaP framework achieves state-of-the-art performance on both RRSIS-D and RIS-LAD benchmarks.

(Detailed quantitative comparison tables can be found in our paper.)

## 📜 Citation

If you find this work helpful for your research, please cite our paper:

```bibtex

@article{lv2025resap,
  title={ReSaP: Reasoning-enhanced and Scale-aware Prompting for Referring Remote Sensing Image Segmentation},
  author={Lv, Ning and Wang, Teng and Dang, Jisheng and Liu, Yichu and Wang, Bimei and Chua, Tat-Seng},
  journal={Submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025}
}
```

## 📧 Contact

If you have any questions, please feel free to contact **Ning Lv** at `lvn2023@lzu.edu.cn`.
> （注：文档部分内容可能由 AI 生成）
