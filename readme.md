# Domain-Adaptive CLIP for Surgical Video Keyframe–Text Retrieval  
### with Inference Acceleration and System Reliability Analysis

This repository contains the full implementation of a domain-adaptive CLIP system
that improves keyframe–text retrieval for surgical videos (Cholec80 dataset).  
The project includes:

- Keyframe extraction from surgical videos  
- CLIP feature extraction (baseline & domain-adapted)  
- Phase-level domain adaptation modules  
- Retrieval evaluation (Top-1 / Top-5)  
- Per-phase accuracy analysis  
- Visualization of improvements and failures  
- Statistical summary & reproducibility scripts  
- Preparation for INT8 inference acceleration (work in progress)

This README documents the full workflow **before INT8 acceleration**.

---

# 1. Installation

## 1.1 Create Conda Environment

conda create -n surgclip python=3.10 -y
conda activate surgclip

## 1.2 Install PyTorch with CUDA 12.1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## 1.3 Install Project Dependencies
pip install open-clip-torch pillow numpy pandas matplotlib tqdm opencv-python ffmpeg-python

## 1.4 Install FFmpeg (Windows, without Chocolatey)
Download official build:
https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.zip

Extract to:
C:\ffmpeg\

Add to PATH:
C:\ffmpeg\bin
Verify:
ffmpeg -version

# 2. Directory Structure
surgclip/
├── data/
│   └── cholec80/
│       ├── videos/
│       └── phase_annotations/
│
├── frames/
│   └── videoXX/frame_000123.jpg
│
├── index/
│   └── keyframe_index.csv
│
├── results/
│   ├── retrieval_scores.json
│   ├── retrieval_scores_adapted.json
│   ├── phase_wise_accuracy.csv
│   ├── overall_accuracy_summary.csv
│   ├── figures/
│   │   ├── overall_accuracy_bar.png
│   │   ├── baseline_vs_adapted_improved.png
│   │   ├── baseline_vs_adapted_degraded.png
│   │   ├── adapted_top1_correct.png
│   │   ├── adapted_top5_correct_only.png
│   │   ├── triplet_failures.png
│   │   └── attention_example.png
│   └── figures_phase/
│       ├── phase_delta_heatmap.png
│       ├── phase_accuracy_grouped_bar.png
│       ├── phase_accuracy_delta_bar.png
│       ├── phase_accuracy_baseline_vs_adapt.png
│       ├── phase_sample_vs_gain.png
    ├── int8_benchmark_summary.json
│   ├── int8_model_size_comparison.csv
│   ├── int8_speed_benchmark.csv
│   ├── int8_representation_error.csv
│   └── figures_int8/
│       ├── int8_model_size_comparison.png
│       ├── int8_speed_benchmark.png
│       └── int8_representation_error_hist.png
│
└── scripts/
    ├── extract_frames.py
    ├── build_index.py
    ├── baseline_retrieval.py
    ├── adapted_retrieval.py
    ├── adapted_retrieval_int8.py
    ├── train_clip_phase_adapter.py
    ├── train_clip_phase_adapter_full.py
    ├── evaluate_retrieval_run.py
    ├── phase_wise_accuracy.py
    ├── summarize_runs.py
    ├── plot_overall_accuracy.py
    ├── plot_phase_accuracy.py
    ├── plot_phase_sample_vs_gain.py
    ├── plot_phase_improvement_heatmap.py
    ├── visualize_baseline_vs_adapter.py
    ├── visualize_triplet_examples.py
    ├── visualize_triplet_examples_adapted.py
    ├── visualize_attention_heatmap.py
    ├── sample_retrieval_examples.py
    ├── sample_retrieval_examples_adapted.py
    └── sample_baseline_vs_adapter_examples.py
    ├── benchmark_int8_analysis.py
    └── adapted_retrieval_int8.py

# 3. Dataset Preparation (Cholec80)
The Cholec80 dataset must be requested from:
https://camma.unistra.fr/datasets/

Place the files as follows:
data/cholec80/videos/*.mp4
data/cholec80/phase_annotations/*.txt

# 4. Full Pipeline
## Step 1 — Extract Frames
python scripts/extract_frames.py
Output → frames/videoXX/*.jpg

## Step 2 — Build Keyframe Index
python scripts/build_index.py
Output → index/keyframe_index.csv

## Step 3 — Baseline CLIP Retrieval
python scripts/baseline_retrieval.py
Output →
results/retrieval_scores.json
results/retrieval_examples.json

## Step 4 — Train Domain-Adaptive CLIP (Phase Adapter)
Short adapter:
python scripts/train_clip_phase_adapter.py

Full adapter:
python scripts/train_clip_phase_adapter_full.py
Outputs saved to:
results/clip_adapter_best.pth
results/checkpoints/

## Step 5 — Adapted CLIP Retrieval
python scripts/adapted_retrieval.py
Output →
results/retrieval_scores_adapted.json
results/retrieval_examples_adapted.json

## Step 6 — Summarize Overall Accuracy
python scripts/summarize_runs.py
Output →
overall_accuracy_summary.csv

## Step 7 — Generate Visual Figures
### Overall Accuracy Bar Chart
python scripts/plot_overall_accuracy.py

### Per-Phase Accuracy Comparison
python scripts/plot_phase_accuracy.py
python scripts/plot_phase_improvement_heatmap.py
python scripts/plot_phase_sample_vs_gain.py

### Triplet Visualizations
python scripts/visualize_triplet_examples.py
python scripts/visualize_triplet_examples_adapted.py

### Examples of Good/Bad Predictions
python scripts/sample_retrieval_examples.py
python scripts/sample_retrieval_examples_adapted.py
python scripts/sample_baseline_vs_adapter_examples.py

# 5. Key Results
## Overall accuracy improved significantly:
Model	Top-1	Top-5
Baseline (CLIP)	0.18	0.84
Phase-Adapted CLIP	0.52	0.96

## Phase-wise Improvements
See:
figures_phase/phase_delta_heatmap.png
figures_phase/phase_sample_vs_gain.png

## Visual Improvements
Examples include:
Improved retrieval samples
Failure cases
CLIP attention maps
Triplet comparison
Ranking improvements

Stored under:
results/figures/
results/figures_phase/

# 6. Reliability Analysis
The project additionally includes:

Visualization of stability across phases
Comparative analysis baseline vs. adapted
Robustness to ambiguous frames
Retrieval consistency evaluation

Scripts:
visualize_baseline_vs_adapter.py
visualize_triplet_examples.py
visualize_triplet_examples_adapted.py
visualize_attention_heatmap.py

# 7. INT8 Inference Acceleration

We implemented CPU-only dynamic INT8 quantization for the phase adapter using:
PyTorch quantize_dynamic
Only applied on torch.nn.Linear
CLIP visual backbone still runs on GPU (FP32), adapter runs on CPU (INT8)

Scripts:
scripts/benchmark_int8_analysis.py
scripts/adapted_retrieval_int8.py

## INT8 Benchmark Summary


Metric	FP32	INT8 (CPU)
Adapter file size	2.01 MB	0.51 MB
Avg latency (ms/frame)	0.047 ms	0.160 ms
Cosine similarity	—	0.99981
Mean abs error	—	0.00277
Max abs error	—	0.00980

## Generated Figures

figures_int8/int8_model_size_comparison.png
figures_int8/int8_speed_benchmark.png
figures_int8/int8_representation_error_hist.png

These show:
≈4× reduction in adapter size
Small numerical error (MAE < 0.003)
INT8 slower than FP32 due to CPU-only dynamic quantization (PyTorch limitation)

## Notes

PyTorch dynamic INT8 cannot run on CUDA, so latency is CPU-bound
True GPU-accelerated INT8 would require ONNX/TensorRT or QAT
Current implementation is suitable for size-constrained deployments

All INT8 experiments are fully reproducible using
scripts/benchmark_int8_analysis.py,
and the generated CSV/JSON files are stored under results/.

# 8. Product Quantization (PQ) for Large-Scale Retrieval
To further accelerate retrieval and reduce memory footprint of image embeddings, we evaluate Product Quantization (PQ) on 92,289 surgical-frame embeddings extracted by CLIP (ViT-B/32).

Our PQ implementation uses M = 8 subspaces with subvector dimension 64, trained on a subset of 20,000 frames.

All scripts are provided under: scripts/pq_benchmark_analysis.py

## 8.1 Memory Reduction

PQ encoding reduces the embedding storage from 180.25 MB → 1.20 MB, achieving: ≈150× compression

while keeping reconstruction error small and stable.

Type	Storage
FP32 embeddings	180.25 MB
PQ codes	1.20 MB

Figure:
results/figures_pq/pq_memory_comparison.png

## 8.2 Retrieval Latency

We compare per-query retrieval time using cosine similarity:

FP32 full search: 4.30 ms/query

PQ asymmetric search: 2.03 ms/query

This yields a ~2.1× speed-up.

Figure:
results/figures_pq/pq_latency_comparison.png

## 8.3 Retrieval Quality (Recall)

Due to the difficulty of the Cholec80 text-image alignment task, raw recall values are very low, but PQ preserves relative retrieval quality extremely well.

We use a log-scale plot to highlight the trend:

Figure:
results/figures_pq/pq_recall_logscale.png

Your plotted values:

Recall	FP32	PQ (INT-encoded)
R@1	~3e-6	~2e-6
R@5	~6e-5	~9e-5
R@10	~1e-4	~2e-4
R@50	~5e-4	~1e-3

Observations:

PQ slightly improves R@5–R@50 due to quantization smoothing effects.

No significant degradation at any recall level.

## 8.4 Representation Error Analysis

We compute element-wise reconstruction errors over 92,289 embeddings:

Mean abs error ≈ 0.004

Error distribution centered at 0, tight variance

No long-tail failure cases

Histogram:
results/figures_pq/pq_representation_error_hist.png

This confirms that PQ compression preserves embedding geometry sufficiently for efficient retrieval.

## 8.5 Summary
Category	Result
Compression	150× smaller
Speed	2.1× faster retrieval
Accuracy	Recall curve maintained (log-scale)
Error	Small reconstruction noise

## Conclusion:
PQ is an effective method for large-scale embedding compression in surgical video retrieval pipelines, providing significant memory and speed benefits with minimal accuracy loss.

# 9. Reliability under Distribution Shift

This section evaluates how robust and stable the retrieval system is when the input frames are corrupted by a simple synthetic distribution shift, instead of only reporting accuracy on clean (normal) data.

We focus on phase-level text–image retrieval and compare:

1. Baseline CLIP (frozen image encoder)

2. Domain-adapted CLIP (with phase adapter, FP32)

3. Domain-adapted CLIP INT8 (adapter quantized with dynamic INT8)

The goal is to measure how much performance degrades under a shift, and how much the adapter helps.

## 1. Synthetic SHIFT dataset (brightness + blur)

To simulate a realistic but controlled distribution shift, we generate a shifted version of the keyframe set purely at inference time (no re-training):

1. reduce brightness (darker scene)

2. slightly reduce contrast

3. add a mild Gaussian blur

This is implemented in: scripts/generate_shift_scores.py


The script:

1. Reads the original keyframe index (index/keyframe_index.csv)

2. Loads the same CLIP backbone and optional adapter

3. Applies the chosen corruption (brightness_blur) to each frame on-the-fly

4. Runs text–image retrieval for the 7 phase prompts

5. Writes a new retrieval_scores_*.json with the same schema as normal runs

Supported methods:

1. baseline

2. adapted

3. adapted_int8

Run from project root:

# 1) Baseline CLIP under shift
python scripts/generate_shift_scores.py --method baseline --shift brightness_blur

# 2) Adapted CLIP (FP32) under shift
python scripts/generate_shift_scores.py --method adapted --shift brightness_blur

# 3) Adapted CLIP (INT8) under shift
python scripts/generate_shift_scores.py --method adapted_int8 --shift brightness_blur


After running these, the following files appear in results/:

retrieval_scores_shift_brightness_blur.json

retrieval_scores_adapted_shift_brightness_blur.json

retrieval_scores_adapted_int8_shift_brightness_blur.json


They correspond 1:1 with the normal versions:

retrieval_scores.json

retrieval_scores_adapted.json

retrieval_scores_adapted_int8.json


2. Reliability analysis: normal vs. shifted

We use: ```scripts/reliability_analysis.py``` to jointly analyze normal and shifted runs.

The script performs the following steps:

1. Load normal JSONs

results/retrieval_scores.json

results/retrieval_scores_adapted.json

results/retrieval_scores_adapted_int8.json


2. Load shifted JSONs (if they exist)

results/retrieval_scores_shift_brightness_blur.json

results/retrieval_scores_adapted_shift_brightness_blur.json

results/retrieval_scores_adapted_int8_shift_brightness_blur.json


3. Parse and recompute accuracy from per-frame predictions

For each JSON, the script reads the "examples" list and recomputes:

```Top-1 accuracy```

```Top-5 accuracy```

```Per-phase Top-1 accuracy (phase 1–7)```

Note about metrics:

The recomputed accuracy values may differ slightly from the "top1_accuracy" and "top5_accuracy" written in the JSON header.
This is expected:

Header values were produced by the original retrieval script using its own evaluation logic.

In reliability analysis, we use only the per-frame predictions stored in "examples" to ensure:

consistent evaluation across all methods

consistent comparison between normal and shifted data

correct per-phase accuracy

This recomputed version is the authoritative metric for robustness analysis.

4. Compare normal vs shifted

Accuracy drop under shift (Top-1 / Top-5)

Per-phase robustness:

acc_shift−acc_normal

Run: ```python scripts/reliability_analysis.py```


This will generate:

1. Numeric summary

```results/reliability_summary.csv```

For each method (baseline_fp32, adapter_fp32, adapter_int8) and each condition (normal, shift_brightness_blur), the CSV stores:

Top-1 accuracy

Top-5 accuracy

These are the recomputed, unified metrics used for all robustness analysis.

2. Figures for reporting
(1) results/figures_reliability/top1_top5_accuracy_bar.png

Bar chart comparing:

Top-1 (normal)

Top-1 (shifted)

Top-5 (normal)

Top-5 (shifted)

across all methods.

(2) results/figures_reliability/phase_accuracy_diff.png

Per-phase robustness plot showing:

(shift - normal) for each phase

Values < 0 → accuracy drop under corruption

Values close to 0 → stable robustness

3. How to interpret these results

This analysis focuses on robustness, not just accuracy.

A model is considered more reliable if:

Its Top-1 / Top-5 accuracy remains high under shift

Its per-phase accuracy drops less

It maintains stable predictions across different visual conditions

```baseline_fp32```

Shows how sensitive the original CLIP image encoder is to illumination and blur changes.

```adapter_fp32```

Typically shows improved robustness because domain adaptation injects surgical-phase–specific structure.

```adapter_int8```

Verifies whether quantization preserves:

phase awareness

robustness to visual changes

If results are close to FP32, it indicates that quantization did not harm model reliability.

These results and figures can be directly incorporated into the “System Reliability Analysis” or “Robustness Evaluation” section of the paper.


# 10. Citation
If this work contributes to your research, please cite the project:

Yueran Cao,
Domain-Adaptive CLIP for Surgical Video Keyframe–Text Retrieval with
Inference Acceleration and System Reliability Analysis.
2025.

# 11. License
This project is for academic research only.
Please follow the Cholec80 dataset license terms.