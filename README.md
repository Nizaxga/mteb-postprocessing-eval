# MTEB Post-Processing Evaluation Pipeline

My senior project.

This project evaluates different post-processing techniques applied to text embeddings using the MTEB (Massive Text Embedding Benchmark) evaluation suite.

## What it does

The tool takes pre-trained embedding models and applies various post-processing strategies before evaluating them on MTEB tasks.

### Supported techniques include

- **Label Projection**: Supervised projection using class information
  - Result embedding dim: number of labels in the dataset
- **PCA + All-But-The-Top (PCAP)**: Statistical compression for unsupervised tasks
  - Result embedding dim: half of original embedding size
- **Adaptive Post-Processor**: Learned projection layer using relational distillation
  - Result embedding dim: fixed at 128
- **Disentangled Adaptive Post-Processor**: Learned projection with dimension independence
  - Result embedding dim: fixed at 128

### Currently evaluated datasets

- "Banking77Classification.v2": classification
- "EmotionClassification.v2": classification
- "STSBenchmark": STS
- "TwentyNewsgroupsClustering.v2": clustering
- "SprintDuplicateQuestions": pair_classification
- "AskUbuntuDupQuestions": reranking
- "NFCorpus": retrieval

## Setup

### Requirements

- Python 3.10+
- uv (package manager)

### Installation

```sh
# Clone or navigate to the project directory
# Install dependencies
uv sync
```

## Running

```sh
# Run the benchmark
uv run main.py
```

The script will:

1. Load the base embedding model
2. Apply post-processing strategies to each task according to task registry
3. Run MTEB evaluations and save results to the `results/` directory

Note: Results are cached locally to avoid re-running identical evaluations.

Note: Pair-classification and retrieval tasks are not recommended for running with limited GPU resources. Both tasks take significantly longer to evaluate compared to others.
