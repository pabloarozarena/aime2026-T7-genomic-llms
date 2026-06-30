# AIME 2026 T7 Genomic LLMs: From LLMs to DNA

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AIME 2026](https://img.shields.io/badge/AIME-2026-purple)]()

This repository contains the hands-on material for the **AIME 2026 tutorial “From LLMs to DNA”**.  
The tutorial introduces the main steps required to build, adapt, and use **Genomic Language Models (gLMs)** for biological sequence analysis, starting from DNA tokenization and masked language modeling, and moving toward downstream representation learning and SARS-CoV-2 variant prioritization.

The repository is designed as a practical, notebook-based learning resource. It includes example datasets, pretrained lightweight MiniBERT checkpoints, custom genomic tokenizers, and three guided notebooks.

---

## Overview

Large Language Models can be adapted to biological sequences by treating DNA as a structured symbolic language. In this repository, DNA sequences are represented using either:

- **character-level tokenization**, where each nucleotide is treated as one token;
- **k-mer tokenization**, where fixed-length nucleotide fragments are used as tokens.

The notebooks show how to:

1. build custom genomic tokenizers;
2. train a compact BERT-style masked language model from scratch;
3. fine-tune or continually adapt the model to new genomic domains;
4. extract latent sequence embeddings from trained models;
5. evaluate whether embeddings capture biological signals such as GC content and exon/intron separation;
6. use a SARS-CoV-2-adapted MiniBERT model to prioritize Omicron Spike mutations through a delta log-likelihood score.

---

## Repository structure

```text
aime2026-T7-genomic-llms/
├── data/
│   ├── chr22.csv
│   ├── sarscov2.csv.gz
│   └── sarscov2_reference.fasta
│
├── models/
│   ├── character/
│   │   ├── tokenizer/
│   │   └── weights/
│   │
│   ├── kmer_k3_s1/
│   ├── kmer_k3_s3/
│   ├── kmer_k4_s1/
│   ├── kmer_k4_s2/
│   ├── kmer_k5_s1/
│   ├── kmer_k5_s2/
│   ├── kmer_k6_s1/
│   ├── kmer_k6_s3/
│   ├── kmer_k8_s4/
│   │   ├── tokenizer/
│   │   └── weights/
│   │
│   └── sarscov2/
│       ├── character/
│       │   ├── tokenizer/
│       │   └── weights/
│       └── kmer/
│           ├── tokenizer/
│           └── weights/
│
├── notebooks/
│   ├── notebook1.ipynb
│   ├── notebook2.ipynb
│   └── notebook3.ipynb
│
├── .gitignore
└── README.md
```

---

## Notebooks

### Notebook 1 — Training a Custom Genomic BERT from Scratch

`notebooks/notebook1.ipynb`

This notebook introduces the full training pipeline for a compact genomic BERT model.

Main steps:

- load a human coding-sequence dataset from Hugging Face;
- normalize DNA sequences;
- choose between character-level and k-mer tokenization;
- build a custom tokenizer from scratch;
- tokenize train, validation, and test splits;
- define a lightweight MiniBERT architecture for masked language modeling;
- train the model with dynamic masking;
- visualize the training loss;
- save tokenizer files and model weights;
- demonstrate fine-tuning and continual learning workflows.

The notebook is useful if you want to understand how genomic language models are built from the ground up.

---

### Notebook 2 — Genomic Sequence Embeddings and Downstream Tasks

`notebooks/notebook2.ipynb`

This notebook uses pretrained tokenizer and checkpoint assets from the `models/` folder to extract sequence-level embeddings from Chromosome 22 examples.

Main steps:

- select a pretrained model option, for example `character`, `kmer_k6_s3`, or another k-mer configuration;
- reconstruct the same MiniBERT architecture used during training;
- load model weights from the repository;
- tokenize the `data/chr22.csv` dataset;
- extract 128-dimensional embeddings using mean pooling over the final hidden states;
- reduce embeddings to two dimensions using PCA;
- visualize the embedding manifold by GC content;
- visualize exon/intron separation;
- train a downstream gradient boosting probe on the extracted embeddings.

This notebook is useful for showing how a genomic language model can act as a representation learner for downstream biological tasks.

---

### Notebook 3 — SARS-CoV-2 Variant Prioritization Using Genomic Language Models

`notebooks/notebook3.ipynb`

This notebook applies a SARS-CoV-2-adapted MiniBERT model to Omicron variant sequences.

Main steps:

- load the Wuhan-Hu-1 reference genome from `data/sarscov2_reference.fasta`;
- load SARS-CoV-2 variant sequences from `data/sarscov2.csv.gz`;
- select clean Omicron sequences without ambiguous bases;
- align Omicron Spike regions against the reference sequence;
- identify nucleotide substitutions;
- map mutations to SARS-CoV-2 genomic regions;
- infer synonymous and non-synonymous amino acid changes;
- mask mutation sites in local sequence context;
- compute a delta log-likelihood score:

```math
\Delta \log P =
\log P(\text{mutated token} \mid \text{context})
-
\log P(\text{reference token} \mid \text{context})
```

More negative values indicate that the model assigns lower probability to the observed mutated token relative to the reference token in the same context. In this tutorial, the score is used as a model-based prioritization signal for mutation-level exploration, not as a standalone clinical or epidemiological predictor.

---

## Data

The `data/` folder contains small tutorial-ready data assets:

| File | Description |
|---|---|
| `chr22.csv` | Chromosome 22 examples used for embedding extraction and downstream probing. |
| `sarscov2.csv.gz` | Compressed SARS-CoV-2 variant dataset used in the variant prioritization notebook. |
| `sarscov2_reference.fasta` | Wuhan-Hu-1 reference sequence used for mutation mapping. |

Notebook 1 also downloads the `gonzalobenegas/human-genome-cds` dataset from Hugging Face during execution.

---

## Model assets

The `models/` directory stores pretrained lightweight MiniBERT assets organized by tokenization strategy.

Each general genomic model folder contains:

```text
tokenizer/
├── tokenizer.json
└── tokenizer_config.json

weights/
└── model_state_dict.pt
```

Available general model options include:

```text
character
kmer_k3_s1
kmer_k3_s3
kmer_k4_s1
kmer_k4_s2
kmer_k5_s1
kmer_k5_s2
kmer_k6_s1
kmer_k6_s3
kmer_k8_s4
```

The SARS-CoV-2-specific models are stored separately:

```text
models/sarscov2/character/
models/sarscov2/kmer/
```

These include tokenizers and checkpoints adapted for the variant prioritization workflow in Notebook 3.

---

## MiniBERT architecture

The tutorial uses a compact BERT-style masked language model implemented with Hugging Face Transformers.

The default architecture used in the notebooks is:

```python
BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    max_position_embeddings=512,
)
```

This small architecture is intentionally chosen for tutorial settings, fast experimentation, and execution in environments such as Google Colab.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pabloarozarena/aime2026-T7-genomic-llms.git
cd aime2026-T7-genomic-llms
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install jupyter datasets pandas numpy torch transformers tokenizers scikit-learn matplotlib tqdm openpyxl biopython
```

The notebooks also include installation cells, so they can be executed directly in Google Colab.

---

## Running the tutorial

Start Jupyter:

```bash
jupyter notebook
```

Then run the notebooks in the following order:

```text
1. notebooks/notebook1.ipynb
2. notebooks/notebook2.ipynb
3. notebooks/notebook3.ipynb
```

Notebook 1 can be used to train models from scratch.  
Notebook 2 and Notebook 3 can also be run independently because they load pretrained tokenizers and weights from the repository.

---

## Key configuration options

### Notebook 1

Choose the tokenization strategy:

```python
TOKENIZATION_STYLE = "character"  # options: "character" or "kmer"
K_SIZE = 6
STRIDE = 3
```

### Notebook 2

Choose which pretrained model to load:

```python
MODEL_OPTION = "kmer_k6_s3"
```

Available values:

```text
character
kmer_k3_s1
kmer_k3_s3
kmer_k4_s1
kmer_k4_s2
kmer_k5_s1
kmer_k5_s2
kmer_k6_s1
kmer_k6_s3
kmer_k8_s4
```

### Notebook 3

Choose the SARS-CoV-2 model:

```python
MODEL_CHOICE = "character"  # options: "character" or "kmer"
```

---

## Expected outputs

Depending on the notebook, expected outputs include:

- trained tokenizer files;
- MiniBERT model checkpoints;
- masked language modeling loss curves;
- 128-dimensional genomic sequence embeddings;
- PCA visualizations of embedding spaces;
- GC-content gradient plots;
- exon/intron separation plots;
- downstream classification results;
- SARS-CoV-2 mutation prioritization tables with delta log-likelihood scores.

---

## Reproducibility notes

Some cells use fixed random seeds or deterministic sampling, for example `random_state=42`, to make tutorial outputs easier to reproduce.

Results may still vary depending on:

- GPU/CPU hardware;
- PyTorch and CUDA versions;
- package versions;
- execution environment;
- stochasticity during training.

For stronger reproducibility, consider pinning package versions in a `requirements.txt` or `environment.yml` file.

---

## Troubleshooting

### CUDA is not available

The notebooks automatically fall back to CPU when CUDA is unavailable. Training will be slower on CPU, but inference and small-scale tutorial execution should still work.

### Tokenizer or checkpoint cannot be loaded

Check that the selected model name matches one of the folders in `models/`. For example:

```python
MODEL_OPTION = "kmer_k6_s3"
```

must correspond to:

```text
models/kmer_k6_s3/
```

### Large model or data files

Some checkpoints and compressed datasets may be large. If the repository grows, consider using Git LFS for binary model files and large data assets.

### Ambiguous SARS-CoV-2 bases

Notebook 3 removes Omicron sequences containing `N` or `n` before mutation scoring. This is intentional, because ambiguous bases can affect tokenization and masked language modeling probabilities.

---

## Scientific scope and limitations

This repository is intended for education and research prototyping.

The models and scores generated here should not be used as standalone tools for:

- clinical decision-making;
- patient diagnosis;
- epidemiological surveillance;
- public health policy;
- variant risk classification.

The SARS-CoV-2 delta log-likelihood score is a model-derived prioritization signal. It should be interpreted together with biological annotation, epidemiological evidence, experimental data, and domain expertise.

---

## Suggested citation

If you use this repository in academic work, please cite it as:

```bibtex
@misc{aime2026_t7_genomic_llms,
  title        = {AIME 2026 T7 Genomic LLMs: From LLMs to DNA},
  author       = {{AIME 2026 T7 Genomic LLMs Tutorial Authors}},
  year         = {2026},
  howpublished = {\url{https://github.com/pabloarozarena/aime2026-T7-genomic-llms}},
  note         = {Tutorial repository}
}
```

If this repository accompanies a paper, workshop, or tutorial proceedings entry, please cite the official publication once available.

---

## Acknowledgments

This tutorial uses open-source tools from the Python scientific and machine learning ecosystem, including:

- PyTorch;
- Hugging Face Transformers;
- Hugging Face Datasets;
- Tokenizers;
- scikit-learn;
- Biopython;
- pandas;
- NumPy;
- matplotlib.

---

## Contributing

Contributions are welcome.

Possible contributions include:

- improving notebook readability;
- adding missing documentation;
- adding environment files;
- improving reproducibility;
- extending downstream genomic tasks;
- adding additional genomic tokenization strategies;
- reporting bugs or broken links.

Please open an issue or submit a pull request.

---

## License

This project is released under the MIT License. See the [`LICENSE`](LICENSE) file for details.

Unless otherwise stated, the MIT License applies to the code and tutorial material in this repository. External datasets, model checkpoints, and third-party libraries may be subject to their own licenses and terms of use.
