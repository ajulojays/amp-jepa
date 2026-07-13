# AMP-JEPA: Joint Embedding Predictive Architecture for Antimicrobial Peptide Design

AMP-JEPA is a self-supervised deep learning framework that combines **JEPA-style latent predictive learning** with **VAE** and **ESM2** embeddings for de novo design of antimicrobial peptides (AMPs).

---

## Motivation

Most current AMP generation methods rely on supervised signals or simple generative models. AMP-JEPA explores a different direction by:

- Learning rich latent representations of experimentally validated AMPs using self-supervised JEPA objectives.
- Enabling meaningful exploration of peptide latent space.
- Generating novel peptide candidates.
- Validating generated candidates using strong external predictive models (APEX).

---

## Key Features

- Uses experimentally validated AMP sequences from the APD database
- ESM2 protein language model embeddings
- JEPA-inspired latent predictive learning
- Variational Autoencoder (VAE) latent generation
- Integration with APEX antimicrobial activity predictor
- Focus on diversity, novelty, and broad-spectrum AMP discovery

---

## Project Structure

```text
amp-jepa/
├── src/
│   └── ampjepa/          # Core Python package
├── scripts/              # Training and evaluation scripts
├── data/                 # Input datasets
├── checkpoints/          # Saved models
├── results/              # Outputs, figures, generated peptides
├── notebooks/            # Exploratory notebooks
├── README.md
└── requirements.txt
```

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/amp-jepa.git
cd amp-jepa
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Train the Model

```bash
python scripts/train.py
```

### Generate and Evaluate AMP Candidates

```bash
python scripts/compare_with_benchmarks.py
```

---

## Model Overview

AMP-JEPA combines three complementary components:

1. **ESM2 Encoder**
   - Produces contextual protein embeddings.

2. **JEPA Latent Predictor**
   - Learns predictive latent representations without reconstruction of raw sequences.

3. **Variational Autoencoder (VAE)**
   - Enables latent space sampling and novel peptide generation.

Generated peptides are subsequently evaluated using external antimicrobial activity predictors such as **APEX**.

---

## Current Status

- [x] APD dataset loading
- [x] ESM2 embedding pipeline
- [x] JEPA latent prediction architecture
- [x] VAE latent model
- [ ] Sequence decoder optimization
- [ ] APEX ensemble integration
- [ ] Bayesian optimization
- [ ] Large-scale peptide generation
- [ ] Diversity and novelty evaluation

---

## Planned Evaluation

Generated peptides will be evaluated using:

- Antimicrobial activity prediction
- Novelty against training data
- Sequence diversity
- Physicochemical properties
- Latent space coverage
- Broad-spectrum prediction
- Toxicity prediction
- Hemolysis prediction

---

## Roadmap

### Phase 1

- Complete JEPA training
- Train decoder
- Validate latent representations

### Phase 2

- Generate novel peptide candidates
- Integrate APEX scoring
- Bayesian optimization in latent space

### Phase 3

- Compare against existing AMP generation methods
- Comprehensive benchmarking
- External validation

---

## Citation

This repository is under active development.

If you use AMP-JEPA in your research, please cite the accompanying publication once available.

---

## License

MIT License

---

## Contact

AMP-JEPA is an ongoing research project exploring Joint Embedding Predictive Architectures (JEPA) for antimicrobial peptide discovery using self-supervised representation.
