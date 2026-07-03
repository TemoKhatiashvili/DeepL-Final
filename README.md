<p align="center">
  <strong>DeepL-Final</strong><br>
  Attention-based image captioning on Flickr8k
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter"></a>
</p>

---

A university deep-learning project that turns photographs into natural-language captions. The model pairs a **ResNet-50** visual encoder with **Bahdanau attention** and an **LSTM decoder**, trained end-to-end on the Flickr8k dataset.

## How it works

```
  Image  →  ResNet-50  →  Spatial features (7×7)
                              ↓
                         Attention weights
                              ↓
  Caption  ←  LSTM decoder  ←  Context vector
```

At each decoding step, the model learns *where to look* in the image before predicting the next word — a classic encoder–decoder design for vision-and-language tasks.

| | |
|---|---|
| **Encoder** | ResNet-50, ImageNet-pretrained, 2048-d spatial maps |
| **Attention** | Bahdanau soft attention over 49 regions |
| **Decoder** | LSTM with 256-d embeddings, 512-d hidden state |
| **Dataset** | Flickr8k — 8,091 images, ~40,000 captions |
| **Vocabulary** | 2,658 tokens (min. frequency 5) |

## Repository layout

```
DeepL-Final/
├── data_and_training.ipynb   # Dataset prep, training, evaluation, export
├── inference.ipynb           # Caption generation (Colab-ready)
├── requirements.txt
├── Data/caption_data/        # captions.txt + Images/  (not bundled)
└── model/                    # Exported weights & metadata
    ├── best_model.pt
    ├── config.json
    ├── vocabulary.json
    └── data_splits.json
```

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/TemoKhatiashvili/DeepL-Final.git
cd DeepL-Final

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add the dataset**

Download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it under:

```
Data/caption_data/
├── captions.txt
└── Images/
```

Alternatively, drop `caption_data.zip` into `Data/` — the training notebook will extract it automatically.

**3. Train or run inference**

| Goal | Command / file |
|------|----------------|
| Full training | Open `data_and_training.ipynb` (GPU recommended) |
| Quick inference | `python run_inference.py` |
| Colab demo | Open `inference.ipynb` |

## Training

`data_and_training.ipynb` handles the full pipeline:

- Image-level split — 80% train · 10% validation · 10% test  
- Vocabulary construction with special tokens (`<start>`, `<end>`, `<pad>`, `<unk>`)  
- Staged encoder fine-tuning (frozen → partial unfreeze)  
- Mixed-precision training with label smoothing and early stopping  
- Test-set evaluation — loss, perplexity, BLEU-4, qualitative samples  
- Checkpoint export to `model/`

<details>
<summary><strong>Hyperparameters</strong></summary>

| Parameter | Value |
|-----------|-------|
| Epochs | 30 (early stopping, patience 6) |
| Batch size | 32 GPU · 16 CPU |
| Embedding / hidden dim | 256 / 512 |
| Encoder / decoder LR | 1×10⁻⁵ / 3×10⁻⁴ |
| Max caption length | 40 tokens |
| Image size | 224 × 224 |

</details>

## Sample outputs

Greedy decoding on held-out test images:

| | |
|---|---|
| **Image** | `2256218522_53b92bcbb2.jpg` |
| **Generated** | *a black dog is running on the grass* |
| **Reference** | *A black and white dog is running through the sand at a beach.* |

| | |
|---|---|
| **Image** | `256283122_a4ef4a17cb.jpg` |
| **Generated** | *a man in a red dog is running on the grass* |
| **Reference** | *A dog leaps over a man in a white cap to catch an orange Frisbee.* |

The model captures common visual patterns — subjects, colours, actions — while struggling with fine-grained details and rare objects. Full GPU training yields substantially better captions than a short CPU fine-tune.

## Architecture

**Encoder.** ResNet-50 without the final pooling layer, producing a 7×7 grid of 2048-dimensional feature vectors.

**Attention.** A learned alignment between decoder state and spatial features produces a context vector at every timestep:

```
αₜ = softmax(score(hₜ, featureᵢ))     contextₜ = Σ αᵢ · featureᵢ
```

**Decoder.** Word embeddings are concatenated with the context vector and passed through an LSTM cell. The hidden state maps to vocabulary logits; inference selects the highest-probability token until `<end>` or the length limit.

## Evaluation

The training notebook reports:

- **Cross-entropy loss** and **perplexity** on train / val / test  
- **BLEU-4** (corpus-level, smoothed)  
- **Word overlap** against multiple reference captions per image  
- Side-by-side visual grids of successes and failures  

## Acknowledgements

- Flickr8k — Hodosh, Peter, and Matthieu, *Framing Image Description as a Ranking Task* (2013)  
- ResNet backbone — torchvision ImageNet weights  
- Attention mechanism — Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2015)

---

<p align="center">
  <sub>Deep Learning · Computer Vision · Natural Language Generation</sub>
</p>
