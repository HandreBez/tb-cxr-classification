# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

*[Fill in once framed — e.g. "Does ImageNet transfer learning improve TB detection over a from-scratch CNN baseline, given a small labeled dataset (~800 images total)?"]*

## Method

- **Data:** Montgomery County + Shenzhen TB chest X-ray datasets (NLM/NIH)
- **Baseline:** CNN trained from scratch
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights)
- **Augmentation:** *[e.g. RandomCrop, HorizontalFlip, ColorJitter — fill in once finalized]*
- **Tracking:** All experiments logged in Weights & Biases
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence

## Results

*[Results table — baseline vs. fine-tuned models, key metrics: AUC, sensitivity, specificity]*

| Model | Accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline CNN | | | | |
| ResNet-18 (fine-tuned) | | | | |
| DenseNet-121 (fine-tuned) | | | | |

## Limitations

*[dataset size (~800 images total across both sets), single-center data, no external validation set, class imbalance if present, etc.]*

## Setup

```bash
# instructions to be added once code exists
```

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)
