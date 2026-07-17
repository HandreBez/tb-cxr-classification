# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

*[Fill in once framed — e.g. "Does ImageNet transfer learning improve TB detection over a from-scratch CNN baseline, given a small labeled dataset (~800 images total)?"]*

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of next week's transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test) — favored over 80/20 given the small dataset size
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned for next week*
- **Augmentation:** *not yet applied to the baseline — planned as a next step, see Limitations*
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

Baseline CNN, trained from scratch, 15 epochs:

| Model | Train Accuracy | Train Loss | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Baseline CNN (scratch) | 93.06% | 0.167 | 81.25% | 0.469 |
| ResNet-18 (fine-tuned) | | | | |
| DenseNet-121 (fine-tuned) | | | | |

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Honest read:** there's a clear gap between train (93%) and test (81%) accuracy, and train loss (0.167) is roughly a third of test loss (0.469) — a real overfitting signature, not unexpected given ~800 total images against a model with ~25k parameters in the final layer alone (worked out by hand before writing the architecture). This is exactly the kind of gap transfer learning next week is expected to help close, since pretrained ImageNet features should generalize better than features learned from scratch on this little data.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No data augmentation yet on the baseline (RandomCrop/Flip planned, same pattern as the CIFAR-10 exercise)
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so the 81.25% test accuracy has real sampling noise; it will vary somewhat between runs
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases

## Setup

```bash
pip install torch torchvision pillow
python data.py    # sanity check: prints dataset counts and one batch shape
python train.py   # trains baseline CNN for 15 epochs, then evaluates on test set
```

## Status

- [x] Data pipeline (loading, labeling, transforms, train/test split, dataloaders)
- [x] Baseline CNN architecture (from scratch)
- [x] Training loop (with per-epoch loss/accuracy)
- [x] Evaluation loop (test-set loss/accuracy)
- [x] First baseline result recorded (93.06% train / 81.25% test accuracy)
- [ ] Research question formally locked in
- [ ] Data augmentation added to baseline
- [ ] Transfer learning: ResNet-18
- [ ] Transfer learning: DenseNet-121
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)