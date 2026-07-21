# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap (93.06% train vs. 81.25% test accuracy). The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

*Fallback, if time is tight:* drop the "independent of model capacity" clause (i.e. skip the from-scratch ResNet-18 control) and answer the simpler question — does pretraining reduce the generalization gap, full stop. Methodologically weaker but still a legitimate question, and cheap to upgrade later if a spare training run becomes available.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of next week's transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test) — favored over 80/20 given the small dataset size
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Capacity control:** ResNet-18, randomly initialized (no pretrained weights) — isolates the effect of model capacity/architecture from the effect of pretraining. *Not yet started.*
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned for next week*
- **Augmentation:** *not yet applied to the baseline — planned as a next step, see Limitations*
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

15 epochs each, same 90/10 split, single run (see Limitations re: split stability):

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN (scratch) | 93.06% | 0.167 | 81.25% | 0.469 | 11.81 pts |
| ResNet-18 (scratch, capacity control) | | | | | |
| ResNet-18 (pretrained) | | | | | |
| DenseNet-121 (pretrained) | | | | | |

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Honest read:** the baseline shows a clear ~12-point generalization gap (93.06% train vs. 81.25% test accuracy), and train loss (0.167) is roughly a third of test loss (0.469) — a real overfitting signature, not unexpected given ~800 total images against a model with ~25k parameters in the final layer alone (worked out by hand before writing the architecture). The research question above is built directly around this gap: the ResNet-18 (scratch) row exists specifically to check whether a bigger/deeper architecture alone narrows this gap, so that any further narrowing seen in the pretrained rows can be attributed to the pretrained weights rather than just model capacity.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No data augmentation yet on the baseline (RandomCrop/Flip planned, same pattern as the CIFAR-10 exercise)
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so the 81.25% test accuracy has real sampling noise; it will vary somewhat between runs
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single 90/10 train/test split. The generalization gap reported (e.g. the baseline's 11.81 points) can be measured and reported, but with n≈800 and one split, its *stability* can't be strongly claimed — a different random split could plausibly move it a few points in either direction. This is a real scope limitation of the current setup, not a flaw in the research question itself; a more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows

## Setup

```bash
pip install torch torchvision pillow
python data.py    # sanity check: prints dataset counts and one batch shape.
python train.py   # trains baseline CNN for 15 epochs, then evaluates on test set.
```

## Status

- [x] Data pipeline (loading, labeling, transforms, train/test split, dataloaders)
- [x] Baseline CNN architecture (from scratch)
- [x] Training loop (with per-epoch loss/accuracy)
- [x] Evaluation loop (test-set loss/accuracy)
- [x] First baseline result recorded (93.06% train / 81.25% test accuracy)
- [x] Research question formally locked in
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