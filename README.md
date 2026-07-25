# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap (93.06% train vs. 81.25% test accuracy). The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test) — favored over 80/20 given the small dataset size
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Capacity control:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input (no averaging needed, since there are no pretrained weights to preserve in this condition), `fc` replaced with `Linear(512, 2)` for the 2-class task. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned next*
- **Augmentation:** *not yet applied to the baseline — planned as a next step, see Limitations*
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

15 epochs each, same 90/10 split, single run (see Limitations re: split stability):

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN (scratch) | 93.06% | 0.167 | 81.25% | 0.469 | 11.81 pts |
| ResNet-18 (scratch, capacity control) | 91.25% | 0.205 | 86.25% | 0.377 | 5.00 pts |
| ResNet-18 (pretrained) | | | | | |
| DenseNet-121 (pretrained) | | | | | |

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Honest read:** the baseline shows a clear ~12-point generalization gap (93.06% train vs. 81.25% test accuracy), and train loss (0.167) is roughly a third of test loss (0.469) — a real overfitting signature, not unexpected given ~800 total images against a model with ~25k parameters in the final layer alone (worked out by hand before writing the architecture).

The from-scratch ResNet-18 capacity control roughly **halves** the gap (5.00 pts vs. 11.81 pts), despite having zero pretrained knowledge — same random initialization philosophy as the baseline. This is an important result for the research question: it shows that a meaningful chunk of any future gap reduction from the pretrained ResNet-18/DenseNet-121 runs is *not* automatically attributable to pretraining. Architecture alone (likely batch normalization smoothing optimization, and residual connections improving gradient flow) already buys a substantial reduction in overfitting here, independent of ImageNet weights. Whatever gap reduction the pretrained runs show *beyond* this 5.00 pt figure is the part that can be more confidently attributed to pretraining specifically — which is exactly why this capacity-control row was built into the experiment design rather than skipped.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No data augmentation yet on the baseline (RandomCrop/Flip planned, same pattern as the CIFAR-10 exercise)
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so test accuracy has real sampling noise; it will vary somewhat between runs
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single 90/10 train/test split. Reported generalization gaps can be measured and compared, but with n≈800 and one split, their *stability* can't be strongly claimed — a different random split could plausibly move them a few points in either direction. This is a real scope limitation of the current setup, not a flaw in the research question itself; a more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows
- Only one from-scratch capacity control (ResNet-18) was run — DenseNet-121's scratch/pretrained gap contribution isn't separately isolated, so its "independent of capacity" comparison relies partly on the ResNet-18 control's architecture-effect finding generalizing across backbones, which is a real assumption, not confirmed

## Setup

```bash
pip install torch torchvision pillow
python data.py    # sanity check: prints dataset counts and one batch shape.
python train.py   # trains selected model for 15 epochs, then evaluates on test set.
```

## Status

- [x] Data pipeline (loading, labeling, transforms, train/test split, dataloaders)
- [x] Baseline CNN architecture (from scratch)
- [x] Training loop (with per-epoch loss/accuracy)
- [x] Evaluation loop (test-set loss/accuracy)
- [x] First baseline result recorded (93.06% train / 81.25% test accuracy)
- [x] Research question formally locked in
- [x] ResNet-18 (scratch, capacity control) architecture built and trained
- [X] Data augmentation added to baseline
- [ ] Transfer learning: ResNet-18 (pretrained)
- [ ] Transfer learning: DenseNet-121 (pretrained)
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)