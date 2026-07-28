# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap (93.06% train vs. 81.25% test accuracy). The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test), shuffled before splitting so both source datasets (Shenzhen, Montgomery) are represented in both splits — favored over 80/20 given the small dataset size
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Capacity control:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input (no averaging needed, since there are no pretrained weights to preserve in this condition), `fc` replaced with `Linear(512, 2)` for the 2-class task. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned next*
- **Augmentation (train set only):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. Design choices, reasoned through explicitly rather than copied from the CIFAR-10 exercise:
  - **No horizontal flip** — chest anatomy is not left-right symmetric (heart position, lobe structure) and TB findings are not symmetric either, so a flipped image would not represent a physically valid chest X-ray. Standard practice in the chest X-ray literature avoids this augmentation for the same reason.
  - **Mild `RandomCrop`, not aggressive lung-isolation cropping** — padding of 12px (vs. a naive proportional scale-up from CIFAR-10's `padding=4` on 32px images, which would suggest ~28px) was deliberately scaled back, since aggressive cropping risks cutting into diagnostic lung regions and chest X-rays are centrally framed and consistent across patients, unlike natural photos where CIFAR-10-style jitter originated.
  - **`ColorJitter` restricted to `brightness`/`contrast` only** — `saturation` and `hue` are color concepts with no meaning on a single-channel grayscale image and were left untouched.
  - **`brightness=0.1, contrast=0.1`, not `0.2/0.2`** — an initial run at 0.2/0.2 was tested and compared directly (see Augmentation strength comparison below); 0.1/0.1 was chosen as the setting carried forward into all remaining experiments.
  - **Train transform only** — the test set uses a clean `Resize`/`ToTensor`/`Normalize` pipeline with no augmentation, so evaluation always reflects performance on realistic, unmodified data.
  - Applied to the baseline CNN this week; the same fixed augmentation config will be used consistently across the ResNet-18 and DenseNet-121 runs so that gap differences between conditions are attributable to architecture/pretraining, not to augmentation changing between runs.
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

15 epochs each, same 90/10 split, single run (see Limitations re: split stability):

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN (scratch, no augmentation) | 93.06% | 0.167 | 81.25% | 0.469 | 11.81 pts |
| Baseline CNN (scratch, augmented) | 82.64% | 0.428 | 78.75% | 0.590 | 3.89 pts |
| ResNet-18 (scratch, capacity control) | 91.25% | 0.205 | 86.25% | 0.377 | 5.00 pts |
| ResNet-18 (pretrained) | | | | | |
| DenseNet-121 (pretrained) | | | | | |

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Honest read (no augmentation):** the baseline shows a clear ~12-point generalization gap (93.06% train vs. 81.25% test accuracy), and train loss (0.167) is roughly a third of test loss (0.469) — a real overfitting signature, not unexpected given ~800 total images against a model with ~25k parameters in the final layer alone (worked out by hand before writing the architecture).

**Augmentation strength comparison:** two `ColorJitter` strengths were tested on the augmented baseline before locking one in — `brightness=0.2, contrast=0.2` and `brightness=0.1, contrast=0.1` (`RandomCrop(224, padding=12)` held constant across both, to isolate the effect of intensity jitter specifically, since TB findings are picked out largely by density/contrast differences on the film).

| Config | Train Acc | Test Acc | Test Loss | Gap |
|---|---|---|---|---|
| ColorJitter 0.2 / 0.2 | 78.89% | 78.75% | 0.4596 | 0.14 pts |
| ColorJitter 0.1 / 0.1 | 82.64% | 78.75% | 0.5896 | 3.89 pts |

Test accuracy was identical between the two settings. The stronger setting (0.2/0.2) nearly eliminates the gap, but train accuracy is also lower and test loss is worse — a plausible sign of mild underfitting, where the augmentation is aggressive enough relative to the ~720-image training set that the model struggles to fit the training data well, rather than genuinely learning a more generalizable representation. **0.1/0.1 was chosen** as the setting to carry forward: it still cuts the no-augmentation gap by roughly two-thirds (11.81 → 3.89 pts) while keeping train accuracy and test loss more favorable, avoiding the underfitting signature seen at 0.2/0.2. This is a judgment call, not a uniquely correct answer — the tradeoff between gap-minimization and fit quality is noted here explicitly rather than picking the smallest-gap number in isolation.

**Augmented vs. unaugmented baseline:** augmentation (at the chosen 0.1/0.1 setting) reduced the generalization gap substantially (11.81 → 3.89 pts) at the cost of a small drop in test accuracy (81.25% → 78.75%). Whether that tradeoff is worthwhile depends on what the project prioritizes — a smaller, more honest gap vs. the single highest test accuracy number — and is discussed further as an open question for the final writeup.

**Capacity control:** the from-scratch ResNet-18 (no augmentation applied yet) roughly halves the *unaugmented* baseline's gap (5.00 pts vs. 11.81 pts), despite having zero pretrained knowledge. This is an important result for the research question: it shows that a meaningful chunk of any future gap reduction from the pretrained ResNet-18/DenseNet-121 runs is *not* automatically attributable to pretraining — architecture alone (likely batch normalization smoothing optimization, and residual connections improving gradient flow) already buys a substantial reduction in overfitting here, independent of ImageNet weights. Note: this ResNet-18 run predates the augmentation decision above and was not trained with augmentation; augmentation will be applied consistently going forward for the remaining ResNet-18 (pretrained) and DenseNet-121 runs, which introduces a minor inconsistency worth flagging (see Limitations).

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so test accuracy has real sampling noise; it will vary somewhat between runs
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single 90/10 train/test split. Reported generalization gaps can be measured and compared, but with n≈800 and one split, their *stability* can't be strongly claimed — a different random split could plausibly move them a few points in either direction. This is a real scope limitation of the current setup, not a flaw in the research question itself; a more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows
- Only one from-scratch capacity control (ResNet-18) was run — DenseNet-121's scratch/pretrained gap contribution isn't separately isolated, so its "independent of capacity" comparison relies partly on the ResNet-18 control's architecture-effect finding generalizing across backbones, which is a real assumption, not confirmed
- The ResNet-18 (scratch) result was trained *before* the augmentation config was finalized, so it does not include augmentation, while the augmented baseline and all remaining planned runs (ResNet-18 pretrained, DenseNet-121) will. This is a real inconsistency in the current results table — a like-for-like ResNet-18 (scratch, augmented) rerun is a reasonable addition if time allows, to keep every row in the final table on equal footing
- Augmentation hyperparameters (`ColorJitter` strength, crop padding) were chosen via a single informal A/B comparison (0.2/0.2 vs 0.1/0.1), not a systematic sweep — the chosen values are defensible but not proven optimal

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
- [x] Data augmentation added to baseline (train-only, `RandomCrop` + `ColorJitter`, strength compared and locked in at 0.1/0.1)
- [ ] Transfer learning: ResNet-18 (pretrained)
- [ ] Transfer learning: DenseNet-121 (pretrained)
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)