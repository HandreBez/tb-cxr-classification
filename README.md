# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap. The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test), shuffled with a **fixed random seed (42)** before splitting, so every experimental condition trains/tests on the identical split — critical for fair comparison across conditions (see Known bugs)
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Capacity control — ResNet-18:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input, `fc` replaced with `Linear(512, 2)`. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning — ResNet-18 (pretrained):** `models.resnet18(weights=ResNet18_Weights.DEFAULT)`. The pretrained `conv1` (`[64, 3, 7, 7]`) is averaged across the input-channel dimension (`dim=1`, `keepdim=True`) to `[64, 1, 7, 7]` and loaded into a freshly-constructed 1-channel `Conv2d`, preserving pretrained structure instead of discarding it. `fc` replaced with `Linear(512, 2)`.
- **Transfer learning — DenseNet-121 (pretrained):** `models.densenet121(weights=DenseNet121_Weights.DEFAULT)`. Same channel-averaging approach as ResNet-18: the pretrained `features.conv0` (`[64, 3, 7, 7]`) is averaged across the input-channel dimension to `[64, 1, 7, 7]` and loaded into a freshly-constructed 1-channel `Conv2d`. `classifier` (DenseNet's equivalent of ResNet's `fc`) replaced with `Linear(1024, 2)`, matching `features.norm5`'s 1024-channel output.
- **Capacity control — DenseNet-121 (scratch):** `models.densenet121(weights=None)` — `features.conv0` built directly for 1-channel input (no averaging, since there are no pretrained weights to preserve), `classifier` replaced with `Linear(1024, 2)`. Run under differential LR only, to give a matched-LR pretrained-vs-scratch comparison for this architecture, mirroring the ResNet-18 control run above.
- **Optimizer:** Adam. Two strategies, switchable via a `use_differential_lr` flag in `train.py`, independent of which model is active:
  - **Uniform LR:** `lr=0.001` across all parameters.
  - **Differential LR:** parameters split via `model.named_parameters()` into backbone (everything except the head) and head (`fc` for ResNet-18, `classifier` for DenseNet-121) — backbone at `lr=0.0001`, head at `lr=0.001`. Originally introduced to protect pretrained ResNet-18's features from being overwritten early in training — subsequently also tested on the scratch ResNet-18 as a control, which revealed differential LR has a second effect independent of pretraining (see Results).
- **Augmentation (train set only):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. No horizontal flip (anatomically invalid for chest X-rays). `saturation`/`hue` left untouched (meaningless on grayscale). `augment=True` locked as default across all conditions.
- **Training:** CrossEntropyLoss, 15 epochs, batch size 32, for every condition.
- **Tracking:** not yet in W&B — currently plain console output, planned.
- **Deployment:** FastAPI endpoint not yet built.

## Results

All results below use the fixed seeded split (seed=42), `augment=True`, 15 epochs, batch size 32 — directly comparable to each other.

| Model | LR strategy | Train Acc | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN | uniform | 77.64% | 68.75% | 0.532 | 8.89 pts |
| ResNet-18 (scratch) | uniform | 80.00% | 76.25% | 0.476 | 3.75 pts |
| ResNet-18 (scratch) | **differential** | **85.28%** | **86.25%** | **0.344** | **−0.97 pts** |
| ResNet-18 (pretrained) | uniform | 96.11% | 67.50% | 1.162 | 28.61 pts |
| ResNet-18 (pretrained) | differential | 98.33% | 88.75% | 0.501 | 9.58 pts |
| DenseNet-121 (scratch) | differential | 84.17% | 81.25% | 0.4331 | 2.92 pts |
| DenseNet-121 (pretrained) | differential | 99.44% | 90.00% | 0.3458 | 9.44 pts |

*No-augmentation baseline/ResNet-18-scratch rows omitted for brevity — see git history; both showed larger gaps than their augmented counterparts (15.42 pts and 18.47 pts respectively).*

*AUC, sensitivity, specificity not yet computed.*

**Architecture alone does not reduce the gap.** No-augmentation ResNet-18 showed a larger gap than the no-augmentation baseline CNN, not a smaller one — greater capacity appears to increase overfitting when nothing else regularizes it.

**Augmentation's effect differs by architecture.** For the baseline CNN, augmentation shrinks the gap but costs real test accuracy. For scratch ResNet-18 (uniform LR), augmentation's effect is larger and cleaner.

**A uniform learning rate destroys pretrained structure before it can be used.** The first pretrained ResNet-18 run, trained with `lr=0.001` uniformly, produced the worst gap of any condition tested (28.61 pts). Train accuracy climbed past 90% within 8 epochs while test loss ended at 1.162, the worst of any condition — a rate tuned for random weights is too aggressive for weights that already encode useful structure; early epochs overwrite pretrained features faster than 720 training images can re-learn anything comparable ("catastrophic forgetting").

**Control run — differential LR applied to scratch ResNet-18 — isolates what differential LR is actually doing.** Switching pretrained ResNet-18 to differential LR improved it substantially (67.50% → 88.75% test acc), which initially looked like clear evidence differential LR was protecting pretrained features specifically. Applying the same differential-LR setup to the *scratch* ResNet-18 (nothing pretrained to protect) also improved it substantially — test accuracy rose from 76.25% to 86.25%, gap collapsed from 3.75 pts to essentially zero (−0.97 pts).

This means differential LR does two separate jobs:
1. **Protecting pretrained structure** — relevant only when weights start pretrained.
2. **General optimization regularization** — a smaller backbone LR slows how fast the model fits the training set at all, limiting memorization on a dataset this small regardless of initialization.

**Once LR strategy is held constant — the only fully-controlled comparison in this project — pretraining's actual effect is modest and is not a gap-reducer.** Scratch-differential-LR (86.25% test acc, −0.97 pt gap) vs. pretrained-differential-LR (88.75% test acc, 9.58 pt gap): pretraining contributed a real but modest accuracy gain (+2.5 points) while making the gap substantially worse (+10.55 points). **The bigger, more surprising lever in this project turned out to be training procedure (LR structure), not pretraining or architecture** — revising the framing built from the earlier uniform-LR-only results.

**The pretraining effect replicates in DenseNet-121, but at a different magnitude.** Holding LR strategy constant (differential, matching the ResNet-18 control), DenseNet-121 scratch (81.25% test acc, 2.92 pt gap) vs. DenseNet-121 pretrained (90.00% test acc, 9.44 pt gap) shows the same qualitative pattern as ResNet-18: pretraining improves accuracy while worsening the gap. The magnitude differs meaningfully, however. DenseNet-121's accuracy gain from pretraining (+8.75 pts) is more than three times ResNet-18's (+2.5 pts), while its gap cost (+6.52 pts) is smaller than ResNet-18's (+10.55 pts). In other words, DenseNet-121 gets a larger accuracy benefit from pretraining for a smaller generalization-gap cost than ResNet-18 does — the trade-off is directionally the same across both architectures, but quantitatively more favorable for DenseNet-121. This directly answers the "does this effect differ between ResNet-18 and DenseNet-121" clause of the research question: yes, the direction of the effect is consistent, but its size is architecture-dependent.

## Known bugs already hit and fixed

- **Unseeded random split (significant, fixed):** produced accuracy swings up to 20 points between otherwise-identical runs on the 80-image test set. Fixed via `random.seed(42)` before `random.shuffle(all_images)`.
- `matplotlib.transforms` import shadowing `torchvision.transforms` — fixed.
- `DataLoader` missing import — fixed.
- Training loop referencing `train_loader` before creation — fixed.
- `num_workers` slowdown — bumped to 6, `pin_memory=True` added to both `DataLoader`s. GPU utilization still caps well below 100% during training — root cause is full-resolution source images (Montgomery ~4892×4020px) being decoded/resized fresh every epoch on CPU. Proper fix (pre-resize/cache to disk) identified, not yet implemented.
- Single `get_transforms()` applied before split — fixed via pre-split-then-two-datasets approach.
- **Uniform learning rate on pretrained ResNet-18 (significant, fixed):** produced the worst gap of any condition (28.61 pts). Fixed via differential learning rates through optimizer parameter groups (backbone `lr=0.0001`, head `lr=0.001`, split via `model.named_parameters()` filtering on the head's attribute name — `"resnet.fc"` for ResNet-18, `"densenet.classifier"` for DenseNet-121). LR strategy is now a standalone `use_differential_lr` flag in `train.py`, decoupled from which model is active.

## Limitations

- Dataset size (~800 images) small relative to model capacity — visible overfitting in most conditions
- Single-center-ish data, no external validation set, no class imbalance check yet
- Test set is small (80 images, ~1.25 pts/image) — real sampling noise
- Labels from filename suffix, not free-text radiologist findings
- Single fixed 90/10 split (seed=42) — k-fold cross-validation is a reasonable future extension
- Capacity controls (scratch runs) exist for both architectures, but each only under differential LR — neither was tested under uniform LR, so the "differential LR provides general optimization regularization independent of pretraining" finding (established for ResNet-18) was assumed to hold for DenseNet-121 rather than separately re-confirmed there
- Augmentation hyperparameters chosen via one informal A/B comparison on an unseeded split, not a systematic sweep
- Earlier pretrained-vs-scratch comparisons (both uniform-LR and differential-LR) each changed two things at once; resolved via the scratch-differential-LR control runs above — the *original* pretrained-vs-scratch numbers reported in earlier project notes should not be read as isolating pretraining's effect on their own
- The `0.0001`/`0.001` differential LR split was carried over unchanged to every scratch/architecture combination — never separately tuned per model, only ever validated as "better than uniform," not shown to be optimal

## Setup

```bash
pip install torch torchvision pillow
python data.py    # sanity check: prints dataset counts and one batch shape.
python train.py   # trains selected model for 15 epochs, then evaluates on test set.
```

## Status

- [x] Data pipeline (loading, labeling, transforms, train/test split, dataloaders)
- [x] Baseline CNN architecture (from scratch)
- [x] Training loop, evaluation loop
- [x] Research question formally locked in
- [x] Reproducible train/test split (fixed seed), augmentation toggle
- [x] Baseline CNN: no-augmentation and augmented results recorded
- [x] ResNet-18 (scratch, capacity control): uniform-LR and differential-LR results recorded
- [x] Transfer learning: ResNet-18 (pretrained) — conv1 averaging, uniform-LR and differential-LR results recorded
- [x] Control run: scratch ResNet-18 with differential-LR parameter grouping
- [x] Transfer learning: DenseNet-121 (pretrained) — conv0 averaging, differential-LR results recorded
- [x] Control run: scratch DenseNet-121 with differential-LR parameter grouping
- [ ] Housekeeping: pre-resize/cache source images to reduce CPU bottleneck
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)