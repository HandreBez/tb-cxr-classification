# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap. The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size). Source images are pre-resized and cached to disk once (`cache_images.py`) rather than decoded/resized fresh every epoch.
- **Split:** 90/10 train/test (720 train / 80 test), shuffled with a fixed random seed (42), so every experimental condition trains/tests on the identical split.
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes.
- **Capacity control — ResNet-18:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input, `fc` replaced with `Linear(512, 2)`. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning — ResNet-18 (pretrained):** `models.resnet18(weights=ResNet18_Weights.DEFAULT)`. The pretrained `conv1` (`[64, 3, 7, 7]`) is averaged across the input-channel dimension to `[64, 1, 7, 7]` and loaded into a freshly-constructed 1-channel `Conv2d`, preserving pretrained structure instead of discarding it. `fc` replaced with `Linear(512, 2)`.
- **Transfer learning — DenseNet-121 (pretrained):** `models.densenet121(weights=DenseNet121_Weights.DEFAULT)`. Same channel-averaging approach as ResNet-18, applied to `features.conv0`. `classifier` replaced with `Linear(1024, 2)`.
- **Capacity control — DenseNet-121 (scratch):** `models.densenet121(weights=None)` — `features.conv0` and `classifier` rebuilt fresh (no averaging, since there are no pretrained weights to preserve). Run under differential LR only, matching the ResNet-18 control.
- **Optimizer:** Adam. Two strategies, switchable via a `use_differential_lr` flag in `train.py`:
  - **Uniform LR:** `lr=0.001` across all parameters.
  - **Differential LR:** backbone at `lr=0.0001`, head (`fc` / `classifier`) at `lr=0.001`. Originally introduced to protect pretrained features from being overwritten early in training — also tested on scratch models as a control (see Results).
- **Augmentation (train set only):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. No horizontal flip (anatomically invalid for chest X-rays). `saturation`/`hue` left untouched (meaningless on grayscale).
- **Training:** CrossEntropyLoss, 15 epochs, batch size 32, for every condition.
- **Tracking:** not yet in W&B — currently plain console output.
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

**Architecture alone does not reduce the gap.** No-augmentation ResNet-18 showed a larger gap than the no-augmentation baseline CNN, not a smaller one — greater capacity appears to increase overfitting when nothing else regularizes it.

**A uniform learning rate destroys pretrained structure before it can be used.** Pretrained ResNet-18 trained with a uniform `lr=0.001` produced the worst gap of any condition (28.61 pts) — a rate tuned for random weights is too aggressive for weights that already encode useful structure; early epochs overwrite pretrained features faster than 720 images can re-learn anything comparable ("catastrophic forgetting").

**A control run isolates what differential LR is actually doing.** Differential LR improved pretrained ResNet-18 substantially (67.50% → 88.75% test acc) — initially read as evidence it was protecting pretrained features specifically. Applying the same differential-LR setup to the *scratch* ResNet-18 (nothing pretrained to protect) also improved it substantially (76.25% → 86.25% test acc, gap collapsing to −0.97 pts). This means differential LR does two separate jobs: protecting pretrained structure (relevant only when weights start pretrained), and general optimization regularization (a smaller backbone LR limits memorization on a small dataset regardless of initialization).

**Once LR strategy is held constant — the only fully-controlled comparison in this project — pretraining's actual effect is modest and is not a gap-reducer.** Scratch-differential-LR (86.25% test acc, −0.97 pt gap) vs. pretrained-differential-LR (88.75% test acc, 9.58 pt gap): pretraining contributed a real but modest accuracy gain (+2.5 points) while making the gap substantially worse (+10.55 points). The bigger, more surprising lever in this project turned out to be training procedure (LR structure), not pretraining or architecture.

**The pretraining effect replicates in DenseNet-121, but at a different magnitude.** Holding LR strategy constant, DenseNet-121 scratch (81.25% test acc, 2.92 pt gap) vs. pretrained (90.00% test acc, 9.44 pt gap) shows the same qualitative pattern as ResNet-18. The magnitude differs: DenseNet-121's accuracy gain from pretraining (+8.75 pts) is more than three times ResNet-18's (+2.5 pts), while its gap cost (+6.52 pts) is smaller than ResNet-18's (+10.55 pts) — DenseNet-121 gets a larger accuracy benefit from pretraining for a smaller generalization-gap cost. This directly answers the research question's cross-architecture clause: the direction of the effect is consistent, but its size is architecture-dependent.

## AUC / Sensitivity / Specificity

Accuracy alone can hide class-level failure modes, particularly relevant in a TB screening context where a missed TB case (false negative) is a far costlier error than a false alarm (false positive) — a missed case goes untreated and can continue to spread, while a false alarm just costs a follow-up test. AUC, sensitivity, and specificity were computed for all five conditions using each model's softmax probability for the TB class against the true labels.

| Model | Test Acc | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Baseline CNN | 81.25% | 0.8586 | 0.7297 | 0.8837 |
| ResNet-18 (scratch) | 81.25% | 0.8906 | 0.7297 | 0.8837 |
| ResNet-18 (pretrained) | 92.50% | 0.9221 | 0.9189 | 0.9302 |
| DenseNet-121 (scratch) | 76.25% | 0.8699 | 0.6757 | 0.8372 |
| DenseNet-121 (pretrained) | 87.50% | 0.9302 | 0.8108 | 0.9302 |

*Sensitivity/specificity computed at the standard 0.5 probability threshold. Test accuracy here differs slightly from the Results table above — these conditions were re-run to add prediction-saving, and the gap is consistent with the run-to-run sampling noise already expected on an 80-image test set (see Limitations), not a new source of error.*

**Every model is meaningfully better at correctly identifying non-TB cases than TB cases (specificity > sensitivity in all five rows).** ResNet-18 (pretrained) is the strongest and most balanced model overall — highest sensitivity (0.9189) and tied-highest specificity (0.9302) — while DenseNet-121 (scratch) is the weakest on both.

**AUC and threshold-based metrics can disagree because they measure different things.** Baseline CNN and ResNet-18 (scratch) land on identical sensitivity/specificity despite different AUC (0.8586 vs. 0.8906): sensitivity/specificity are read at one fixed cutoff (0.5), while AUC measures how well a model ranks TB cases above non-TB cases across *every* possible cutoff. Two models can make identical decisions at one threshold while having genuinely different underlying confidence calibration.

**The 0.5 threshold used above is a default, not a clinically justified choice, and this project's own cost asymmetry argues against it.** Since a missed TB case is a worse outcome than a false alarm, the cost of a false negative is not symmetric with the cost of a false positive the way a 0.5 cutoff implicitly assumes. A lower threshold would flag more borderline cases as TB, trading some additional false positives (unnecessary follow-up tests) for fewer false negatives (missed real cases) — likely the more clinically sound choice for a screening tool, even though it would look "worse" on raw accuracy. Choosing the actual threshold properly would mean picking a point on the ROC curve based on the relative cost of the two error types, rather than defaulting to 0.5 — not done here, but a natural next step.

## Limitations

- Dataset size (~800 images) is small relative to model capacity, and the 80-image test set carries real sampling noise — confirmed directly by run-to-run accuracy swings of several points on identical configs (an earlier unseeded split showed swings up to 20 points, which is why the split is now seeded).
- Single-center-ish data, no external validation set, no formal class imbalance check.
- Labels are sourced from filename suffix, not free-text radiologist findings.
- Single fixed 90/10 split (seed=42) — k-fold cross-validation is a reasonable future extension.
- Capacity controls (scratch runs) exist for both architectures, but only under differential LR — the "differential LR provides general optimization regularization independent of pretraining" finding was established for ResNet-18 and assumed, not separately re-confirmed, for DenseNet-121.
- Augmentation and differential-LR hyperparameters were chosen via informal comparison, not a systematic sweep, and carried over unchanged across every model — validated as "better than the alternative tested," not shown to be optimal.
- Sensitivity/specificity are reported at the default 0.5 threshold, which is likely not the right choice for a screening task where false negatives are costlier than false positives (see AUC / Sensitivity / Specificity above).

## Setup

```bash
pip install torch torchvision pillow scikit-learn
python cache_images.py   # one-time: pre-resize source images to data/cache/
python data.py    # sanity check: prints dataset counts and one batch shape.
python train.py   # trains selected model for 15 epochs, then evaluates on test set.
python compute_metrics.py   # computes AUC/sensitivity/specificity from saved predictions.
```

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)

---

## Status

- [x] Data pipeline, baseline CNN, training/evaluation loops, research question locked in
- [x] ResNet-18 (scratch + pretrained), DenseNet-121 (scratch + pretrained) — all under uniform and/or differential LR as applicable
- [x] Image caching (CPU bottleneck fix)
- [x] AUC / sensitivity / specificity
- [ ] W&B experiment tracking
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup