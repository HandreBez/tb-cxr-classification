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
- **Capacity control:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input (no averaging needed, since there are no pretrained weights to preserve in this condition), `fc` replaced with `Linear(512, 2)` for the 2-class task. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning — ResNet-18 (pretrained):** `models.resnet18(weights=ResNet18_Weights.DEFAULT)`. Since the pretrained `conv1` expects 3-channel RGB input but this dataset is single-channel grayscale, the original `[64, 3, 7, 7]` conv1 weights are averaged across the input-channel dimension (`dim=1`, `keepdim=True`) to produce `[64, 1, 7, 7]` weights, which are then loaded into a freshly-constructed 1-channel `Conv2d` layer — preserving the pretrained filter structure rather than discarding it (contrast with the scratch condition's `conv1`, which is simply reinitialized). `fc` replaced with `Linear(512, 2)`, same as the scratch condition.
- **Transfer learning — DenseNet-121 (pretrained):** *not yet started, planned next*
- **Optimizer — scratch conditions (baseline CNN, ResNet-18 scratch):** Adam, uniform `lr=0.001` across all parameters.
- **Optimizer — pretrained ResNet-18:** Adam with **differential learning rates** — pretrained backbone parameters (everything except `fc`) at `lr=0.0001`, freshly-initialized `fc` head at `lr=0.001`. This was not the original design — see "Learning rate findings" below for why it became necessary.
- **Augmentation (train set only, toggleable via `get_transforms(augment=True/False)` → `get_dataloaders(..., augment=True/False)`):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. Design choices, reasoned through explicitly rather than copied from the CIFAR-10 exercise:
  - **No horizontal flip** — chest anatomy is not left-right symmetric (heart position, lobe structure) and TB findings are not symmetric either, so a flipped image would not represent a physically valid chest X-ray. Standard practice in the chest X-ray literature avoids this augmentation for the same reason.
  - **Mild `RandomCrop`, not aggressive lung-isolation cropping** — padding of 12px (vs. a naive proportional scale-up from CIFAR-10's `padding=4` on 32px images, which would suggest ~28px) was deliberately scaled back, since aggressive cropping risks cutting into diagnostic lung regions and chest X-rays are centrally framed and consistent across patients, unlike natural photos where CIFAR-10-style jitter originated.
  - **`ColorJitter` restricted to `brightness`/`contrast` only** — `saturation` and `hue` are color concepts with no meaning on a single-channel grayscale image and were left untouched.
  - **`brightness=0.1, contrast=0.1`** — an initial informal comparison against 0.2/0.2 (on an unseeded split, see Known bugs) suggested 0.1/0.1 avoided an underfitting signature seen at the stronger setting; carried forward as the standard config.
  - **Train transform only** — the test set uses a clean `Resize`/`ToTensor`/`Normalize` pipeline with no augmentation, so evaluation always reflects performance on realistic, unmodified data.
  - `augment=True` is the locked default across all experiments in this project, so gap differences between conditions are attributable to architecture/pretraining, not to augmentation changing between runs.
- **Training:** CrossEntropyLoss, 15 epochs, batch size 32, for every condition
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

All results below use the fixed seeded split (seed=42) and are directly comparable to each other. 15 epochs each, batch size 32, `augment=True`.

| Model | Train Acc | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|
| Baseline CNN, augmented | 77.64% | 68.75% | 0.532 | 8.89 pts |
| ResNet-18 (scratch), augmented | 80.00% | 76.25% | 0.476 | 3.75 pts |
| ResNet-18 (pretrained), uniform lr=0.001 | 96.11% | 67.50% | 1.162 | 28.61 pts |
| ResNet-18 (pretrained), differential LR | 98.33% | 88.75% | 0.501 | 9.58 pts |
| DenseNet-121 (pretrained) | | | | |

*No-augmentation baseline/ResNet-18-scratch rows omitted above for brevity — see prior session notes / git history; both showed larger gaps than their augmented counterparts (15.42 pts and 18.47 pts respectively).*

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Architecture alone does not reduce the gap — if anything, it makes it worse.** Comparing the no-augmentation baseline CNN and no-augmentation scratch ResNet-18 (not shown above, see git history): ResNet-18 shows a *larger* gap than the baseline CNN, not a smaller one. ResNet-18's much greater capacity (~11M parameters vs. ~25k in the baseline's final layer alone) appears to make it more prone to overfitting when given nothing else to regularize it, not less.

**Augmentation's effect differs meaningfully by architecture, and isn't uniformly "better."** For the baseline CNN, augmentation shrinks the gap but at a real cost to test accuracy. For ResNet-18 (scratch), augmentation's effect is both larger and cleaner: the gap collapses to 3.75 points while test accuracy only drops modestly.

**A uniform learning rate destroys pretrained structure before it can be used.** The first pretrained ResNet-18 run, trained with the same `lr=0.001` used for every scratch condition, produced the worst gap of any condition tested (28.61 pts) despite starting from ImageNet weights — worse than even the no-augmentation baseline CNN. The training curve shows why: train accuracy climbed past 90% within 8 epochs while test loss ended at 1.162, well above every other condition's test loss. A learning rate tuned for training random weights from scratch is too aggressive for weights that already encode useful structure — early epochs at that rate overwrite pretrained features faster than the small training set (720 images) can re-learn anything comparably useful, a failure mode sometimes called catastrophic forgetting in the fine-tuning literature.

Switching to differential learning rates (backbone layers at `lr=0.0001`, the freshly-initialized `fc` head at `lr=0.001`) changed the result substantially: test accuracy rose to 88.75%, the best of any condition tested, and test loss dropped to 0.501. However, the gap (9.58 pts) is *not* the smallest observed — it's worse than the augmented scratch ResNet-18 control (3.75 pts), despite much higher absolute performance. Train accuracy also rose sharply under this setup (80.00% → 98.33%), suggesting the pretrained features let the model fit the training set almost perfectly, which grows the gap even as generalization (test accuracy) also improves. **This is a genuinely counter-intuitive result relative to the naive hypothesis that pretraining should shrink the gap** — under a properly-tuned learning rate, pretraining on this task improved absolute performance without improving (and arguably worsening) the generalization gap specifically, relative to the capacity-matched scratch control.

**The interesting finding, taken as a whole, is that neither capacity nor pretraining is a straightforward "gap-reducer" on its own — regularization (augmentation) and training procedure (learning rate) both matter as much or more than the headline architectural choice.** This has a direct bearing on how the DenseNet-121 run should be read: without matching training conditions (same augmentation, same differential-LR structure), any DenseNet-121 vs. ResNet-18 comparison would be confounded the same way the early pretrained-ResNet-18 run was.

## Known bugs already hit and fixed

- **Unseeded random split (significant, now fixed):** `get_dataloaders`'s `random.shuffle(all_images)` originally had no fixed seed, so every training run used a genuinely different random 90/10 split. With only 80 test images, this alone produced accuracy swings of up to 20 points between otherwise-identical runs (confirmed directly: rerunning ResNet-18 scratch+augmented on an unseeded split gave 66.25% test accuracy / 15.28pt gap, vs. 76.25% / 3.75pt gap on the seeded split). Fixed by adding `random.seed(42)` immediately before `random.shuffle(all_images)` inside `get_dataloaders`.
- `from matplotlib import transforms` silently shadowed `torchvision.transforms`, causing `AttributeError: module 'matplotlib.transforms' has no attribute 'Compose'`. Fixed by removing the matplotlib import and using `from torchvision import transforms`.
- `DataLoader` used in `data.py` without being imported — `NameError`.
- Training loop originally referenced `train_loader` before it was created, because the loop sat outside/after the `if __name__ == "__main__":` block where `train_loader` was actually built. Fixed by moving the whole training loop inside that block.
- `num_workers` default of 2 was a real slowdown given the huge source image files (Montgomery images are ~4892×4020 px before resizing). Bumped to 4 (out of 8 available CPU cores) via `get_dataloaders(..., num_workers=4)`.
- `data.py`'s original single `get_transforms()` returned one transform applied before `random_split`, making separate train/test augmentation impossible. Fixed by splitting the raw image list first (shuffled, seeded), then building two separate `TBXrayDataset` objects from the two slices, each with its own transform.
- **Uniform learning rate on pretrained ResNet-18 (significant, now fixed):** the first pretrained ResNet-18 run used the same `lr=0.001` as every scratch condition, applied uniformly across all parameters including the pretrained backbone. This produced the worst generalization gap of any condition tested (28.61 pts), consistent with the learning rate being too aggressive for already-structured pretrained weights on a small dataset. Fixed by switching to differential learning rates via PyTorch optimizer parameter groups — backbone parameters (identified via `model.named_parameters()`, filtered on whether the name starts with `"resnet.fc"`) at `lr=0.0001`, `fc` head parameters at `lr=0.001`.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test in every condition tested
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so test accuracy has real sampling noise
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single fixed 90/10 train/test split (seed=42). A more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows
- Only one from-scratch capacity control (ResNet-18) is planned — DenseNet-121's scratch/pretrained gap contribution won't be separately isolated
- Augmentation hyperparameters (`ColorJitter` strength, crop padding) were chosen via a single informal A/B comparison on an unseeded split, not a systematic sweep
- **The pretrained-vs-scratch ResNet-18 comparison is confounded by a training-procedure change, not just a weights change:** the pretrained run uses differential learning rates (0.0001 backbone / 0.001 head), while the scratch run uses a single uniform 0.001 rate for all parameters. It is not yet established how much of the pretrained condition's accuracy/gap difference is attributable to pretraining itself versus to the differential-LR training procedure alone. A direct control — rerunning the scratch ResNet-18 condition with the same differential-LR parameter grouping applied to its (randomly-initialized) backbone and head — is planned to isolate this.
- The DenseNet-121 condition, once run, must use matched training conditions (augmentation, differential LR structure) to the pretrained ResNet-18 run for the architecture comparison in the research question to be valid — otherwise any observed difference is confounded with mismatched training procedure, the same issue the learning-rate finding above surfaced for ResNet-18 alone.

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
- [x] Research question formally locked in
- [x] Reproducible train/test split (fixed seed)
- [x] Augmentation toggle (`augment=True/False`)
- [x] Baseline CNN: no-augmentation and augmented results recorded on seeded split
- [x] ResNet-18 (scratch, capacity control): no-augmentation and augmented results recorded on seeded split
- [x] Transfer learning: ResNet-18 (pretrained) — conv1 channel-averaging implemented, differential learning rate implemented after diagnosing a uniform-LR failure mode
- [ ] Control run: scratch ResNet-18 with differential-LR parameter grouping (isolates LR effect from pretraining effect)
- [ ] Transfer learning: DenseNet-121 (pretrained), matched training conditions to pretrained ResNet-18
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)