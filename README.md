# TB CXR Classification

Tuberculosis detection from chest X-rays using transfer learning, built as Project 1 of an Honours-track medical computer vision portfolio.

## Question

**Does ImageNet pretraining reduce the train/test generalization gap in TB detection from chest X-rays under a small-data regime (~800 images, Montgomery + Shenzhen), independent of the increase in model capacity — and does this effect differ between ResNet-18 and DenseNet-121?**

The outcome measured is the train/test *generalization gap* (not just raw accuracy) — motivated by the baseline result below, which showed a clear gap (93.06% train vs. 81.25% test accuracy). The "independent of model capacity" clause requires a from-scratch (randomly initialized) ResNet-18 run as a capacity-matched control, so that any improvement from the pretrained ResNet-18 run can be attributed to the pretrained weights specifically, not just to using a bigger/deeper architecture than the baseline CNN.

## Method

- **Data:** Montgomery County (138 images) + Shenzhen (662 images) TB chest X-ray datasets (NLM/NIH) — 800 images total combined
- **Preprocessing:** Grayscale conversion (1 channel), resized to 224×224 (matches ImageNet input size, chosen ahead of transfer learning experiments so the pipeline doesn't need reworking)
- **Split:** 90/10 train/test (720 train / 80 test), shuffled with a **fixed random seed (42)** before splitting, so every experimental condition trains/tests on the identical split — critical for fair comparison across conditions (see Known bugs below; earlier results in this project predate this fix and are not directly comparable to results from here forward)
- **Baseline:** CNN trained from scratch — 3 conv layers (1→8→16→32 channels), 3×3 kernels, padding=1, max pooling after each conv, flatten to 25,088, linear output to 2 classes
- **Capacity control:** ResNet-18, randomly initialized (`weights=None`) — `conv1` built directly for 1-channel input (no averaging needed, since there are no pretrained weights to preserve in this condition), `fc` replaced with `Linear(512, 2)` for the 2-class task. Isolates the effect of model capacity/architecture from the effect of pretraining.
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned next*
- **Augmentation (train set only, toggleable via `get_transforms(augment=True/False)`):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. Design choices, reasoned through explicitly rather than copied from the CIFAR-10 exercise:
  - **No horizontal flip** — chest anatomy is not left-right symmetric (heart position, lobe structure) and TB findings are not symmetric either, so a flipped image would not represent a physically valid chest X-ray. Standard practice in the chest X-ray literature avoids this augmentation for the same reason.
  - **Mild `RandomCrop`, not aggressive lung-isolation cropping** — padding of 12px (vs. a naive proportional scale-up from CIFAR-10's `padding=4` on 32px images, which would suggest ~28px) was deliberately scaled back, since aggressive cropping risks cutting into diagnostic lung regions and chest X-rays are centrally framed and consistent across patients, unlike natural photos where CIFAR-10-style jitter originated.
  - **`ColorJitter` restricted to `brightness`/`contrast` only** — `saturation` and `hue` are color concepts with no meaning on a single-channel grayscale image and were left untouched.
  - **`brightness=0.1, contrast=0.1`, not `0.2/0.2`** — an initial run at 0.2/0.2 was tested and compared directly (see Augmentation strength comparison below); 0.1/0.1 was chosen as the setting carried forward into all remaining experiments.
  - **Train transform only** — the test set uses a clean `Resize`/`ToTensor`/`Normalize` pipeline with no augmentation, so evaluation always reflects performance on realistic, unmodified data.
  - `augment=True` is the default for all remaining experiments this week (ResNet-18 pretrained, DenseNet-121), so gap differences between conditions are attributable to architecture/pretraining, not to augmentation changing between runs. `augment=False` is used deliberately for capacity-control/ablation comparisons (see Results).
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

15 epochs each, batch size 32. Results below are grouped by which train/test split they used — see Known bugs for why this matters.

**Seeded split (seed=42) — directly comparable to each other:**

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| ResNet-18 (scratch, no augmentation) | 97.22% | 0.073 | 78.75% | 0.943 | 18.47 pts |
| ResNet-18 (scratch, augmented) | 80.00% | 0.436 | 76.25% | 0.476 | 3.75 pts |

**Earlier, unseeded runs — each on a *different* random split, not directly comparable to each other or to the seeded results above:**

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN (scratch, no augmentation) | 93.06% | 0.167 | 81.25% | 0.469 | 11.81 pts |
| Baseline CNN (scratch, augmented, ColorJitter 0.2/0.2) | 78.89% | 0.428 | 78.75% | 0.460 | 0.14 pts |
| Baseline CNN (scratch, augmented, ColorJitter 0.1/0.1) | 82.64% | 0.428 | 78.75% | 0.590 | 3.89 pts |
| ResNet-18 (scratch, no augmentation) | 91.25% | 0.205 | 86.25% | 0.377 | 5.00 pts |

**Still to run (planned):** Baseline CNN and ResNet-18 (pretrained) / DenseNet-121 (pretrained), all on the seeded split, to complete a fully consistent table.

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Honest read — architecture alone does *not* close the gap here:** the corrected, seeded ResNet-18 (scratch, no augmentation) result shows an **18.47-point gap** — larger than the baseline CNN's 11.81-point gap, not smaller. This reverses an earlier (incorrect) reading based on an unseeded run, which showed only a 5.00-point gap and looked like architecture alone (batch norm, residual connections) was substantially reducing overfitting independent of augmentation or pretraining. With a fair, matched split, that's not what the data shows: ResNet-18's much larger capacity (~11M parameters vs. ~25k in the baseline's final layer) appears to make it *more* prone to overfitting than the baseline when given nothing else to regularize it.

**Augmentation is doing substantial, load-bearing work for ResNet-18 specifically:** on the same seeded split, augmentation cuts ResNet-18's gap from 18.47 to 3.75 points — a much larger absolute and proportional reduction than augmentation produced for the baseline CNN. This matters directly for the research question: it means a meaningful share of any future gap reduction seen in the pretrained ResNet-18/DenseNet-121 runs could plausibly come from augmentation continuing to do this work, not from pretraining specifically. Isolating pretraining's unique contribution will require reading the pretrained results *relative to* this 3.75-point augmented-scratch baseline, not relative to the no-augmentation number.

**Augmentation strength comparison (baseline CNN, unseeded, informal, see Known bugs):** two `ColorJitter` strengths were tested — `brightness=0.2, contrast=0.2` and `brightness=0.1, contrast=0.1` (`RandomCrop(224, padding=12)` held constant across both, to isolate the effect of intensity jitter specifically, since TB findings are picked out largely by density/contrast differences on the film). Test accuracy was identical between the two settings (78.75%). The stronger setting (0.2/0.2) nearly eliminated the gap but showed lower train accuracy and worse test loss — a plausible sign of mild underfitting, where the augmentation is aggressive enough relative to the ~720-image training set that the model struggles to fit the training data well. **0.1/0.1 was chosen** as the setting carried forward into all subsequent experiments: it meaningfully reduces overfitting while keeping train accuracy and test loss more favorable than the stronger setting.

## Known bugs already hit and fixed

- **Unseeded random split (significant, affects result comparability):** `get_dataloaders`'s `random.shuffle(all_images)` had no fixed seed, so every training run — across multiple days — used a genuinely different random 90/10 train/test split. With only 80 test images, this alone produced accuracy swings of up to 20 points between otherwise-identical runs (confirmed directly: rerunning ResNet-18 scratch+augmented on an unseeded split gave 66.25% test accuracy / 15.28pt gap, while the same config on the seeded split gave 76.25% / 3.75pt gap). This means **all results recorded before the fix are not safely comparable to each other or to results after the fix** — the "Earlier, unseeded runs" table above is kept for the record but should not be used to draw architecture/pretraining conclusions. Fixed by adding `random.seed(42)` immediately before `random.shuffle(all_images)` inside `get_dataloaders`, so every call produces an identical split.
- `from matplotlib import transforms` silently shadowed `torchvision.transforms`, causing `AttributeError: module 'matplotlib.transforms' has no attribute 'Compose'`. Fixed by removing the matplotlib import and using `from torchvision import transforms`.
- `DataLoader` used in `data.py` without being imported — `NameError`.
- Training loop originally referenced `train_loader` before it was created, because the loop sat outside/after the `if __name__ == "__main__":` block where `train_loader` was actually built. Fixed by moving the whole training loop inside that block.
- `num_workers` default of 2 was a real slowdown given the huge source image files (Montgomery images are ~4892×4020 px before resizing). Bumped to 4 (out of 8 available CPU cores) via `get_dataloaders(..., num_workers=4)`.
- `data.py`'s original single `get_transforms()` returned one transform applied before `random_split`, making separate train/test augmentation impossible. Fixed by splitting the raw image list first (shuffled, seeded), then building two separate `TBXrayDataset` objects from the two slices, each with its own transform.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so test accuracy has real sampling noise. This was confirmed directly and dramatically by the unseeded-split bug above (up to ~20pt swings on an otherwise identical config), not just a theoretical concern
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single fixed 90/10 train/test split (seed=42). Reported generalization gaps can be measured and compared *within* this seeded set of results, but with n≈800 and one split, their stability against a *different* split can't be strongly claimed — the seeded-vs-unseeded comparison above shows this concretely. A more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows
- Only one from-scratch capacity control (ResNet-18) is planned — DenseNet-121's scratch/pretrained gap contribution won't be separately isolated, so its "independent of capacity" comparison will rely partly on the ResNet-18 control's architecture-effect finding generalizing across backbones, which is a real assumption, not confirmed
- Baseline CNN has not yet been rerun on the seeded split (with or without augmentation) — the baseline rows in the results table above are all from the earlier, unseeded runs and should be rerun before final conclusions are drawn
- Augmentation hyperparameters (`ColorJitter` strength, crop padding) were chosen via a single informal A/B comparison (0.2/0.2 vs 0.1/0.1) on the unseeded split, not a systematic sweep — the chosen values are defensible but not proven optimal, and that comparison itself inherits the split-variance caveat above

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
- [x] First baseline result recorded (93.06% train / 81.25% test accuracy) — *unseeded, needs rerun*
- [x] Research question formally locked in
- [x] ResNet-18 (scratch, capacity control) architecture built and trained — seeded, with and without augmentation
- [x] Data augmentation added to baseline (train-only, `RandomCrop` + `ColorJitter`, strength compared and locked in at 0.1/0.1)
- [x] Reproducible train/test split (fixed seed) — fixed after discovering split-variance bug
- [x] Augmentation toggle (`augment=True/False` parameter through `get_dataloaders` → `get_transforms`)
- [ ] Baseline CNN rerun on seeded split (with and without augmentation)
- [ ] Transfer learning: ResNet-18 (pretrained)
- [ ] Transfer learning: DenseNet-121 (pretrained)
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)