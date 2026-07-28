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
- **Transfer learning:** Fine-tuned ResNet-18 / DenseNet-121 (torchvision pretrained weights) — *not yet started, planned next*
- **Augmentation (train set only, toggleable via `get_transforms(augment=True/False)` → `get_dataloaders(..., augment=True/False)`):** `RandomCrop(224, padding=12)` + `ColorJitter(brightness=0.1, contrast=0.1)`. Design choices, reasoned through explicitly rather than copied from the CIFAR-10 exercise:
  - **No horizontal flip** — chest anatomy is not left-right symmetric (heart position, lobe structure) and TB findings are not symmetric either, so a flipped image would not represent a physically valid chest X-ray. Standard practice in the chest X-ray literature avoids this augmentation for the same reason.
  - **Mild `RandomCrop`, not aggressive lung-isolation cropping** — padding of 12px (vs. a naive proportional scale-up from CIFAR-10's `padding=4` on 32px images, which would suggest ~28px) was deliberately scaled back, since aggressive cropping risks cutting into diagnostic lung regions and chest X-rays are centrally framed and consistent across patients, unlike natural photos where CIFAR-10-style jitter originated.
  - **`ColorJitter` restricted to `brightness`/`contrast` only** — `saturation` and `hue` are color concepts with no meaning on a single-channel grayscale image and were left untouched.
  - **`brightness=0.1, contrast=0.1`** — an initial informal comparison against 0.2/0.2 (on an unseeded split, see Known bugs) suggested 0.1/0.1 avoided an underfitting signature seen at the stronger setting; carried forward as the standard config.
  - **Train transform only** — the test set uses a clean `Resize`/`ToTensor`/`Normalize` pipeline with no augmentation, so evaluation always reflects performance on realistic, unmodified data.
  - `augment=True` is the default going into the remaining experiments this week (ResNet-18 pretrained, DenseNet-121), so gap differences between conditions are attributable to architecture/pretraining, not to augmentation changing between runs.
- **Training:** Adam optimizer (lr=0.001), CrossEntropyLoss, 15 epochs, batch size 32
- **Tracking:** *not yet in W&B — currently plain console output, planned*
- **Deployment:** FastAPI endpoint — accepts image, returns prediction + confidence *(not yet built)*

## Results

All results below use the fixed seeded split (seed=42) and are directly comparable to each other. 15 epochs each, batch size 32.

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Gap (Train − Test Acc) |
|---|---|---|---|---|---|
| Baseline CNN, no augmentation | 96.67% | 0.101 | 81.25% | 0.574 | 15.42 pts |
| Baseline CNN, augmented (0.1/0.1) | 77.64% | 0.468 | 68.75% | 0.532 | 8.89 pts |
| ResNet-18 (scratch), no augmentation | 97.22% | 0.073 | 78.75% | 0.943 | 18.47 pts |
| ResNet-18 (scratch), augmented (0.1/0.1) | 80.00% | 0.436 | 76.25% | 0.476 | 3.75 pts |
| ResNet-18 (pretrained) | | | | | |
| DenseNet-121 (pretrained) | | | | | |

*AUC, sensitivity, specificity not yet computed — only accuracy/loss tracked so far.*

**Architecture alone does not reduce the gap — if anything, it makes it worse.** Comparing the two no-augmentation rows: ResNet-18 shows a *larger* gap than the baseline CNN (18.47 vs. 15.42 pts), not a smaller one. ResNet-18's much greater capacity (~11M parameters vs. ~25k in the baseline's final layer alone) appears to make it more prone to overfitting when given nothing else to regularize it, not less. This is worth stating plainly because it revises an earlier, incorrect reading of this project based on unseeded runs (see Known bugs) that looked like architecture alone was doing a lot of the gap-closing work — it isn't.

**Augmentation's effect differs meaningfully by architecture, and isn't uniformly "better."** For the baseline CNN, augmentation shrinks the gap (15.42 → 8.89 pts) but at a real cost: test accuracy drops substantially, 81.25% → 68.75%. That's not an unambiguous win — a smaller gap paired with meaningfully worse real-world performance is a genuine tradeoff, not a strict improvement. For ResNet-18, augmentation's effect is both larger and cleaner: the gap collapses from 18.47 to 3.75 points, while test accuracy only drops modestly, 78.75% → 76.25%.

**The interesting finding is the interaction between capacity and augmentation, not either alone.** Without augmentation, ResNet-18's extra capacity works against it — it overfits more severely than the simpler baseline. With augmentation, ResNet-18 pulls ahead of the baseline on both test accuracy (76.25% vs. 68.75%) and gap (3.75 vs. 8.89 pts). In other words: ResNet-18's architectural capacity only becomes an advantage once it's paired with enough regularization (here, data augmentation) to keep it in check — architecture and augmentation are not independent, additive effects, they interact. This has a direct bearing on the pretrained runs still to come: since augmentation alone already closes most of ResNet-18's gap, isolating what pretraining adds *on top of* that will mean reading the pretrained result relative to this 3.75-point augmented-scratch figure, not relative to the no-augmentation number.

## Known bugs already hit and fixed

- **Unseeded random split (significant, now fixed):** `get_dataloaders`'s `random.shuffle(all_images)` originally had no fixed seed, so every training run used a genuinely different random 90/10 split. With only 80 test images, this alone produced accuracy swings of up to 20 points between otherwise-identical runs (confirmed directly: rerunning ResNet-18 scratch+augmented on an unseeded split gave 66.25% test accuracy / 15.28pt gap, vs. 76.25% / 3.75pt gap on the seeded split). All results now in this README were rerun on the fixed seed (42) for consistency; no unseeded numbers are reported here anymore. Fixed by adding `random.seed(42)` immediately before `random.shuffle(all_images)` inside `get_dataloaders`.
- `from matplotlib import transforms` silently shadowed `torchvision.transforms`, causing `AttributeError: module 'matplotlib.transforms' has no attribute 'Compose'`. Fixed by removing the matplotlib import and using `from torchvision import transforms`.
- `DataLoader` used in `data.py` without being imported — `NameError`.
- Training loop originally referenced `train_loader` before it was created, because the loop sat outside/after the `if __name__ == "__main__":` block where `train_loader` was actually built. Fixed by moving the whole training loop inside that block.
- `num_workers` default of 2 was a real slowdown given the huge source image files (Montgomery images are ~4892×4020 px before resizing). Bumped to 4 (out of 8 available CPU cores) via `get_dataloaders(..., num_workers=4)`.
- `data.py`'s original single `get_transforms()` returned one transform applied before `random_split`, making separate train/test augmentation impossible. Fixed by splitting the raw image list first (shuffled, seeded), then building two separate `TBXrayDataset` objects from the two slices, each with its own transform.

## Limitations

- Dataset size (~800 images total across both sets) is small relative to model capacity — visible overfitting gap between train/test in every condition tested
- Single-center-ish data (two public datasets, not a diverse clinical population)
- No external validation set beyond the 10% held-out split
- No class imbalance check performed yet
- Test set is small (80 images) — each image is ~1.25 percentage points, so test accuracy has real sampling noise. This was confirmed directly and dramatically by the unseeded-split bug above (up to ~20pt swings on an otherwise identical config), not just a theoretical concern
- Labels are sourced from filename convention (`_0`/`_1` suffix), not the free-text radiologist findings, since the free text is inconsistent and non-standardized across cases
- All results are from a single fixed 90/10 train/test split (seed=42). Reported generalization gaps can be measured and compared *within* this seeded set of results, but with n≈800 and one split, their stability against a *different* split can't be strongly claimed — the seeded-vs-unseeded comparison above shows this concretely. A more rigorous version (k-fold cross-validation) is a reasonable future extension if time allows
- Only one from-scratch capacity control (ResNet-18) is planned — DenseNet-121's scratch/pretrained gap contribution won't be separately isolated, so its "independent of capacity" comparison will rely partly on the ResNet-18 control's architecture-effect finding generalizing across backbones, which is a real assumption, not confirmed
- Augmentation hyperparameters (`ColorJitter` strength, crop padding) were chosen via a single informal A/B comparison on an unseeded split, not a systematic sweep — the chosen values are defensible but not proven optimal, and that specific comparison inherits the split-variance caveat above (though the augmented-vs-no-augmented conclusions reported in Results here are from the corrected seeded split)
- The baseline CNN's augmented result shows a real accuracy/gap tradeoff (higher gap-reduction, lower test accuracy) that hasn't been further diagnosed — e.g., whether a milder augmentation setting would recover some test accuracy while keeping most of the gap reduction is untested for the baseline specifically (only tested for the unseeded case)

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
- [x] Reproducible train/test split (fixed seed) — added after discovering split-variance bug
- [x] Augmentation toggle (`augment=True/False` parameter through `get_dataloaders` → `get_transforms`)
- [x] Baseline CNN: no-augmentation and augmented results recorded on seeded split
- [x] ResNet-18 (scratch, capacity control): no-augmentation and augmented results recorded on seeded split
- [ ] Transfer learning: ResNet-18 (pretrained)
- [ ] Transfer learning: DenseNet-121 (pretrained)
- [ ] W&B experiment tracking
- [ ] AUC / sensitivity / specificity computed
- [ ] FastAPI deployment endpoint
- [ ] 500-word writeup (question → method → results → limitations)

## References

- Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases." *Quantitative Imaging in Medicine and Surgery*, 2014.
- Dataset source: [NLM Tuberculosis Chest X-ray Datasets](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html)