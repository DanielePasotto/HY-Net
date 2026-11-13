# Experiments

Protocols behind the comparisons of the paper and the follow-up experiments. All commands are run from the repository root; the datasets and `outputs/` are not versioned (see the README).

## 1. Cross-validated comparison with baselines and ablations
All models are trained and scored with the same code, folds and metrics. Folds are **grouped by well** (optionally by donor/plate with `--groups_csv`),
every well is in the test set of exactly one fold, so each model gets out-of-fold predictions for all the wells.

```bash
uv run --extra cu128 hynet-make-folds                      # -> folds/k5_seed42.json (5 folds, 66 wells, balanced by organoid density)

# for each fold k = 0..4
F="--folds_file folds/k5_seed42.json --fold $k"
uv run --extra cu128 hynet-train --harmnet true --hynet false --amp true $F      # HarmNet of the fold (trained on the fold's training wells only)
for m in hynet hynet_frozen hynet_nogate harm_unet ynet_raw unet attunet; do
  uv run --extra cu128 hynet-train    --model $m --amp true $F         # weights -> outputs/runs/<model>/fold<k>/model.pth
  uv run --extra cu128 hynet-evaluate --model $m $F                    # masks and metrics -> outputs/eval/<model>/fold<k>/
done

uv run --extra cu128 hynet-compare --folds_file folds/k5_seed42.json --models hynet unet attunet ynet_raw
```
`hynet-compare` prints per-well mean [95% bootstrap CI] of each metric and the paired per-well differences with respect to the reference model (`--reference`, default the first).
Models (`--model`): `hynet` (full), `hynet_frozen` (HarmNet not fine-tuned), `hynet_nogate` (concatenation instead of attention+gating),
`ynet_raw` (dual-stream Y-Net without harmonization), `harm_unet` (HarmNet + single-stream U-Net), `unet`, `attunet`.

`--amp true` trains in fp16 mixed precision (loss in fp32). The dual-stream models need it on a 12 GB GPU (about 11.5 GB peak for `hynet` with batch size 4; if it still runs out of memory, `--micro_batches 2` accumulates the gradients over two halves of each batch: same batch size, about half the activation memory);
`hynet-benchmark --amp true` reports memory and speed per model, and `hynet-benchmark --inference true` the number of parameters and the latency of a forward pass.

**Training seeds.** `--seed` sets the seed; to keep the runs of other seeds apart use `--runs_root outputs/runs_s43` for `hynet-train` and `hynet-evaluate`
and `--out_dir outputs/eval_s43` for `hynet-evaluate`. `hynet-compare --extra_eval_dirs outputs/eval_s43 outputs/eval_s44` then also reports, for the models
available in them, the per-seed Dice, its standard deviation between seeds and the paired differences of the seed-averaged per-well values.

Masks produced by any other model can be scored by writing them as binary PNGs (480x640) in `outputs/eval/<model>/fold<k>/predictions/` and using `--reuse_predictions true`.

## 2. External datasets
Models trained on the folds can be applied, without adaptation, to public datasets with masks. `hynet-prepare-external` converts them to the layout of this
repository (each image is its own "well"; `--letterbox` keeps the aspect ratio instead of stretching the image to 480x640):

```bash
uv run --extra cu128 hynet-prepare-external --dataset organosegment    # OrgaSegment OrganoidBasic val+eval -> data/external/organosegment, folds/external_organosegment.json
uv run --extra cu128 hynet-prepare-external --dataset organoid         # OrganoID testing sets          -> data/external/organoid, folds/external_organoid.json

# the model of every fold k on the external images (all the images are "fold 0" of the external folds file)
uv run --extra cu128 hynet-evaluate --model hynet --folds_file folds/external_organoid.json --fold 0 --dataset_path data/external/organoid \
    --weights_path outputs/runs/hynet/fold$k/model.pth --out_dir outputs/eval_ext/organoid/m$k

# score and average the five fold models
uv run --extra cu128 hynet-compare --dataset_path data/external/organoid --folds_file folds/external_organoid.json --models hynet unet \
    --eval_dir outputs/eval_ext/organoid/m0 --extra_eval_dirs outputs/eval_ext/organoid/m{1,2,3,4} --replicates_label "fold models"
```
`scripts/visualize_harmonization.py` saves panels (input, HarmNet before and after the end-to-end training) and statistics of what HarmNet does to the images of each dataset.

## 3. Few-shot adaptation to an external dataset
How many annotated images of a new dataset does a model need? The images of an external dataset are split in 4 folds, and the model trained on fold *j* of our data
is fine-tuned on *N* images of the external training wells (`--init_weights`, `--n_train`, `--n_val`, `--subset_seed`; the subsets of one seed are nested) and scored on the external test fold *j*:

```bash
uv run --extra cu128 hynet-make-folds --dataset_path data/external/organoid --folds_k 4 --folds_out folds/external_organoid_cv4.json
scripts/finetune_external.sh organoid hynet pretrained   # also: ynet_raw, unet; `scratch` trains the same N images from random weights
uv run --extra cu128 hynet-compare --folds_file folds/external_organoid_cv4.json --dataset_path data/external/organoid --models hynet ynet_raw unet \
    --eval_dir outputs/ft/organoid/pretrained/n8_r0 --extra_eval_dirs outputs/ft/organoid/pretrained/n8_r1 --replicates_label "random draws"
```

## 4. Training on pooled datasets (leave-one-dataset-out)
Our dataset, OrgaSegment and OrganoID can be pooled to train on two of them and test on the third. `hynet-prepare-pool` writes the pooled dataset (`data/pool`, files named
`IMG_<domain>-<id>_Interval<n>`, domain `ours`, `orgaseg` or `organoid`; instance masks are binarized) and the folds `folds/lodo_<domain>.json`, which train on the training images of the other two
datasets and test on `<domain>`. The training samples are drawn with the same probability from each dataset (`--domain_balance true --samples_per_epoch N`), and HarmNet is retrained on the pool:

```bash
uv run --extra cu128 hynet-prepare-pool      # needs the original datasets in data/ (see the paths in --help)
uv run --extra cu128 hynet-make-targets --input_root data/pool --output_root data/pool_processed
scripts/lodo.sh organoid hynet               # also: ynet_raw, unet; test domain ours | orgaseg | organoid
uv run --extra cu128 hynet-compare --folds_file folds/lodo_organoid.json --dataset_path data/pool --models hynet ynet_raw unet --eval_dir outputs/lodo/organoid/eval
```

## 5. Robustness to photometric perturbations and augmentation baseline
`hynet.perturb` defines five perturbations of the (min-max normalized) input image: gamma, additive noise, Gaussian blur, an uneven-illumination ramp and JPEG compression
(brightness and contrast changes are not included: the normalization undoes them). `hynet-robustness` segments the test wells of a trained model clean and with every
level of the sweep and writes the per-well Dice; `scripts/robustness.sh` runs it over folds and models, `scripts/summarize_robustness.py` compares the models (Dice and Dice lost from the clean images, paired over wells):

```bash
scripts/robustness.sh cv_s42 outputs/runs hynet ynet_raw unet hynet_nogate        # -> outputs/robustness/cv_s42/<model>/fold<k>/per_well.csv
python scripts/summarize_robustness.py --reference ynet_raw --model hynet=hynet:cv_s42 --model ynet_raw=ynet_raw:cv_s42
```

`hynet-postproc` (driver `scripts/postproc.sh`, summary `scripts/summarize_postproc.py`) tests two inference options on the same models: the threshold chosen on the validation wells of each fold, and averaging the probabilities of the image and its horizontal, vertical and double flips. Both are applied to every model compared.

`--augment true` trains the segmentation model with a random version of the same perturbations (every one with probability 1/2 and a random level), the baseline against which the
harmonization is compared (`AUGMENT=true scripts/lodo.sh organoid ynet_raw`). The default ranges leave the strongest levels of the sweep out; `--augment_range wide` (`AUGMENT_RANGE=wide`) contains all of them, to check whether a gain outside the training ranges survives an augmentation that covers them. `--perturb noise:0.05` applies one perturbation to all the images of `hynet-evaluate` and the other commands.
