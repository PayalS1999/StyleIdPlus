# StyleID+ Extensions

This repository contains two main extensions to the original StyleID project:

1. **Automasking Branch** (`automask`)
   - Applies automatic foreground/background segmentation to protect subjects during style transfer.

2. **Colour Harmonisation Branch** (`colour_harmonisation`)
   - Adds a post-processing Reinhard LAB mean/std color transfer to perfectly match the style palette.

3 **Adaptive Multi-Scale Style Injection** (`adaptive-style-injection`)
- Injects style features (keys & values) at multiple spatial resolutions of the U-Net decoder—with learned layer-wise weights—so that coarse layers capture global color and layout, while finer layers refine textures and local details.
- Schedules the strength of style injection over the diffusion timesteps via a cosine-annealed γ parameter, preserving content structure early on and smoothly ramping up style influence toward the end for artifact-free, harmonious stylization.
---

## Prerequisites

- **Conda** (or Miniconda)  
- **Git**

## Environment Setup

Each branch has its own environment YAML, capturing all dependencies.

```bash
# Automasking branch
git checkout automask
conda env create -f environment_automask.yaml --name styleid_automask
conda activate styleid_automask

# Colour Harmonisation branch
git checkout colour_harmonisation
conda env create -f environment_colour_harmonisation.yaml --name styleid_colour
conda activate styleid_colour

# Adaptive Multi-Scale & Progressive Style Injection​
git checkout adaptive-style-injection
conda env create -f environment.yaml
conda activate StyleID
```

## Running Style Transfer

1. Automasking and Color harmonization scripts live under `diffusers_implementation/` and follow the same interface:

```bash
python run_styleid_diffusers.py   --cnt_fn <path/to/content.png>   --sty_fn <path/to/style.png>   [--mask_fn <path/to/mask.png>]   --gamma <query_preserve_ratio>   --T <attention_temperature>   --save_dir <output_folder>   [--without_init_adain]   [--without_attn_injection]
```

2. Adaptive Multi-Scale & Progressive Style Injection​

```bash
python run_styleid_with_layer_weights.py --cnt <CONTENT_IMAGES_DIR> --sty <STYLE_IMAGES_DIR> --precomputed "<PRECOMPUTED_FEATURES_DIR or empty>" --layer_weights <LAYER_WEIGHTS_JSON_OR_LIST> --gamma_start <START_GAMMA> --gamma_end <END_GAMMA> --T <ATTENTION_TEMPERATURE> --start_step <INJECTION_START_STEP> --output_path <OUTPUT_DIR> [--without_init_adain] [--without_attn_injection]
```

### Examples

- **Automasking (protect subject)**
  ```bash
  python diffusers_implementation/run_styleid_diffusers.py     --cnt_fn data/cnt/04.png     --sty_fn data/sty/17.png     --gamma 0.75 --T 1.5     --save_dir results/automask_output
  ```

- **Colour Harmonisation (post-process color)**
  ```bash
  python diffusers_implementation/run_styleid_diffusers.py     --cnt_fn data/cnt/04.png     --sty_fn data/sty/17.png     --gamma 0.75 --T 1.5     --save_dir results/harmonised_output
  ```

  ```bash
  python run_styleid_with_layer_weights.py --precomputed "" --layer_weight 0.6 --gamma_start 1.0 --gamma_end 0.45 --output_path outputs_adaptive_style
  ```

## Batch Processing

```bash
# Automask batch
bash run_all_automask.sh outputs_automask

# Colour Harmonisation batch
bash run_all_harmonisation.sh outputs_harmonised
```

## Evaluation

Use ArtFID∞ and CFSD:

1. Prepare inputs:
   ```bash
   python util/copy_inputs.py --cnt data/cnt --sty data/sty
   ```

2. Compute metrics:
   ```bash
   cd evaluation
   python eval_artfid.py      --sty ../data/sty_eval      --cnt ../data/cnt_eval      --tar ../outputs_automask      --batch_size 8      --num_workers 8      --content_metric lpips      --mode art_fid_inf      --device cuda
   ```
   Replace `--tar` with `../outputs_harmonised` for the colour branch.

3. View printed results.

---

Enjoy experimenting with enhanced style injection features!
