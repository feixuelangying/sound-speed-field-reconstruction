# Sound-speed field reconstruction and uncertainty evaluation

Source code for reconstructing a four-dimensional ocean sound-speed field from sparse asynchronous temperature and salinity profiles and evaluating pointwise uncertainty.

- `exp/` contains preprocessing, scene construction, model training, GP fitting, calibration, acceptance tests, baselines, ablations, and sensitivity experiments.
- `exp/lib/` contains the shared numerical and data-processing routines.

## Environment

Python 3.12 is recommended. Install the experiment dependencies with `python -m pip install -r requirements.txt`. The original runs used Python 3.12.3 and PyTorch 2.7.1 with CUDA 11.8. The scripts select CUDA when available. The requirements use a NumPy range compatible with the declared dependency of GSW 3.6.23. The original archived NumPy version was 1.26.4.

## Released checkpoint

`weights/exp3/meanet_final.pt` is the trained residual mean network used in the archived experiments. Its companion `meanet_final.json` provides the architecture, input feature selection, and normalization configuration. Keep the two files together. Running EXP-3 retrains the network and overwrites this local pair, so use a separate working copy if you want to preserve the released checkpoint while retraining.

## Experiment entry points

The experiments require study data. This repository includes the code and trained checkpoint only. Run the commands from the repository root using Windows PowerShell.

```powershell
python -m exp.exp0_pipeline
python -m exp.exp1_uncertainty
python -m exp.exp2_scenes
python -m exp.exp2_verify
python -m exp.exp3_meanet

$env:HP_DIR = 'exp4'
python -m exp.exp4_gp
python -m exp.exp5_conformal
python -m exp.exp6c_banded_accept
$env:K_MAIN_FILE = 'k_main_v1.json'
python -m exp.exp6d_k_conservative
Remove-Item Env:K_MAIN_FILE

$env:HP_DIR = 'exp4b'
python -m exp.exp11_s2moment
$env:EXP6C_OUT = 'exp6c_v2'
$env:EXP5B_OUT = 'exp5b_v2'
$env:EXP6D_OUT = 'exp6d_v2'
python -m exp.exp5_conformal
python -m exp.exp6c_banded_accept
python -m exp.exp6d_k_conservative
Copy-Item results/frozen/k_main.json results/frozen/k_main_v2.json
python -m exp.exp6_acceptance
python -m exp.exp7_scenarios
python -m exp.exp8_baselines
python -m exp.exp9_ablation
python -m exp.exp10_sensitivity
python -m exp.exp12_compare_v1v2
```

The commands generate local outputs under `results/` and `weights/`. The historical v1/v2 comparison in `exp.exp12_compare_v1v2` reads both sets of archived test predictions and their corresponding frozen k configurations. The exported profile example in EXP-10 represents a residual profile relative to HYCOM.

