# Does wiring protect the brain? Whole-brain connectome simulation of anoxic spreading depolarization in the *Drosophila* brain

Code, small result files and figures for the preprint

> İlke Barış Şenyurt (2026). *Does Wiring Protect the Brain? Whole-Brain Connectome Simulation of Anoxic Spreading Depolarization in the Drosophila Brain.* bioRxiv, doi: [to add]

Author: İlke Barış Şenyurt ([ORCID 0009-0001-5999-4500](https://orcid.org/0009-0001-5999-4500))

The version used in the paper is release **v1.0** of this repository, archived on Zenodo: doi: [to add].

## What this is

A whole-brain model of the fruit fly: a leaky integrate-and-fire network on the connectome (Shiu et al. 2024), coupled to extracellular K⁺ in 10 µm cubes of neuropil, with energy failure, glial K⁺ buffering and spreading depolarization (SD). Oxygen is cut and the model brain dies through SD. The question is whether the real wiring changes how the brain dies, compared with degree-preserving rewired brains.

It was run on two independent connectomes: the female FlyWire brain (release 783) and the male FlyEM CNS (v1.0, brain only).

**Main results** (see the preprint for every caveat):

- In both connectomes the real wiring delayed death: half of all neurons were blocked 6.7 s (female) and 8.7 s (male) later than in rewired brains.
- About half to four-fifths of this protection remained when firing rates were matched.
- At the same depolarization, rewired neurons fired 1.7–1.8 times faster. Across conditions, this firing level tracked the time of death far better than the healthy rate did.
- Of the connection sets tested, and per synapse kept, the excitatory–inhibitory connections among a group of highly connected neurons that excite and inhibit each other carried the most protection. Neither connection strength nor the exact reciprocal pairs explained this.

Everything here is a simulation of one connectome per sex. The model was calibrated to fly measurements of SD speed, onset and recovery, so those matches are a fit, not a validation.

## Repository layout

| Folder | Contents |
|---|---|
| `code/` | The frozen model (v11) and every experiment in the paper. Run scripts from inside this folder. |
| `preprocessing/` | Google Colab notebooks that turn the public connectome data into the model's input files. |
| `results/runs/` | Small result files of every frozen-model run: per-run summaries, measures, calibrations, pre-registered verdicts, cube geometry. |
| `results/logs/` | The console log of every experiment as it ran. |
| `figures/` | Figures 1–7 and S1–S3 (PNG and PDF) and their legends. |
| `docs/project_log.md` | The project log: every model version, calibration and pre-registered hypothesis, written before each run, with its result. |
| `development/` | Earlier model versions (v2–v10) and development scripts, kept for the record. They are not needed to reproduce the paper. |

### Scripts in `code/`

| Script | What it does | Paper |
|---|---|---|
| `death_v11.py`, `fast_sim_v11.py` | The frozen model (LIF network, K⁺ cubes, glia, energy, SD; Numba kernels) | Methods 2.2–2.6 |
| `runner.py` | Runs conditions, healthy control, summaries | – |
| `stage15.py` | Final calibration that froze v11 | Table 1 |
| `stage16.py` | Main experiment: real vs rewired, E→I control (H1–H3) | 3.2–3.4, Table 2 |
| `stage17.py` | Geometry export, reoxygenation, no-lamina, first keep_loops and break_loops runs | 3.4, 3.6–3.7 |
| `stage18.py` | Rate-matched control (H4, H5) | 3.5 |
| `stage19.py` | Onset-site analysis; clearance cut 0.35 (H8) | 3.3, 3.6 |
| `stage20.py` | Other measures of death; E/I-input-preserving rewiring (H6); 20-s warm-up (H7) | 3.4, 3.6 |
| `stage21.py` | Glial capacity (H9); receiver-specific rewiring (H11); firing at −40 mV | 3.4–3.6 |
| `stage22.py`, `stage23.py`, `stage25.py` | Glutamate-neutral test (H10) and the weight-only control | 3.6, Table 4 |
| `stage24.py` | Rate-matched receiver splits (H12), keep_loops replication (H13), photoreceptors (H14) | 3.4, 3.6 |
| `stage26.py` | Rate-matched loop controls (H15–H17) | 3.4, 4.2 |
| `stage27.py` | Strength-matched non-loop control (H18) | 3.4, 4.2 |
| `stage28.py` | Loop connections kept with partners shuffled (H19) | 3.4, 4.2 |
| `check_cubes.py`, `check_cubes2.py`, `check_loops.py` | Checks without simulation (cube counts, neuron overlap of loop sets) | 4.5, 3.4 |
| `make_figures.py` | All figures from the run outputs | Figures |

Each script's header states the hypotheses and reading rules exactly as they were fixed before it ran. In these comments, "review" refers to reviews of the draft by Claude (Anthropic) in separate sessions (see *Use of AI tools* below), not to independent human peer review.

## Data

All data are public. They are not included here because of their size. Put the files in `code/` (or edit the file names at the top of `death_v11.py`).

**Female (FlyWire, release 783).** From [Codex](https://codex.flywire.ai): `neurons.csv`, `connections_princeton.csv`, `coordinates.csv`, `classification.csv`. The synapse table (`flywire_synapses_783.feather`, 9.5 GB, [Zenodo 10676866](https://doi.org/10.5281/zenodo.10676866)) is turned into `synapse_cubes_10um.npz` by `preprocessing/flywire_synapse_cubes.ipynb`, which runs on Google Colab.

**Male (FlyEM male CNS v1.0).** From [male-cns.janelia.org](https://male-cns.janelia.org). `preprocessing/malecns_to_model_inputs.ipynb` (Google Colab) makes `neurons_male.csv`, `connections_male.csv`, `coordinates_male.csv`, `classification_male.csv` and `synapse_cubes_male_10um.npz`, brain only.

No login or API token is needed for any of these downloads.

## Running

The paper's runs used Python 3.10.5 with numpy 1.26.4, scipy 1.15.3, pandas 2.3.2, numba 0.67.0 and matplotlib 3.10.9, on a Windows laptop (Intel Core i7-1255U, 16 GB RAM). Install the same versions:

```
pip install -r requirements.txt
```

Then, from inside `code/`:

```
python stage16.py
```

Each experiment writes to `runs/<date>_<time>_v11_<name>/`. One 90-s whole-brain run took about 1.5 h on that laptop. Experiments run five conditions in parallel by default; set `STAGE16_WORKERS` (or the matching variable for each stage) to use fewer. `STAGEXX_SCALE=0.1` makes short test runs.

To redraw the figures from a folder of run outputs:

```
python make_figures.py runs figures
```

The figure script needs the full run outputs (per-cube SD times, probe and time-series files, about 110 MB), which are larger than what is kept in `results/runs/`. They will be deposited on Zenodo: doi: [to add].

## Pre-registration

Every hypothesis (H1–H19), with its threshold and run design, was written into the project log (`docs/project_log.md`) before the run that tested it. Four failed and are reported in the paper: H6 (female), H10 (male), H12 and H16.

## Use of AI tools

This project was carried out with extensive help from Claude (Anthropic), an AI assistant. The author posed the question, obtained the data, ran every simulation, decided which of the proposed tests to run, and takes full responsibility for the content. Claude wrote and debugged the simulation and analysis code, proposed most of the model design, controls and analyses, searched for and checked the literature, interpreted the results together with the author, made the figures, and drafted and edited the manuscript. Drafts were reviewed by Claude in separate sessions. The work has not yet been reviewed by a human expert.

## Citation

Please cite the preprint (see `CITATION.cff`). Connectome data: Dorkenwald et al. 2024 and Schlegel et al. 2024 (FlyWire); Berg et al. 2026 (male CNS). Neuron model: Shiu et al. 2024.

## Licence

Code: MIT (see `LICENSE`). Figures and documents: CC BY 4.0.

Questions and corrections: please open an issue on this repository.
