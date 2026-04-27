# MMPT Transformer

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

Repository containing tools, training/inference code, and analysis for the MMPT project.

## Overview

This repo organizes code and data used to train, run inference with, and analyze MMPT models. It includes training and inference scripts, example data, analysis notebooks.

## Repository layout

- `model/` - Training and inference scripts
  - [model/train_MMPT.py](model/train_MMPT.py#L1)
  - [model/inference_MMPT.py](model/inference_MMPT.py#L1)
- `analysis/` - Jupyter notebooks and analysis utilities
  - [analysis/analysis_pmv17.ipynb](analysis/analysis_pmv17.ipynb#L1)
  - [analysis/analysis_pmv17_pmv21.ipynb](analysis/analysis_pmv17_pmv21.ipynb#L1)
  - [analysis/calc_properties.py](analysis/calc_properties.py#L1)
- `data/` - Example input data used by scripts and notebooks
  - `chembl_250320_MVR33so.RGP.csv`
  - `chembl_neutralized_250320_smiles.txt`
  - `pmv17.mmp.csv`
  - `pmv_2017_to_pmv_2021_mmps.csv`


See the `LICENSE` file for licensing terms.

## Quick start

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Note: the `requirements.txt` file has pinned versions to ensure reproducible environments. This ensures the code runs consistently across different systems and time periods.

## Common tasks


Key training flags available in `model/train_MMPT.py` include:
- `--model_type` (T5 | GPT | T5Chem | RealT5Chem)
- `--dataset` (dataset name)
- `--epochs`, `--batch_size`, `--learning_rate`, `--output_dir`
- `--use_wandb` to enable Weights & Biases logging


Key inference flags available in `model/inference_MMPT.py` include:
- `--model_type`, `--model_path` (required)
- `--dataset`, `--batch_size`, `--max_length`
- `--num_beams`, `--num_return_sequences`, `--early_stopping`

Adjust `num_beams` and `num_return_sequences` so that `num_beams >= num_return_sequences`.


## Notebooks and analysis

Open the notebooks in `analysis/` for visualizations and exploratory analyses. Example notebooks:

- [analysis/analysis_pmv17.ipynb](analysis/analysis_pmv17.ipynb#L1)
- [analysis/analysis_pmv17_pmv21.ipynb](analysis/analysis_pmv17_pmv21.ipynb#L1)

**Note:** The analysis notebooks require external sample data files (model outputs, REINVENT results) that are not included in this repository. To use these notebooks:

1. Prepare or obtain the required sample data files
2. Update the `DATA_ROOT` configuration variable at the top of each notebook to point to your data directory
3. Run the notebook cells with your configured data path

The analysis utilities (`analysis/calc_properties.py`, `analysis/canonicalize.py`, `analysis/get_umap.py`) are provided for post-processing model outputs and can be adapted for your own analysis workflows.

To start a notebook server locally:

```bash
pip install jupyterlab
jupyter lab
```

## Data

Some exemplary datasets are stored in `data/`. 

## Troubleshooting

- GPU not detected: make sure CUDA and a compatible `torch` build are installed. Check `torch.cuda.is_available()`.
- Tokenizer / model errors: ensure `--model_path` points to the trained model directory containing the tokenizer and model weights.



## Contact

If you need help or want changes to this README (commands, examples, or extra sections), open an issue.

## License

This project is licensed under the terms in the `LICENSE` file.

## Citation
Multiple papers are associated with this repo.

[^1] a chemistry-focused preprint, describing the comparison of **MMPT-FM** with MMP variants **MMP-M2M**, **MMP-C2V**, **MMP-M2T**.  
[^1]: Pang, Hao-Wei, et al. "Scalable and Generalizable Analog Design via Learning Medicinal Chemistry Intuition from Matched Molecular Pair Transformations." (2026).
```bibtex
@article{pang2026scalable,
  title={Scalable and Generalizable Analog Design via Learning Medicinal Chemistry Intuition from Matched Molecular Pair Transformations},
  author={Pang, Hao-Wei and Zhang, Peter Zhiping and Pan, Bo and Zhao, Liang and Yu, Xiang and Zhang, Liying},
  year={2026}
}
```
[^2] a techinical paper describing the modeling details of **MMPT-FM** and its extention **MMPT-RAG**. 
[^2]: Pan, Bo, et al. "Retrieval-Augmented Foundation Models for Matched Molecular Pair Transformations to Recapitulate Medicinal Chemistry Intuition." arXiv preprint arXiv:2602.16684 (2026).
```bibtex
@article{pan2026retrieval,
  title={Retrieval-Augmented Foundation Models for Matched Molecular Pair Transformations to Recapitulate Medicinal Chemistry Intuition},
  author={Pan, Bo and Zhang, Peter Zhiping and Pang, Hao-Wei and Zhu, Alex and Yu, Xiang and Zhang, Liying and Zhao, Liang},
  journal={arXiv preprint arXiv:2602.16684},
  year={2026}
}
```
[^3] Workshop paper at The 2nd Workshop on LLMs4Bio at AAAI 2025 reporting predecessors of **MMP-M2M** and **MMP-M2T** (they were **Mol2Mol** and **Mol2Trans**, respecitively). 
[^3]: Pan, Bo, et al. "Transformer-Based Approach for Automated Functional Group Replacement in Chemical Compounds." arXiv preprint arXiv:2601.07930 (2026).
```bibtex
@article{pan2026transformer,
  title={Transformer-Based Approach for Automated Functional Group Replacement in Chemical Compounds},
  author={Pan, Bo and Zhang, Zhiping and Spiekermann, Kevin and Chen, Tianchi and Yu, Xiang and Zhang, Liying and Zhao, Liang},
  journal={arXiv preprint arXiv:2601.07930},
  year={2026}
}
```
