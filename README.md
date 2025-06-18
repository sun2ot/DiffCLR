## Dependencies

- python==3.10.16
- pytorch==1.12.1
- numpy==1.21.6
- scipy==1.9.1
- scikit-learn==1.4.0
- toml==0.10.2
- safetensors==0.5.3

We recommend using `conda/mamba env create -f environment.yml` to create a virtual environment.

## How to run

```bash
# tiktok
python main.py -c conf/tiktok.toml

# yelp
python main.py -c conf/yelp.toml

# sports
python main.py -c conf/sports.toml
```

## Reproduce

The hyperparameters required to reproduce the results of paper have been recorded in default configuration file.

If you need to reproduce the paper results exactly the same, please refer to the `Reproduction` section in `docs.ipynb`.

## Datasets

The datasets of DiffCLR have published on [HuggingFace](https://huggingface.co/datasets/sun2ot/DiffCLR), including `Tiktok`, `Yelp`, and `Amazon-Sports`.

## Acknowledgments

Our work is based on the implementation of [DiffMM](https://github.com/HKUDS/DiffMM). Thank them for their contributions to this field.

## Note

The code of the model will be made public after the paper is accepted.