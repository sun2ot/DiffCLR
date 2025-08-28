# Diffusion-based contrastive learning for multimodal recommendation

## Dependencies

- python==3.10.16
- pytorch==1.12.1
- numpy==1.21.6
- scipy==1.9.1
- scikit-learn==1.4.0
- toml==0.10.2
- safetensors==0.5.3
- scikit-learn==1.4.0

You can use `conda/mamba env create -f environment.yml` to create a virtual environment.

Of course, the more advanced pixi can conveniently do this via `pixi install`.

## How to run

```bash
# tiktok
python main.py -c conf/tiktok.toml

# yelp
python main.py -c conf/yelp.toml

# sports
python main.py -c conf/sports.toml
```

If you use pixi, we have already prepared the tasks:
```bash
pixi run tiktok
pixi run yelp
pixi run sports
```

## Reproduce

The hyperparameters required to reproduce the results of paper have been recorded in default configuration file.

If you need to reproduce the paper results exactly the same, please refer to the `Reproduction` section in `docs.ipynb`. We have made the required safetensors files public on [Google Drive](https://drive.google.com/file/d/1o1Y2aBqnhM7ipGuu4RJzO15mr3GZnLuT/view?usp=sharing).

## Datasets

The datasets of DiffCLR have published on [HuggingFace](https://huggingface.co/datasets/sun2ot/DiffCLR), including `Tiktok`, `Yelp`, and `Amazon-Sports`.

## Acknowledgments

Our work is based on the implementation of [DiffMM](https://github.com/HKUDS/DiffMM). Thank them for their contributions to this field.

## Note

The code of the model will be made public after the paper is accepted.