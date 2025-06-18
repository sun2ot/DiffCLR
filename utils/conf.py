from dataclasses import dataclass
import toml
import os

"""
*The configurations in the @dataclass are initialized with default values.
*The `load_config` function loads a TOML configuration file and updates the dataclass instances with the values from the file.
"""

@dataclass
class BaseConfig:
    latdim: int = 64
    topk: int = 20
    gpu: str = "0"
    seed: int = 42
    hidden_dim: int = 1024
    step_dim: int = 10
    cl_method: int = 0
    enable_save: bool = False
    timestamp: str = ""

@dataclass
class DataConfig:
    name: str = ""
    dir: str = ""
    # The following configurations will be updated when load_data()
    user_num: int = 0
    item_num: int = 0
    image_feat_dim: int = 0
    text_feat_dim: int = 0
    audio_feat_dim: int = 0

@dataclass
class HyperConfig:
    modal_cl_temp: float = 0.5
    modal_cl_rate: float = 0.01
    cross_cl_temp: float = 0.2
    cross_cl_rate: float = 0.2
    noise_degree: float = 0.2
    noise_scale: float = 0.1
    noise_min: float = 0.0001
    noise_max: float = 0.02
    steps: int = 5
    align_weight: float = 0.1
    residual_weight: float = 0.5
    modal_adj_weight: float = 0.2
    sampling_step: int = 0
    knn_topk: int = 10

@dataclass
class TrainConfig:
    lr: float = 0.001
    batch: int = 1024
    reg: float = 1e-5
    epoch: int = 50
    test_epoch: int = 1
    use_lr_scheduler: bool = True

@dataclass
class Config:
    base: BaseConfig = BaseConfig()
    data: DataConfig = DataConfig()
    hyper: HyperConfig = HyperConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: str) -> Config:
    with open(path, 'r') as file:
        raw_config = toml.load(file)
    return Config(
        base = BaseConfig(**raw_config.get("base", {})),
        data = DataConfig(**raw_config.get("data", {})),
        hyper = HyperConfig(**raw_config.get("hyper", {})),
        train = TrainConfig(**raw_config.get("train", {})),
    )


def check_config(config: Config):
    """
    Check the configuration values and raise errors if any are invalid.
    """
    save_path = os.path.join("persist", config.data.name)
    if config.base.enable_save:
        if not os.path.exists(save_path):
            print("✅ Created `persist` path for saving.")
            os.makedirs(save_path)
    else:
        print("❗️ Ensure that you do not need to enable the save option.")
