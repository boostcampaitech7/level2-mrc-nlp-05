import argparse
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from module.dpr.dense_ret_train import ret_train
from module.evaluate_ret import ret_evaluate
from module.inference import inference
from module.train import train


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(sys.path)

    if cfg.mode == "train":
        print("train")

        train(cfg.train)
    elif cfg.mode == "inference":
        print("inference")

        inference(cfg.inference)
    elif cfg.mode == "ret_train":
        print("ret_train")

        ret_train(cfg.ret_train)
    elif cfg.mode == "ret_evaluate":
        print("ret_evaluate")

        ret_evaluate(cfg.ret_evaluate)


if __name__ == "__main__":
    main()
