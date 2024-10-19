import os
import argparse
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from module.train import train
from module.inference import inference
from module.dense_ret_train import ret_train



@hydra.main(version_base="1.3",config_path="configs", config_name="config")
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
if __name__ == "__main__":
    main()