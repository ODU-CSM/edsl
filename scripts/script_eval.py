from fugep.setup import load_path, parse_configs_and_run
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.chdir('/scratch/ml-csm/projects/fgenom/mtl/mtl-dlRM/training/')

configs = load_path("../configs/config_eval.yml")
parse_configs_and_run(configs, lr=0.2)