import numpy as np
import torch


def set_random_seed(random_seed):
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def print_and_save(s, fname):
    print(s)
    with open(fname, "a") as f:
        f.write(s + "\n")
