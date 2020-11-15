import glob
import os
import logging
import re

import torch

def get_logger(filename, verbosity=1, name=None):
    if not os.path.exists(filename):
        open(filename, mode='w+')
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def findLastCheckpoint(save_dir, name):
    file_list = glob.glob(os.path.join(save_dir, name, ('model_*.pth')))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def evalute_acc(Y_pred, Y):
    return (Y_pred.argmax(dim=1) == Y).to(torch.float32).mean()
