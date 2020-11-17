import glob
import os
import logging
import re
import datetime
import torch


def get_logger(verbosity=1, name=None):
    str_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.join(os.getcwd(), 'log')
    log_name = os.path.join(log_path, (str_date+'.log'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_name):
        open(log_name, mode='w+')

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(log_name, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def findLastCheckpoint(saveDir, model):
    file_dir = os.path.join(saveDir, model)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_list = glob.glob(os.path.join(file_dir, ('model_*.pth')))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    model = None
    if initial_epoch > 0:
        model = torch.load(os.path.join(
            saveDir, model, 'model_%03d.pth' % initial_epoch))
    return (initial_epoch, model)


def evalute_acc(Y_pred, Y):
    return (Y_pred.argmax(dim=1) == Y).to(torch.float32).mean()
