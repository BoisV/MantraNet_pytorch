import os
import re
import torch
import glob
import logging
from torch.utils.data.dataloader import DataLoader
from Mantra_net_pytorch.network import MantraNet, FeatexVGG16, IMTFE, bayarConstraint
from torch import device, mode, optim, nn, cuda
import argparse
from torchvision.transforms import transforms
from Mantra_net_pytorch.dataset import MyDataset


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


logger = get_logger('./log/train.log')

parser = argparse.ArgumentParser(description='Mantra Net')
parser.add_argument('--dataDir', default='./dataset', type=str,
                    help='choose the dir of dataset')
parser.add_argument('--saveDir', default='./nets',
                    type=str, help='choose the saveDir')
parser.add_argument('--name', default='IMTFE',
                    type=str, help='select IMTFE or MantraNet')
parser.add_argument('--patch_size', default=256,
                    type=int, help='set patch size')
parser.add_argument('--number', default=100,
                    type=int, help='set patches number')
args = parser.parse_args()


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


save_dir = './nets'
data_dir = './dataset'


def evalute_acc(Y_pred, Y):
    return (Y_pred.argmax(dim=1) == Y).to(torch.float32).mean()


if __name__ == "__main__":
    # torch.cuda.set_device(1)
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    model = IMTFE(Featex=FeatexVGG16(), in_size=128)
    initial_epoch = findLastCheckpoint(
        save_dir=save_dir, name='IMTFE')  # IMTFE or MantraNet
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(
            save_dir, 'model_%03d.pth' % initial_epoch))
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    batch_size = 64
    batches = 10
    MAX_EPOCH = 1000
    lr = 1e-3

    transform = transforms.Compose([transforms.RandomCrop(size=(128, 128)),
                                    transforms.ToTensor()])
    dataset_train = MyDataset(root_dir='./dataset/train', names_file='./dataset/train/train.txt', transform=transform)
    dataset_val = MyDataset(root_dir='./dataset/val', names_file='./dataset/val/val.txt', transform=transform)
    train_iter = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    val_iter = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logger.info('start training!')
    for epoch in range(MAX_EPOCH):
        # loss_epochs = 0
        # n_batch = 0
        model = model.train()
        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)
            break
        Y_pred = model(X)
        Y_pred = Y_pred.reshape(Y_pred.shape[0], Y_pred.shape[1])
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if isinstance(model, nn.DataParallel):
            weight = model.module.Featex.combinedConv.bayarConv2d.weight
            model.module.Featex.combinedConv.bayarConv2d.weight = nn.Parameter(bayarConstraint(weight))
        else:
            weight = model.Featex.combinedConv.bayarConv2d.weight
            model.Featex.combinedConv.bayarConv2d.weight = nn.Parameter(bayarConstraint(weight))
        torch.cuda.empty_cache()
        # loss_epochs += loss
        # n_batch += 1
        acc = evalute_acc(Y_pred, Y)
        logger.info(
            'Epoch:[{}/{}] batch_idx:{}\t ==TRAIN== loss={:.5f}\t acc={:.5f}'.format(0, MAX_EPOCH, 0, loss, acc))
        # if n_batch > batches:
        #     break
        # torch.cuda.empty_cache()
        acc = 0
        model = model.eval()
        with torch.no_grad():
            for X, Y in val_iter:
                X = X.to(device)
                Y = Y.to(device)
                Y_pred = model(X)
                Y_pred = Y_pred.reshape(Y_pred.shape[0], Y_pred.shape[1])
                loss = criterion(Y_pred, Y)
                acc += evalute_acc(Y_pred, Y)
                break
        logger.info(
            'Epoch:[{}/{}]\t ==TEST== loss={:.5f}\t acc={:.3f}'.format(0, MAX_EPOCH, loss,
                                                              acc))
        # torch.save(model.getFeatex(),
        #            os.path.join(save_dir, args.name, ('model_%d.pth' % (epoch))))
