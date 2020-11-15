import os
import re
import torch
import glob
import logging
from torch.utils.data.dataloader import DataLoader
from network import MantraNet, FeatexVGG16, IMTFE, bayarConstraint
from torch import device, mode, optim, nn, cuda
import argparse
from torchvision.transforms import transforms
from dataset import MyDataset

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
            result = re.findall(".*model_(.*).pth.*" , file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def evalute_acc(Y_pred, Y):
    return (Y_pred.argmax(dim=1) == Y).to(torch.float32).mean()

if __name__ == "__main__":
    
    model = IMTFE(Featex=FeatexVGG16(), in_size=128)
    initial_epoch = findLastCheckpoint(
        save_dir=args.saveDir, name='IMTFE')  # IMTFE or MantraNet
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(
            args.saveDir, 'model_%03d.pth' % initial_epoch))

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
    model = model.cuda()
    model = nn.DataParallel(model)

    batch_size = 64
    batches = 1000
    MAX_EPOCH = 100

    transform = transforms.Compose([transforms.RandomCrop(size=(128,128)),
                                    transforms.ToTensor()])
    dataset_train = MyDataset(root_dir=os.path.join(args.dataDir, 'train'), names_file=os.path.join(args.dataDir, 'train', 'train.txt'), transform=transform)
    dataset_val = MyDataset(root_dir=os.path.join(args.dataDir, 'val'), names_file=os.path.join(args.dataDir, 'val', 'val.txt'), transform=transform)
    train_iter = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)
    val_iter = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1)
    logger.info('start training!')
    for epoch in range(MAX_EPOCH):
        loss_epochs = 0
        n_batch = 0
        model = model.train()
        for X, Y in train_iter:
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            Y_pred = model(X)
            Y_pred = Y_pred.reshape(Y_pred.shape[0], Y_pred.shape[1])
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if isinstance(model, nn.DataParallel):
                weight = model.module.Featex.combinedConv.bayarConv2d.weight
                model.module.Featex.combinedConv.bayarConv2d.weight = nn.Parameter(bayarConstraint(weight))
            else:
                weight = model.Featex.combinedConv.bayarConv2d.weight
                model.Featex.combinedConv.bayarConv2d.weight = nn.Parameter(bayarConstraint(weight))
            torch.cuda.empty_cache()
            loss_epochs += loss
            n_batch += 1
            acc = evalute_acc(Y_pred, Y)
            logger.info(
                'Epoch:[{}/{}] batch_idx:{} lr:{:.5f}\t loss={:.5f}\t acc={:.5f}'.format(epoch, MAX_EPOCH, scheduler.get_lr()[0], n_batch, loss, acc))
            if n_batch > batches:
                break
        torch.cuda.empty_cache()
        acc = 0
        model = model.eval()
        for X, Y in val_iter:
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            Y_pred = model(X)
            acc += evalute_acc(Y_pred, Y)
        logger.info(
                'Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, MAX_EPOCH, loss_epochs/MAX_EPOCH, acc/len(dataset_val)))
        if isinstance(model, nn.DataParallel):
            state = model.module.Featex.state_dict()
        else:
            state = model.Featex.state_dict()
        torch.save(state,
                   os.path.join(args.saveDir, args.name, ('model_%d.pth'%(epoch))))
