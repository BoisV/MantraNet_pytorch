import torch
from torch import batch_norm, nn, set_flush_denormal
from torch.nn import functional as F
from collections import OrderedDict

from torch.nn.modules import module
from convlstm import ConvLSTM


class SRMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        # X1 =
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)


# def BayarConstraint(params):


class CombinedConv2D(nn.Module):
    def __init__(self, in_channels=3):
        super(CombinedConv2D, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=10, stride=1, kernel_size=5, padding=2)
        self.bayarConv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=3, stride=1, kernel_size=5, padding=2)
        self.SRMConv2d = SRMConv2D(
            in_channels=3, out_channels=3, stride=1, padding=2)

    def forward(self, X):
        X1 = F.relu(self.conv2d(X))
        X2 = F.relu(self.bayarConv2d(X))
        X3 = F.relu(self.SRMConv2d(X))
        return torch.cat([X1, X2, X3], dim=1)


class FeatexVGG16(nn.Module):
    def __init__(self, type=1):
        super(FeatexVGG16, self).__init__()
        # block1
        self.combinedConv = CombinedConv2D(in_channels=3)
        self.block1 = nn.Sequential(OrderedDict([
            ('b1c1', nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)),
            ('b1ac', nn.ReLU())
        ]))

        # block2
        self.block2 = nn.Sequential(OrderedDict([
            ('b2c1', nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('b2ac1', nn.ReLU()),
            ('b2c2', nn.Conv2d(in_channels=64,
                               out_channels=64, kernel_size=3, stride=1, padding=1)),
            ('b2ac2', nn.ReLU())
        ]))

        # block3
        self.block3 = nn.Sequential(OrderedDict([
            ('b3c1', nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('b3ac1', nn.ReLU()),
            ('b3c2', nn.Conv2d(in_channels=128, kernel_size=3,
                               out_channels=128, stride=1, padding=1)),
            ('b3ac2', nn.ReLU()),
            ('b3c3', nn.Conv2d(in_channels=128,
                               out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('b3ac3', nn.ReLU())
        ]))

        # block4
        self.block4 = nn.Sequential(OrderedDict([
            ('b4c1', nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('b4ac1', nn.ReLU()),
            ('b4c2', nn.Conv2d(in_channels=256,
                               out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('b4ac2', nn.ReLU()),
            ('b4c3', nn.Conv2d(in_channels=256,
                               out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('b4ac3', nn.ReLU())
        ]))

        # block5
        self.block5 = nn.Sequential(OrderedDict([
            ('b5c1', nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('b5ac1', nn.ReLU()),
            ('b5c2', nn.Conv2d(in_channels=256,
                               out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('b5ac2', nn.ReLU())
        ]))

        self.transform = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.activation = None if type >= 1 else nn.Tanh()

    def forward(self, X):
        X= self.combinedConv(X)
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        X = self.block5(X)
        X = self.transform(X)
        if self.activation is not None:
            X = self.activation(X)
        return nn.functional.normalize(X, 2, dim=-1)


class ZPool2D(nn.Module):
    def __init__(self, kernel_size):
        super(ZPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, X):
        mu = self.avgpool(X)
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
                            ).sum() / (X.shape[-2] * X.shape[-1]))
        D = X - mu
        return D / sigma


class ZPool2DGlobal(nn.Module):
    def __init__(self, size=[1, 64, 1, 1], epsilon=1e-5):
        super(ZPool2DGlobal, self).__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.zeros(size), requires_grad=True)

    def forward(self, X):
        mu = torch.mean(X, dim=(2, 3), keepdim=True)
        D = X - mu
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
                            ).sum(dim=(-1, -2), keepdim=True) / (X.shape[-2] * X.shape[-1]))
        sigma = torch.max(sigma, self.epsilon + self.weight)
        return D / sigma


class MantraNet(nn.Module):
    def __init__(self, Featex=FeatexVGG16(), pool_size_list=[7, 15, 31]):
        super(MantraNet, self).__init__()
        self.rf = Featex
        self.outlierTrans = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=(1, 1), bias=False)
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.zpoolglobal = ZPool2DGlobal()
        zpools = OrderedDict()
        for i in pool_size_list:
            name = 'ZPool2D@{}x{}'.format(i, i)
            zpools[name] = ZPool2D(i)
        self.zpools = nn.Sequential(zpools)
        self.cLSTM = ConvLSTM(64, 8, (7, 7), 1, batch_first=True)
        self.pred = nn.Conv2d(in_channels=8, out_channels=1,
                              kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        if self.rf is not None:
            X = self.rf(X)
        X = self.bnorm(self.outlierTrans(X))
        Z = []
        Z.append(torch.unsqueeze(self.zpoolglobal(X), dim=1))
        for index in range(len(self.zpools)-1, -1, -1):
            Z.append(torch.unsqueeze(self.zpools[index](X), dim=1))
        Z = torch.cat([i for i in Z], dim=1)
        last_output_list, _ = self.cLSTM(Z)
        X = last_output_list[0][:, -1, :, :, :]
        output = self.sigmoid(self.pred(X))
        return output


class IMTFE(nn.Module):
    def __init__(self, Featex, in_size) -> None:
        super(IMTFE, self).__init__()
        self.Featex = Featex
        self.conv1 = nn.Conv2d(
            in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8,
                               kernel_size=in_size, stride=1, padding=0)

    def forward(self, input):
        out = self.Featex(input)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

    def getFeatex(self):
        return self.Featex


def bayarConstraint(weight_):
    weight = weight_[0]
    h, w = weight.shape[1: 3]
    weight[:, h//2, w//2] = 0
    weight /= weight.sum(dim=(1, 2), keepdim=True)
    weight[:, h//2, w//2] = -1
    return weight_


# X = torch.randn([2, 3, 128, 128])
# net = FeatexVGG16()
# net = IMTFE(Featex=FeatexVGG16(), in_size=128)
# # net = MantraNet(FeatexVGG16())
# Y = net(X)
# print(Y.shape)
# net = CombinedConv2D()
# weight = net.bayarConv2d.weight[0]
# net.bayarConv2d.weight[0] = bayarConstraint(weight)
# print(net.bayarConv2d.weight[0,:, 3,3])
# model = MantraNet(FeatexVGG16())
# X = torch.rand([1,3,128,128])
# out = model(X)
# print(out.shape)