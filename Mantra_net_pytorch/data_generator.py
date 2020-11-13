import numpy as np
from glob import glob
from numpy import random
from skimage import filters, util, transform, exposure, io
from matplotlib import pyplot as plt
import os
import argparse


def img_manipulation(img, mode=0):
    # img = np.array(img, dtype='int8')
    # add manipulation
    if mode == 0:  # gaussian噪声
        return util.random_noise(img, mode='gaussian', mean=0.5, var=0.3)
    elif mode == 1:  # salt噪声
        return util.random_noise(img, mode='salt')
    elif mode == 2:  # 直方图均衡化
        img = np.transpose(img, axes=[2, 0, 1])
        img1 = exposure.equalize_hist(img[0])
        img2 = exposure.equalize_hist(img[1])
        img3 = exposure.equalize_hist(img[2])
        img = np.stack([img1, img2, img3], axis=0)
        return np.transpose(img, [1, 2, 0])
    elif mode == 3:  # 改变对比度
        return exposure.rescale_intensity(1 - filters.sobel(img))
    elif mode == 4:  # 改变伽马值
        adjusted_gamma_image = exposure.adjust_gamma(
            img, gamma=0.4, gain=0.9)
        return adjusted_gamma_image
    elif mode == 5:  # 扭曲
        img = transform.swirl(
            img, center=[100, 100], rotation=0.3, strength=10, radius=120)
        return img
    elif mode == 6:  # 改变尺寸
        return transform.resize(img, (img.shape[0]*1.5, img.shape[1]*2))
    elif mode == 7:  # 反相
        return util.invert(img)


def convertOneImg2Patches(filename, patch_size=256, number=40):
    img = io.imread(filename)
    img_array = np.asarray(img)
    h, w = img_array.shape[:2]
    patches_list = []
    for _ in range(number):
        y = np.random.randint(0, h-patch_size+1)
        x = np.random.randint(0, w-patch_size+1)
        patch = img_array[y:y+patch_size, x:x+patch_size, :]
        patches_list.append(patch)
    return patches_list


def getAllPatches(filedir, number, patch_size):
    dir_list = glob(filedir + '/*')
    numberPerImg = number // (275 * len(dir_list)) + 1
    if numberPerImg == 0:
        numberPerImg = 1
    patches_list = []
    for dir in dir_list:
        print('start reading %s...' % (dir))
        for img in glob(dir+'/*'):
            X = convertOneImg2Patches(
                img, patch_size=patch_size, number=numberPerImg)
            patches_list.extend(X)
        print('finish reading %s...' % (dir))
        if len(patches_list) >= number:
            break
    patches_list = patches_list[:number]
    random.shuffle(patches_list)
    print('get all patches succeed!')
    return patches_list


def manipalate(imgs, mode):
    imgs_manipulated = []
    for img in imgs:
        imgs_manipulated.append(img_manipulation(img, mode))
    return imgs_manipulated


def gen_dataset(fileDir, saveDir, name='train', patch_size=256, number=10000):
    patches_list = getAllPatches(fileDir, number, patch_size)
    n = number // 8
    for mode in range(8):
        patches_m = manipalate(patches_list[mode*n: (mode+1)*n], mode)
        for idx, p in enumerate(patches_m):
            path = saveDir + '/' + name + ('/mode%d' % mode)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.imsave(path + ('/p%05d.jpg' % (idx+1)), p)


parser = argparse.ArgumentParser(description='dataset generator')
parser.add_argument('--fileDir', default='F:\code\MachineLearning\datasets\sp-society-camera-model-identification\\train\\train', type=str,
                    help='choose the dir of dataset')
parser.add_argument('--saveDir', default='./dataset',
                    type=str, help='choose the saveDir')
parser.add_argument('--name', default='val',
                    type=str, help='select train, val or test')
parser.add_argument('--patch_size', default=256,
                    type=int, help='set patch size')
parser.add_argument('--number', default=100,
                    type=int, help='set patches number')
args = parser.parse_args()

if __name__ == "__main__":
    gen_dataset(
        fileDir=args.fileDir,
        saveDir=args.saveDir,
        name=args.name,
        patch_size=args.patch_size,
        number=args.number)
