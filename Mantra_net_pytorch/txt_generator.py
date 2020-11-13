import os
import argparse

parser = argparse.ArgumentParser(description='label Generator')
parser.add_argument('--saveDir', default='./dataset',
                    type=str, help='choose the saveDir')
parser.add_argument('--name', default='val',
                    type=str, help='select train, val or test')
args = parser.parse_args()

def genTxt(saveDir, name='train'):
    dirname = os.path.join(saveDir, name)
    if not os.path.exists(dirname):
        assert('not found: ' + dirname)
    subdirs = os.listdir(dirname)
    filename = os.path.join(dirname, name+'.txt')
    if os.path.exists(filename):
        os.remove(filename)
    for idx, dir in enumerate(subdirs):
        if os.path.isfile(os.path.join(dirname,dir)):
            continue
        print(dir)
        string = ''
        patch_name_list = os.listdir(os.path.join(dirname, dir))
        for name in patch_name_list:
            string = string + '/' + dir + '/' + name + ' ' + dir[4] + '\n'
        with open(filename, mode='a+') as f:
            f.write(string)
            f.close()

if __name__ == "__main__":
    genTxt(saveDir=args.saveDir, name=args.name)
