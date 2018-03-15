import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

from datasets.loader import get_loader

parser = argparse.ArgumentParser(description='4k dataset creator')
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--data-dir', type=str, default='./original_dataset',
                    help='directory which contains input data')
parser.add_argument('--dest-dir', type=str, default='./resized_dataset',
                    help='directory which will hold the 4k dataset')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size (default: 64)')
parser.add_argument('--w', type=int, default=4000,
                    help='width to upsample to (default: 4000)')
parser.add_argument('--h', type=int, default=4000,
                    help='height to upsample to (default: 4000)')
parser.add_argument('--mode', type=str, default='bilinear',
                    help='upsample mode (default: bilinear)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class ClassCounts(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.counts = [0 for _ in range(num_classes)]

    def __getitem__(self, index):
        cc = self.counts[index]
        self.counts[index] += 1
        return cc

def check_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def int_type(use_cuda):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor


def binarize(img):
    ''' simple threshold binarization'''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

def parallel_write_imgs(imgs, labels, classcounts, prefix):
    assert imgs.dim() == 4
    chans = imgs.size(1)
    xform = ToPILImage()

    for label, img in zip(labels, imgs):
        # determine pathing
        path = os.path.join(args.dest_dir, prefix, str(label))
        filename = os.path.join(path, '{}.png'.format(classcounts[label]))

        # binarize if need be
        img = binarize(img.data).type(int_type(args.cuda))*255 if chans == 1 else img.data

        if chans == 1:
            img = np.transpose(img, (1, 2, 0))
            xform(np.uint8(img.cpu().numpy())).save(filename)
        else:
            xform(img.cpu()).save(filename)


def loop(loader, args, cc, prefix='train'):
    for data, labels in loader:
        # push to GPU if requested
        data = Variable(data).cuda() if args.cuda else Variable(data)

        # upsample the data
        upsampled = F.upsample(data, size=(args.h, args.w), mode=args.mode)

        # write data to dest dir
        parallel_write_imgs(upsampled, labels, cc, prefix=prefix)
        del upsampled


def run(args):
    loader = get_loader(args, sequentially_merge_test=False)
    testcc = ClassCounts(loader.output_size)
    traincc = ClassCounts(loader.output_size)

    # create directory for use in torchvision.ImageFolder
    for i in range(loader.output_size):
        check_or_create_dir(os.path.join(args.dest_dir, 'train', str(i)))
        check_or_create_dir(os.path.join(args.dest_dir, 'test', str(i)))

    # write both test and train datasets
    loop(loader.train_loader, args, traincc, prefix='train')
    loop(loader.test_loader, args, testcc, prefix='test')


if __name__ == "__main__":
    run(args)
