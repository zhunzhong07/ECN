from __future__ import absolute_import
import os.path as osp
from PIL import Image
from torchvision.transforms import functional as F
import torch


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class UnsupervisedCamStylePreprocessor(object):
    def __init__(self, dataset, root=None, camstyle_root=None, num_cam=6, transform=None):
        super(UnsupervisedCamStylePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.num_cam = num_cam
        self.camstyle_root = camstyle_root

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        sel_cam = torch.randperm(self.num_cam)[0]
        if sel_cam == camid:
            fpath = osp.join(self.root, fname)
            img = Image.open(fpath).convert('RGB')
        else:
            if 'msmt' in self.root:
                fname = fname[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
            else:
                fname = fname[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
            fpath = osp.join(self.camstyle_root, fname)
            img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, index

