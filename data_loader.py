import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp

from augmentations import *


class CelebAMaskHQ(Dataset):
    """The data reading method modified from the original paper"""

    def __init__(self, img_path, label_path, resize=512, trainsform=None, mode=True):
        self.img_path = img_path
        self.label_path = label_path
        self.train_dataset = []
        self.test_dataset = []
        self.trainsform = trainsform
        self.mode = mode
        self.resize = resize
        self.preprocess()

        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.img_path) if osp.isfile(osp.join(self.img_path, name))])):
            img_path = osp.join(self.img_path, str(i)+'.jpg')
            label_path = osp.join(self.label_path, str(i)+'.png')

            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]

        # Uniform image size
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # Image resized to the same dimension
        image, label = Compose(
            [FreeScale(self.resize)])(image, label)

        if self.mode == True:
            if self.trainsform is not None:
                image, label = self.trainsform(image, label)

            # Convert it to pytorch style
            image = img_transform(image)
            mask = mask_transform(label)
            edge = mask_transform(Image.fromarray(edge_contour(np.asarray(label))))

            return image, mask, edge

        else:
            image = img_transform(image)
            mask = mask_transform(label)

            return image, mask

    def __len__(self):
        return self.num_images


class CustomDataLoader:
    def __init__(self, img_path, label_path, image_size, batch_size, transform=None, mode=True):
        self.img_path = img_path
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        self.transform = transform

    def loader(self):
        dataset = CelebAMaskHQ(
            self.img_path, self.label_path, self.imsize, self.transform, self.mode)

        if self.mode == True:
            loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self.batch,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 drop_last=True,
                                                 pin_memory=True)
            return loader

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=False,
                                             num_workers=4,
                                             drop_last=True,
                                             pin_memory=True)
        return loader
