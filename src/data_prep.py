import wget
import h5py
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import random
from typing import *
import zipfile


def download_data_h5():
    # The dataset we used belongs to Leaf Segmentation and Counting Challenge (LCC and LSC).
    # It is publicly available at https://www.plant-phenotyping.org/datasets-download or via the form:
    # https://docs.google.com/forms/d/1REQf0rQ7uvwgnnQxEyjapzqYFNPUHiVRW13XPH95cow/viewform#start=openform.
    # The data download links are saved to the dataset_urls.txt file.

    imgs_url, ground_truth_url = open("dataset_urls.txt").readlines()

    wget.download(imgs_url)
    wget.download(ground_truth_url)

    with zipfile.ZipFile('./CVPPP2017_training_images.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    with zipfile.ZipFile('./CVPPP2017_training_truth.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')

    print('Successfully downloaded data as h5 at the folder ./data/')


def h5_to_sep_lists():
    groups = ['A1', 'A2', 'A3']

    imgs_filename = 'data/CVPPP2017_training_images.h5'
    imgs = []
    with h5py.File(imgs_filename, "r") as f:
        for group in groups:
            for file_name in list(f[group]):
                imgs.append(np.swapaxes(np.array(f[group][file_name]['rgb'])[:, :, :3], 0, 2) / 255)

    ground_truth_filename = 'data/CVPPP2017_training_truth.h5'
    counts = []
    masks = []
    with h5py.File(ground_truth_filename, "r") as f:
        for group in groups:
            for file_name in list(f[group]):
                counts.append(np.array(f[group][file_name]['count']))
                masks.append(np.clip(np.array(f[group][file_name]['label']), 0, 1).transpose())

    # Shuffling data
    train_lst = list(zip(imgs, counts, masks))
    random.shuffle(train_lst)
    imgs, counts, masks = list(zip(*train_lst))

    print(f"Images: {len(imgs)}, Counts: {len(counts)}, Masks: {len(masks)}")

    return imgs, counts, masks


class LeafDataset(Dataset):
    def __init__(self, imgs, counts, masks, pos_transf=torch.Tensor, color_transf=None):
        super().__init__()
        self.images = imgs
        self.masks = masks
        self.counts = counts

        # Separating positional from coloring transformations.
        # This helps to match the position of the mask and the plant on image.
        self.pos_transforms = pos_transf

        self.color_transforms = color_transf

    def __len__(self):
        return len(self.images)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        # Positional augmentaion
        seed = random.randint(0, 2 ** 32)
        self._set_seed(seed)
        aug_img = self.pos_transforms(np.swapaxes(self.images[idx], 0, 2))
        self._set_seed(seed)
        aug_mask = self.pos_transforms(np.expand_dims(self.masks[idx], 2))

        # Coloring augmentaion
        if self.color_transforms:
            aug_img = self.color_transforms(aug_img)
        return aug_img, self.counts[idx], aug_mask


def get_loaders(lst_imgs: List, lst_masks: List, lst_counts: List, batch_size=4, train_ratio=0.8):

    dataset_len = len(lst_imgs)
    train_len = int(dataset_len * train_ratio)

    positional_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((480, 480)),
                                                transforms.RandomRotation(90),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip()
                                                ])

    coloring_transforms = transforms.Compose([transforms.RandomApply([transforms.RandomGrayscale(p=0.1)], p=0.5),
                                              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                                              ])

    train_dataset = LeafDataset(lst_imgs[:train_len], lst_masks[:train_len], lst_counts[:train_len], positional_transforms, coloring_transforms)
    test_dataset = LeafDataset(lst_imgs[train_len:], lst_masks[train_len:], lst_counts[train_len:], positional_transforms, coloring_transforms)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dl, test_dl


def show_dataset_example(dataloader):
    """
    Shows an example of augmented images with resulting masks.

    :param dataloader: iterable dataloader (only the first value will be retrieved)
    :return:
    """
    # Augmentation example
    aug_example_iter = iter(dataloader)

    aug_images, _, aug_masks = next(aug_example_iter)

    f, ax = plt.subplots(2, 4)

    for i in range(4):
        ax[0, i].imshow(aug_images[i].permute(2, 1, 0))
        ax[1, i].imshow(aug_masks[i].squeeze(0), cmap='gray')

    f.set_figheight(5)
    f.set_figwidth(15)
    plt.show()


def prepare_data():
    """
    Downloads data, shows how the images are augmented and returns training and testing dataloaders.

    :return: train and test dataloaders.
    """
    # download_data_h5()
    imgs, counts, masks = h5_to_sep_lists()
    train_dl, test_dl = get_loaders(imgs, counts, masks)
    show_dataset_example(train_dl)
    return train_dl, test_dl
