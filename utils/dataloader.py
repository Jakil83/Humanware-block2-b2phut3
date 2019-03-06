import os

from PIL import Image
from torch.utils import data

import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.transforms import FirstCrop, Rescale, RandomCrop, ToTensor
from utils.misc import load_obj
from utils.boxes import extract_labels_boxes
from sklearn.model_selection import StratifiedShuffleSplit


class SVHNDataset(data.Dataset):

    def __init__(self, metadata, data_dir, transform=None):
        '''
        Initialize the Dataset.

        Parameters
        ----------
        metadata : dict
        Each key in metadata will contain a filename and all associated
        metadata. The filename is with respect to the directory it's in,
        and metadata contains four 5 fields:
        * `label` - which lists all digits present in image, in order
        * `height`,`width`,`top`,`left` which list the corresponding pixel
        information about the digits bounding boxes.

        {0: {'filename': '1.png',
        'metadata': {'height': [219, 219],
        'label': [1, 9],
        'left': [246, 323],
        'top': [77, 81],
        'width': [81, 96]}},
        1: {'filename': '2.png',
        'metadata': {'height': [32, 32],
        'label': [2, 3],
        'left': [77, 98],
        'top': [29, 25],
        'width': [23, 26]}}, ...

        data_dir : str
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied on a sample.

        '''
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        '''
        Evaluate the length of the dataset object

        Returns
        -------
        int
            The length of the dataset.

        '''
        return len(self.metadata)

    def __getitem__(self, index):
        '''
        Get an indexed item from the dataset and return it.

        Parameters
        ----------
        index : int
            The index of the dataset

        Returns
        -------
        sample: dict
            sample['image'] contains the image array.
            The type may be a torch.tensor or ndarray depending on transforms
            sample['metadata'] will contain the metadata associated to the
            image. It can be one of ['labels','boxes','filename']


        '''
        'Generates one sample of data'

        img_name = os.path.join(self.data_dir,
                                self.metadata[index]['filename'])

        # Load data and get raw metadata (labels & boxes)
        image = Image.open(img_name)
        metadata_raw = self.metadata[index]['metadata']

        labels, boxes = extract_labels_boxes(metadata_raw)

        metadata = {'labels': labels, 'boxes': boxes, 'filename': img_name}

        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FullSVHNDataset(data.Dataset):

    def __init__(self, metadata_train, metadata_extra, train_data_dir,
                 extra_data_dir, transform=None):
        '''
        Initialize the Dataset.

        Parameters
        ----------
        metadata : dict
        Each key in metadata will contain a filename and all associated
        metadata. The filename is with respect to the directory it's in,
        and metadata contains four 5 fields:
        * `label` - which lists all digits present in image, in order
        * `height`,`width`,`top`,`left` which list the corresponding pixel
        information about the digits bounding boxes.

        {0: {'filename': '1.png',
        'metadata': {'height': [219, 219],
        'label': [1, 9],
        'left': [246, 323],
        'top': [77, 81],
        'width': [81, 96]}},
        1: {'filename': '2.png',
        'metadata': {'height': [32, 32],
        'label': [2, 3],
        'left': [77, 98],
        'top': [29, 25],
        'width': [23, 26]}}, ...

        data_dir : str
            Directory with all the images.
        transform : callable, optional
            Optional transform to be applied on a sample.

        '''
        self.metadata_train = metadata_train
        self.metadata_extra = metadata_extra
        self.train_data_dir = train_data_dir
        self.extra_data_dir = extra_data_dir
        self.transform = transform

    def __len__(self):
        '''
        Evaluate the length of the dataset object

        Returns
        -------
        int
            The length of the dataset.

        '''
        return len(self.metadata_train) + len(self.metadata_extra)

    def __getitem__(self, index):
        '''
        Get an indexed item from the dataset and return it.

        Parameters
        ----------
        index : int
            The index of the dataset

        Returns
        -------
        sample: dict
            sample['image'] contains the image array.
            The type may be a torch.tensor or ndarray depending on transforms
            sample['metadata'] will contain the metadata associated to the
            image. It can be one of ['labels','boxes','filename']


        '''
        'Generates one sample of data'

        if index < len(self.metadata_train):
            data_dir = self.train_data_dir
            metadata = self.metadata_train
        else:
            index -= len(self.metadata_train)
            data_dir = self.extra_data_dir
            metadata = self.metadata_extra


        img_name = os.path.join(data_dir,
                                metadata[index]['filename'])

        # Load data and get raw metadata (labels & boxes)
        image = Image.open(img_name)
        metadata_raw = metadata[index]['metadata']

        labels, boxes = extract_labels_boxes(metadata_raw)

        metadata = {'labels': labels, 'boxes': boxes, 'filename': img_name}

        sample = {'image': image, 'metadata': metadata}

        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_dataloaders(dataset_split,
                        dataset_path,
                        metadata_filename,
                        batch_size=32,
                        sample_size=-1,
                        valid_split=0.8,
                        stratified=False):
    '''
    Utility function to prepare dataloaders for training.

    Parameters
    ----------
    dataset_split : str
        Any of 'train', 'extra', 'test'.
    dataset_path : str
        Absolute path to the dataset. (i.e. .../data/SVHN/train')
    metadata_filename : str
        Absolute path to the metadata pickle file.
    batch_size : int
        Mini-batch size.
    sample_size : int
        Number of elements to use as sample size,
        for debugging purposes only. If -1, use all samples.
    valid_split : float
        Returns a validation split of %size; valid_split*100,
        valid_split should be in range [0,1].
    stratified : Whether we use sklearn startified splits or not.

    Returns
    -------
    if dataset_split in ['train', 'extra']:
        train_loader: torch.utils.DataLoader
            Dataloader containing training data.
        valid_loader: torch.utils.DataLoader
            Dataloader containing validation data.

    if dataset_split in ['test']:
        test_loader: torch.utils.DataLoader
            Dataloader containing test data.


    '''

    assert dataset_split in ['train', 'test', 'extra', 'train+extra'], "check dataset_split"



    #  dataset_path = datadir / dataset_split

    firstcrop = FirstCrop(0.3)
    rescale = Rescale((64, 64))
    random_crop = RandomCrop((54, 54))
    to_tensor = ToTensor()

    # Declare transformations

    transform = transforms.Compose([firstcrop,
                                    rescale,
                                    random_crop,
                                    to_tensor,
                                    ])

    if dataset_split in ['train', 'extra']:

        metadata = load_obj(metadata_filename[0])

        dataset = SVHNDataset(metadata,
                              data_dir=dataset_path[0],
                              transform=transform)

        indices = np.arange(len(metadata))
        #  indices = np.random.permutation(indices)

        # Only use a sample amount of data
        if sample_size != -1:
            indices = indices[:sample_size]

        train_idx = indices[:round(valid_split * len(indices))]
        valid_idx = indices[round(valid_split * len(indices)):]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        # Prepare a train and validation dataloader
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=train_sampler)

        valid_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=valid_sampler)

        return train_loader, valid_loader

    elif dataset_split in ['train+extra']:
        # 0 for train, 1 for extra
        metadata_train = load_obj(metadata_filename[0])
        metadata_extra = load_obj(metadata_filename[1])

        train_data_dir = dataset_path[0]
        extra_data_dir = dataset_path[1]

        valid_split_train = valid_split[0]
        valid_split_extra = valid_split[1]

        # Determine class (number of digits) for each image

        y_train = []
        y_extra = []

        for key, value in metadata_train.items():
            y_train.append(min(len(value['metadata']['label']), 5))

        for key, value in metadata_extra.items():
            y_extra.append(min(len(value['metadata']['label']), 5))

        dataset = FullSVHNDataset(metadata_train, metadata_extra, train_data_dir,
                                  extra_data_dir, transform=transform)

        indices_train = np.arange(len(metadata_train))
        indices_extra = np.arange(len(metadata_train), len(metadata_extra) + len(metadata_train))

        # Only use a sample amount of data
        if sample_size[0] != -1:
            indices_train = indices_train[:sample_size[0]]
            y_train = y_train[:sample_size[0]]

        if sample_size[1] != -1:
            indices_extra = indices_extra[:sample_size[1]]
            y_extra = y_extra[:sample_size[1]]

        if stratified: # If we want balanced splits

            train_splitter = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=valid_split_train)

            for train_idx, valid_idx in train_splitter.split(indices_train, y_train):
                train_idx_train = indices_train[train_idx]
                valid_idx_train = indices_train[valid_idx]

            extra_splitter = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=valid_split_extra)

            for train_idx, valid_idx in extra_splitter.split(indices_extra, y_extra):
                train_idx_extra = indices_extra[train_idx]
                valid_idx_extra = indices_extra[valid_idx]

        else:

            train_idx_train = indices_train[:round(valid_split_train * len(indices_train))]
            valid_idx_train = indices_train[round(valid_split_train * len(indices_train)):]

            train_idx_extra = indices_extra[:round(valid_split_extra * len(indices_extra))]
            valid_idx_extra = indices_extra[round(valid_split_extra * len(indices_extra)):]

        train_idx = np.concatenate((train_idx_train, train_idx_extra))
        valid_idx = np.concatenate((valid_idx_train, valid_idx_extra))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        # Prepare a train and validation dataloader
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=train_sampler)

        valid_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  sampler=valid_sampler)

        return train_loader, valid_loader


    elif dataset_split in ['test']:

        metadata = load_obj(metadata_filename[0])

        dataset = SVHNDataset(metadata,
                              data_dir=dataset_path[0],
                              transform=transform)

        indices = np.arange(len(metadata))
        #  indices = np.random.permutation(indices)

        # Only use a sample amount of data
        if sample_size != -1:
            indices = indices[:sample_size]

        test_sampler = torch.utils.data.SequentialSampler(indices)

        # Prepare a test dataloader
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 shuffle=False,
                                 sampler=test_sampler)

        return test_loader
