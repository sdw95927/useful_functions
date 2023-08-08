import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
import numpy as np
import math
import skimage
import skimage.transform

import torch
import torch.utils
import torch.utils.data

from utils_image import rgba2rgb, Crop, Pad

def normalize(image, inverse=False):
    if not inverse:
        image = image/255.
    else:
        image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image

def augmentor(image, label=None):
    """Do image shape and color augmentation.
    
    Args:
        image: [h, w, c], float
        label: [h, w], int
    """
    h = image.shape[0]
    w = image.shape[1]
    n_channels = image.shape[2]
    
    # Shape augmentation
    if np.random.uniform(0, 1) < 0.8:
        # Shift for rotation
        tf_shift = skimage.transform.SimilarityTransform(translation=[-w/2, -h/2])
        
        # Projective transform
        scale = np.random.uniform(0.7, 1.5)
        aspect_ratio = np.random.uniform(0.9, 1.1)
        rotation = np.random.uniform(3, 3)
        translationX = np.random.uniform(-10, 10)
        translationY = np.random.uniform(-10, 10)
        g = np.random.uniform(-0.0001, 0.0001)
        h = np.random.uniform(-0.0001, 0.0001)
        matrix = np.array([[math.cos(rotation) * scale * aspect_ratio, -math.sin(rotation) * scale, translationX],
                          [math.sin(rotation) * scale, math.cos(rotation) * scale / aspect_ratio, translationY],
                          [g, h, 1]])
        # tform = skimage.transform.ProjectiveTransform(matrix=matrix)
        tform = skimage.transform.EuclideanTransform(matrix=matrix)
        
        # Shift back after rotation
        tf_shift_inv = skimage.transform.SimilarityTransform(translation=[w/2, h/2])
        
        # Combine above
        matrix = (tf_shift + (tform + tf_shift_inv)).inverse
        print(matrix)
        # tform = skimage.transform.ProjectiveTransform(matrix=matrix)
        tform = skimage.transform.EuclideanTransform(matrix=matrix)
        image_aug = np.zeros_like(image, dtype=float)
        for ch in range(n_channels):
            image_aug[..., ch] = skimage.transform.warp(image[..., ch], tform, preserve_range=True)
        
        image = image_aug
        
        if label is not None:
            label_aug = np.zeros_like(label, dtype=int)
            label_aug = skimage.transform.warp(label, tform, preserve_range=True, order=0)
            label = label_aug
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% vertical flipping
        image = image[::-1, :, :]
        
        if label is not None:
            label = label[::-1, :]
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% horizontal flipping
        image = image[:, ::-1, :]
        
        if label is not None:
            label = label[:, ::-1]

    # Color augmentation
    # 1) add a global shifting for all channels
    image = image + np.random.randn(1)[0] * 0.01

    # 2) add a shifting & variance for each channel
    for ch in range(n_channels):
        image[:, :, ch] = image[:, :, ch] * np.clip(np.random.randn(1)[0] * 0.01 + 1, 0.95, 1.05) + np.random.randn(1)[0] * 0.01
        
    return image, label

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, image_dict, opt,
                 augmentation=False):
        """
        Args:
            indexes: index used for image_dict
            indexes_A: IHC
            indexes_B: HE
        """
        self.indexes = indexes
        self.image_dict = image_dict
        self.augmentation = augmentation
        self.crop_size = opt['crop_size']
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            idx: indexes of samples (long)
        """
        # Generate indexes of the batch
        data_index = self.indexes[index]
        np.random.seed()
        
        # Generate data
        image, target, score = self.__data_generation(data_index, index)
        
        data = dict()
        data['image'] = image
        data['target'] = target
        data['score'] = score

        return data
    
    def __data_generation(self, index, idx=0):
        """Generates image containing batch_size samples.
        
        Returns:
            image: [ch, h, w]
            target: [h, w]
        """
        image = self.image_dict[index]['image']
        target = None
        
        # Random crop
        image, target = Crop(size=(self.crop_size, self.crop_size), pos='random')([image, target])
        image, target = Pad(size=(self.crop_size, self.crop_size))([image, target])
        
        # Augmentation
        if self.augmentation:
            image, target = augmentor(image/255., target)
        else:
            image = image/255.
        
        # Get label
        target = self.image_dict[index]['label']
        score = self.image_dict[index]['score']
        
        image = torch.tensor(np.transpose(image, (2, 0, 1)).astype(float))
        target = torch.tensor(target).long()
        score = torch.tensor(score).float()
        
        return image, target, score
            
def collate_fn(batch):
    return list(batch)


# Generator
train_set = Dataset(indexes, image_dict, augmentation=True, opt=opt)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, num_workers=1, 
                                          collate_fn=collate_fn)
for data in train_loader:
    plt.imshow(normalize(np.transpose(data[0]['image'].numpy(), (1, 2, 0)), inverse=True), interpolation='nearest', cmap='gray')
    plt.show()
    print(np.transpose(data[0]['image'].numpy(), (1, 2, 0)).shape)
    print(data[0]['target'].numpy())
    print(data[0]['score'].numpy())
    break
