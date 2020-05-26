from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

import sys


class Dataset(metaclass=ABCMeta):
    
    def compute_stats(self, train_data):
        """ 
        Returns the mean and std from input data in [N,C,W,H].
        """
        mean_mat = torch.stack([torch.mean(t, dim=(1,2)) for t, c in train_data])
        mean = torch.mean(mean_mat, dim=0)
        
        std_mat = torch.stack([ torch.std(img, dim=(1,2)) for img, _ in train_data])
        std = torch.mean(std_mat, dim=0)
        
        return mean, std
    
    def init_limits(self, mean, std):
        """ 
        Returns the max, min and epsilon values adjusted for range of normalized images.
        """
        max_val = np.zeros_like(mean)
        min_val = np.zeros_like(mean)
        eps_size = np.zeros_like(mean)
        for i in range(len(mean)):
            max_val[i] = (1. - mean[i]) / std[i]
            min_val[i] = (0. - mean[i]) / std[i]
            eps_size[i] = abs( (1. - mean[i]) / std[i] ) + abs( (0. - mean[i]) / std[i] )
            
        return max_val, min_val, eps_size
    
    def scale(self):
        """ Scale to [0-1] on Test Set"""
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def crop_scale(self):
        """ Scale to [0-1] on Training Set"""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    

    def crop_normalize(self, mean, std):
        """ Per-Channel Zero-Mean Normalization on Train set. """
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def normalize(self, mean, std):
        """ Per-Channel Zero-Mean Normalization on Test set. """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    
    def inverse_normalize(self, mean, std):
        """ Undo Zero-Mean Normalization"""
        return transforms.Normalize( (- mean / std).tolist(), (1.0 / std ).tolist() )
    
    
    
class TORCH_CIFAR10(Dataset):
    
    def __init__(self, ds_path, crop=False, normalized=False):
        self.max_val = None
        self.min_val = None
        self.eps_size = None
        
        # Retrieve train data to compute mean and std
        train_data = CIFAR10(ds_path, train=True, transform=self.scale(), download=True)
        self.mean, self.std = self.compute_stats(train_data)
        
        print(">>> Dataset Mean:", self.mean.data)
        print(">>> Dataset Std:", self.std.data)
        
        # Retrieve agumented dataset
        self.train_data, self.test_data = self._get_datasets(ds_path, crop, normalized, self.mean, self.std)
        
        if normalized:
            self.max_val, self.min_val, self.eps_size = self.init_limits(self.mean, self.std)
            
            
            
    def _get_datasets(self, ds_path, crop, normalize, mean, std):
        
        train_data = None
        test_data = None
        
        if normalize:
            
            if crop:
                print(">>> CROPPING, ROTATING AND NORMALIZING IMAGES WITH ZERO-MEAN...")
                train_data = CIFAR10(ds_path, train=True, transform=self.crop_normalize(mean, std), download=True)
                
            else:
                print(">>> NORMALIZING IMAGES WITH ZERO-MEAN...")
                train_data = CIFAR10(ds_path, train=True, transform=self.normalize(mean, std), download=True)
                
            test_data = CIFAR10(ds_path, train=False, transform=self.normalize(mean, std), download=True)
        
        
        else:
            
            if crop:
                print(">>> CROPPING, ROTATING AND SCALING IMAGES [0-1]...")
                train_data = CIFAR10(ds_path, train=True, transform=self.crop_scale(), download=True)
                
            else:
                print(">>> SCALING IMAGES [0-1]...")
                train_data = CIFAR10(ds_path, train=True, transform=self.scale(), download=True)
            
            test_data = CIFAR10(ds_path, train=False, transform=self.scale(), download=True)
            
        return train_data, test_data
    
    
    def get_datasets(self):
        return (self.train_data, self.test_data)
    
    def get_consts(self):
        return (self.max_val, self.min_val, self.eps_size)
    
    
    
    
    
class TORCH_CIFAR100(Dataset):
    
    def __init__(self, ds_path, crop=False, normalized=False):
        self.max_val = None
        self.min_val = None
        self.eps_size = None
        
        # Retrieve train data to compute mean and std
        train_data = CIFAR100(ds_path, train=True, transform=self.scale(), download=True)
        self.mean, self.std = self.compute_stats(train_data)
        
        print(">>> Dataset Mean:", self.mean.data)
        print(">>> Dataset Std:", self.std.data)
        
        # Retrieve agumented dataset
        self.train_data, self.test_data = self._get_datasets(ds_path, crop, normalized, self.mean, self.std)
        
        if normalized:
            self.max_val, self.min_val, self.eps_size = self.init_limits(self.mean, self.std)
            
            
            
    def _get_datasets(self, ds_path, crop, normalize, mean, std):
        
        train_data = None
        test_data = None
        
        if normalize:
            
            if crop:
                print(">>> CROPPING, ROTATING AND NORMALIZING IMAGES WITH ZERO-MEAN...")
                train_data = CIFAR100(ds_path, train=True, transform=self.crop_normalize(mean, std), download=True)
                
            else:
                print(">>> NORMALIZING IMAGES WITH ZERO-MEAN...")
                train_data = CIFAR10(ds_path, train=True, transform=self.normalize(mean, std), download=True)
                
            test_data = CIFAR100(ds_path, train=False, transform=self.normalize(mean, std), download=True)
        
        
        else:
            
            if crop:
                print(">>> CROPPING, ROTATING AND SCALING IMAGES [0-1]...")
                train_data = CIFAR100(ds_path, train=True, transform=self.crop_scale(), download=True)
                
            else:
                print(">>> SCALING IMAGES [0-1]...")
                train_data = CIFAR100(ds_path, train=True, transform=self.scale(), download=True)
            
            test_data = CIFAR100(ds_path, train=False, transform=self.scale(), download=True)
            
        return train_data, test_data
    
    
    def get_datasets(self):
        return (self.train_data, self.test_data)
    
    def get_consts(self):
        return (self.max_val, self.min_val, self.eps_size)
    
    

def get_dataset(config):
    
    if config.NAME == 'CIFAR10':
        return TORCH_CIFAR10(config.DIR_PATH, config.CROP, config.NORMALIZE)
    
    elif config.NAME == 'CIFAR100':
        return TORCH_CIFAR100(config.DIR_PATH, config.CROP, config.NORMALIZE)
        
    
    else:
        print("ERROR: No dataset with name", config.NAME, "found. Exiting...")
        sys.exit(1)
        
    
if __name__ == "__main__":
    
    ds = TORCH_CIFAR10('datasets/', crop=True, normalized=True)
                
                
            
                
                