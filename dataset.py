from glob import glob
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# load raw images
class CLEAR10IMG(Dataset):
    """ Learning CLEAR10 :) """

    def __init__(self, root_dir, bucket, form="all", split_ratio=0.7, debug=False, transform=None):
        '''
        Args: 
            root_dir(str list): folder path of 11 images
            bucket(int): time bucket id
            form(str): all -> whole dataset; train -> train dataset; test -> test dataset
            split_ratio(float, optional): proportion of train images in dataset
            transform(optional): transformation
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.bucket = bucket
        self.form = form
        self.input_folders = self.root_dir+"/"+str(bucket)
        self.img_paths = list(filter(lambda x: x.endswith(".jpg"), glob(self.input_folders + '/**',recursive=True)))
        
        # code classes by alphabetical order
        self.targets = [self.img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.img_paths))]
        classes_name = sorted(list(set(self.targets)))
        classes_code = range(len(classes_name))
        self.classes_mapping = dict(zip(classes_name,classes_code))
        self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
        
        if debug == True:
            self.img_paths = self.img_paths[:25]
            self.targets = self.targets[:25]
        if form != "all":
            self.train_img_paths = set(random.sample(self.img_paths,int(len(self.img_paths)*split_ratio)))
            self.test_img_paths = list(set(self.img_paths) - self.train_img_paths) 
            self.train_img_paths = list(self.train_img_paths)
            if form == "train":
                self.targets = [self.train_img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.train_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
            else:
                self.targets = [self.test_img_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.test_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()

    def __len__(self): 
        if self.form == "all":
            return len(self.img_paths)
        elif self.form == "train":
            return len(self.train_img_paths)
        else:
            return len(self.test_img_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            img = Image.open(self.img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.img_paths[idx][len(self.input_folders):].split("/")[1] # exclude the first empty entry
        elif self.form == "train":
            img = Image.open(self.train_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.train_img_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            img = Image.open(self.test_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.test_img_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'img': img, 'target': self.classes_mapping[label]}
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample['img'], sample['target']

class CLEAR10MOCO(Dataset):
    def __init__(self, root_dir, bucket, device, form="all", split_ratio=0.7, debug=False): # transform unnecessary
        self.root_dir = root_dir
        self.bucket = bucket
        self.form = form
        self.device = device
        self.input_folders = self.root_dir+"/bucket_"+str(bucket)
        self.tensor_paths = list(filter(lambda x: x.endswith(".pth"), glob(self.input_folders + '/**',recursive=True)))
        self.targets = [self.tensor_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.tensor_paths))]
        classes_name = sorted(list(set(self.targets)))
        classes_code = range(len(classes_name))
        self.classes_mapping = dict(zip(classes_name,classes_code))
        self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
        
        if debug == True:
            self.tensor_paths = self.tensor_paths[:25]
            self.targets = self.targets[:25]
        if form != "all":
            self.train_tensor_paths = set(random.sample(self.tensor_paths,int(len(self.tensor_paths)*split_ratio)))
            self.test_tensor_paths = list(set(self.tensor_paths) - self.train_tensor_paths)
            self.train_tensor_paths = list(self.train_tensor_paths)
            if form == "train":
                self.targets = [self.train_tensor_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.train_tensor_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
            else:
                self.targets = [self.test_tensor_paths[idx][len(self.input_folders):].split("/")[1] for idx in range(len(self.test_tensor_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
    
    def __len__(self): 
        if self.form == "all":
            return len(self.tensor_paths)
        elif self.form == "train":
            return len(self.train_tensor_paths)
        else:
            return len(self.test_tensor_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            tensor = torch.load(self.tensor_paths[idx])
            label = self.tensor_paths[idx][len(self.input_folders):].split("/")[1]
        elif self.form == "train":
            tensor = torch.load(self.train_tensor_paths[idx])
            label = self.train_tensor_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            tensor = torch.load(self.test_tensor_paths[idx])
            label = self.test_tensor_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'feature': tensor, 'target': self.classes_mapping[label]}
        return sample['feature'], sample['target']


class SubDataset(Dataset):
    def __init__(self,data,valid_idx):
        self.data = torch.stack([data[i][0] for i in valid_idx]).float()
        self.targets = torch.Tensor([data[i][1] for i in valid_idx]).int()
        self.task_label = torch.Tensor([data[i][2] for i in valid_idx]).int()
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self,idx):
        return self.data[idx], self.targets[idx]