from glob import glob
import random
from skimage import io
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
        self.img_paths = list(filter(lambda x: x.endswith(".jpg"), glob(self.input_folders,recursive=True)))
        if debug == True:
            self.img_paths = self.img_paths[:25]
        if form != "all":
            self.train_img_paths = set(random.sample(self.img_paths,int(len(self.img_paths)*split_ratio)))
            self.test_img_paths = self.img_paths - self.train_img_paths 

    def __len__(self): 
        if self.form == "all":
            return len(self.img_paths)
        elif self.form == "train":
            return len(self.train_img_paths)
        else:
            return len(self.test_img_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            img = io.imread(self.img_paths[idx])
            label = self.img_paths[idx][len(self.input_folders):].split("/")[1] # exclude the first empty entry
        elif self.form == "train":
            img = io.imread(self.train_img_paths[idx])
            label = self.train_img_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            img = io.imread(self.test_img_paths[idx])
            label = self.test_img_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'img': img, 'label': label}
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample

class CLEAR10MOCO(Dataset):
    def __init__(self, root_dir, bucket, device, form="all", split_ratio=0.7, debug=False): # transform unnecessary
        self.root_dir = root_dir
        self.bucket = bucket
        self.form = form
        self.device = device
        self.input_folders = self.root_dir+"/bucket_"+str(bucket)
        self.tensor_paths = list(filter(lambda x: x.endswith(".pth"), glob(self.input_folders,recursive=True)))
        if debug == True:
            self.tensor_paths = self.tensor_paths[:25]
        if form != "all":
            self.train_tensor_paths = set(random.sample(self.img_paths,int(len(self.img_paths)*split_ratio)))
            self.test_tensor_paths = self.tensor_paths - self.train_tensor_paths
    
    def __len__(self): 
        if self.form == "all":
            return len(self.tensor_paths)
        elif self.form == "train":
            return len(self.train_tensor_paths)
        else:
            return len(self.test_tensor_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            tensor = torch.load(self.tensor_paths[idx], map_location=self.device)
            label = self.img_paths[idx][len(self.input_folders):].split("/")[1]
        elif self.form == "train":
            tensor = torch.load(self.train_tensor_paths[idx], map_location=self.device)
            label = self.train_tensor_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            tensor = torch.load(self.test_tensor_paths[idx], map_loaction=self.device)
            label = self.test_tensor_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'feature': tensor, 'label': label}
        return sample
