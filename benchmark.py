##################################################################
# Replacement for dataset.py following published CLEAR10 structure
##################################################################

from glob import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.models import resnet18
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.strategies import LwF, Replay, CWRStar, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import accuracy_metrics,timing_metrics
from avalanche.logging import InteractiveLogger


class CLEARIMG(Dataset):

    def __init__(self, root_dir, bucket, debug=False):
        '''
            Load raw images.

            Args: 
                root_dir(str): filelist folder path of all txt
                bucket(int): bucket id, in public released version, bucket id == time id
                transform(optional): transformation
            debug(optional): if debug == True, only randomly select 25 images for each bucket
        '''
        self.root_dir = root_dir
        self.bucket = bucket
        df = pd.read_csv(self.root_dir + "/filelists/" + str(bucket) + ".txt", delimiter=" ", names=["filename","label_id"])
        self.img_paths = list(df["filename"].apply(lambda x: root_dir + "/" +x))
        self.targets = torch.tensor(list(df["label_id"]),dtype=torch.int64) # which class
        tL = list(zip(self.img_paths,self.targets))
        tL_shuffled = random.sample(tL, k=len(tL))
        
        if debug == True:
            self.img_paths = [i for (i,j) in tL_shuffled][:25]
            self.targets = [j for (i,j) in tL_shuffled][:25]
        

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self,idx):
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        sample = {'img': img, 'target': self.targets[idx]}
        return sample['img'], sample['target']


class CLEARUNSUP(Dataset):

    def __init__(self, dict_dir, bucket, feature_type, label, label_id, debug=False): 
        '''
            Load extracted features with unsupervised models.

            Args: 
                dict_dir(str): directory of .pth file
                bucket(int): bucket id, in public released version, bucket id == time id
                feature_type(str): any of byol_imagenet/imagenet/moco_b0/moco_imagenet
                label(str): class label of input feature
                label_id(int): label id for label
                debug(optional): if debug == True, only randomly select 4 images for each label in one bucket
        '''
        self.dict_dir = dict_dir
        self.all_features = torch.load(dict_dir+f"/features/{feature_type}/{bucket}/{label}.pth")
        self.img_lst = list(self.all_features.keys())
        self.feature_lst = list(self.all_features.values())
        self.bucket = bucket
        self.targets = torch.tensor([label_id] * len(self.feature_lst),dtype=torch.int64)
        if debug == True:
            self.img_lst = self.img_lst[:4]
            self.feature_lst = self.feature_lst[:4]
            self.targets = self.targets[:4]
        
    def __len__(self):
        return len(self.feature_lst)

    def __getitem__(self,idx):
        sample = {'feature': self.feature_lst[idx], 'target': self.targets[idx]}
        return sample['feature'], sample['target']

class CLEARConcat(ConcatDataset):
    '''
        Insert targets as attributes for creating AvalancheDataset object.
    '''
    def __init__(self, datasets):
        super().__init__(datasets)
        self.targets = torch.cat([datasets[i].targets for i in range(len(datasets))]) 

class CLEARSubset(Subset):
    '''
        Reindex Subset indices starting from 0.
    '''
    def __init__(self, datasets, indices):
        super().__init__(datasets, indices)
        self.indices = range(super().__len__()) # reindexing

def mkCLEAR(root_dir, return_task_id=True, mode="online", split_ratio=0.7, train_transform=None, \
            eval_transform=None, pretrained=False, feature_type=None, debug=False):
    '''
        Make a CLEAR benchmark object. Multiple input root directories are supported.

        Args: 
            root_dir(str or list): if str, root directory
                                   if list, a list with the first index as train, the second index as test.
            return_task_id: if True, task id is assigned by time bucket id; else, all task id is 0.
            mode(str, optional): online -> whole dataset; offline -> train/test split dataset
            split_ratio(float, optional): proportion of train images in dataset
            train_transform(optional): transform for train datasets
            eval_transform(optional): transform for test datasets
            pretrained(bool, optional): if False, load raw image; else, load pretrained features
            feature_type(str, optional): when pretrained == True, feature_type can be any of 
                                         byol_imagenet/imagenet/moco_b0/moco_imagenet
            debug(bool, optional): if debug == True, when pretrained = False, 
                                   only randomly select 25 images for each bucket (25 full, 25 train, 25 test),
                                   when pretrained = True, 
                                   randomly select 4 images for each label in a bucket
        '''
    if mode != "online" and mode != "offline":
        raise ValueError("Invalid Mode. Please choose from online or offline.")
    if (split_ratio > 1 or split_ratio < 0) and isinstance(root_dir, str):
        raise ValueError("split_ratio must be from 0 to 1 when you only provide one root directory.")
    if pretrained == True and (feature_type not in ["byol_imagenet", "imagenet", "moco_b0","moco_imagenet"]):
        raise ValueError("Invalid pretrained features. \
                          Please choose from byol_imagenet, imagenet, moco_b0, moco_imagenet.")
    if mode == "online":
        num_buckets = len(glob(root_dir+"/filelists/**.txt"))
        if pretrained == False:
            # i is already sorted by time, smaller values older buckets
            data_lst = [(i+1,CLEARIMG(root_dir, i+1, debug=debug)) for i in range(num_buckets)]
        else:
            # sorted by alphabetical order
            all_classes = sorted(set(map(lambda x: x.split("/")[-1][:-4], glob(root_dir+f"/features/{feature_type}/*/**.pth", recursive=True)))) 
            num_classes = len(all_classes)
            id2class = dict(list(enumerate(all_classes)))
            print("Class-to-id mapping:", id2class)
            data_lst =  [(i+1,CLEARConcat([CLEARUNSUP(root_dir, i+1, feature_type, id2class[j], j, debug=debug) \
                         for j in range(num_classes)])) for i in range(num_buckets)]
        print(f"Loaded full data from {root_dir}")
        if return_task_id:
            aval_data_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), data_lst))
        else:
            aval_data_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=0), data_lst))
        
        return dataset_benchmark(
        aval_data_lst[:-1],
        aval_data_lst[1:],
        train_transform=train_transform,
        eval_transform=eval_transform,
        ) 

    else: # offline
        if isinstance(root_dir, list):
            if len(root_dir) != 2:
                raise ValueError("You must provide two directories in your list.")
            train_num_buckets = len(glob(root_dir[0]+"/filelists/**.txt"))
            test_num_buckets = len(glob(root_dir[1]+"/filelists/**.txt"))
            test_num_buckets -= 1 # TODO: delete later. Extra 0.txt in test folder
            '''
            if train_num_buckets != test_num_buckets:
                raise ValueError("Number of buckets in training dataset is different from the number of buckets in testing dataset.")
            '''
            num_buckets = train_num_buckets
            if pretrained == False:
                train_lst = [(i+1,CLEARIMG(root_dir[0], i+1, debug=debug)) for i in range(num_buckets)]
                test_lst = [(i+1,CLEARIMG(root_dir[1], i+1, debug=debug)) for i in range(num_buckets)]
            else:
                # default to get all classes from train
                all_classes = sorted(set(map(lambda x: x.split("/")[-1][:-4], glob(root_dir[0]+f"/features/{feature_type}/*/**.pth", recursive=True))))
                num_classes = len(all_classes)
                id2class = dict(list(enumerate(all_classes)))
                print("Class-to-id mapping:", id2class)
                train_lst = [(i+1,ConcatDataset([CLEARUNSUP(root_dir[0], i+1, feature_type, id2class[j], j, debug=debug) \
                            for j in range(num_classes)])) for i in range(num_buckets)]
                test_lst = [(i+1,ConcatDataset([CLEARUNSUP(root_dir[1], i+1, feature_type, id2class[j], j, debug=debug) \
                            for j in range(num_classes)])) for i in range(num_buckets)]
            print(f"Loaded train data from {root_dir[0]}.")
            print(f"Loaded test data from {root_dir[1]}.")
            
        else:
            # split train/test 
            num_buckets = len(glob(root_dir+"/filelists/**.txt"))
            train_lst = []
            test_lst = []
            if pretrained == False:
                for i in range(1,num_buckets+1):
                    num_images = sum(1 for line in open(root_dir+"/filelists/"+str(i)+'.txt'))
                    all_idx = range(num_images)
                    if debug == True:
                        num_images = 25*2
                        all_idx = range(num_images)
                        split_ratio = 0.5
                    train_idx = set(random.sample(all_idx,int(num_images*split_ratio)))
                    test_idx = set(all_idx) - train_idx
                    train_lst.append((i,CLEARSubset(CLEARIMG(root_dir,i,debug=debug),list(train_idx))))
                    test_lst.append((i,CLEARSubset(CLEARIMG(root_dir,i,debug=debug),list(test_idx))))
            else:
                all_classes = sorted(set(map(lambda x: x.split("/")[-1][:-4], glob(root_dir+f"/features/{feature_type}/*/**.pth", recursive=True))))
                num_classes = len(all_classes)
                id2class = dict(list(enumerate(all_classes)))
                print("Class-to-id mapping:", id2class)
                for i in range(1, num_buckets+1):
                    all_bucket_features = ConcatDataset([CLEARUNSUP(root_dir, i, feature_type, id2class[j], j, debug=debug) \
                                                        for j in range(num_classes)]) 
                    num_features = len(all_bucket_features)
                    all_idx = range(num_features)
                    if debug == True:
                        num_features = 4*num_classes*2
                        all_idx = range(num_features)
                        split_ratio = 0.5
                    train_idx = set(random.sample(all_idx,int(num_features*split_ratio)))
                    test_idx = set(all_idx) - train_idx
                    train_lst.append((i,CLEARSubset(all_bucket_features,list(train_idx))))
                    test_lst.append((i,CLEARSubset(all_bucket_features,list(test_idx))))
            print(f"Loaded and splitted train/test data from {root_dir}.")
            
        if return_task_id:
            aval_train_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), train_lst))
            aval_test_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), test_lst))
        else:
            aval_train_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=0), train_lst))
            aval_test_lst = list(map(lambda x: AvalancheDataset(x[1], task_labels=0), test_lst))

        return dataset_benchmark(
            aval_train_lst,
            aval_test_lst,
            train_transform=train_transform,
            eval_transform=eval_transform,
            ) 

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",type=str, choices=["True", "False"],
                    help="only load limited amount of features/images in the dataset for each bucket \
                          for each class to debug",default="False")
    args = parser.parse_args()
    if args.debug == "False":
        args.debug = False
    else:
        args.debug = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #extracted = torch.load("/data3/zhiqiul/CLEAR-10-PUBLIC/features/moco_b0/1/baseball.pth") # dictionary. ImageID => Extracted Features, torch.Size([2048])

    # default transformation for debugging
    def get_transforms(is_pretrained):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if is_pretrained:
            train_transform, eval_transform = None, None
        return train_transform, eval_transform
    
    # test LwF strategy - distillation based
    # test ER strategy - replay based
    # test CWR - architecture based 
    # test SI - regularization based
    def train_eval(num_buckets, train_epochs, train_mb_size,
            eval_mb_size, device, scenario, mode, strat,
            is_pretrained, in_features=2048, num_classes=11):
        
        if torch.cuda.device_count() > 1:
            if is_pretrained:
                model = nn.DataParallel(nn.Linear(in_features,num_classes))
            else:
                model = nn.DataParallel(resnet18(pretrained=False))
        else:
            if is_pretrained:
                model = nn.Linear(in_features,num_classes)
            else:
                model = resnet18(pretrained=False)

        model.to(device)
        optimizer = Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.CyclicLR(optimizer, 0.0001, 0.1, cycle_momentum=False)

        eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                    timing_metrics(epoch=True),
                                    loggers=[InteractiveLogger()],
                                    benchmark=scenario,
                                    strict_checks=False)
        if strat == "LwF":
            strategy = LwF(model, optimizer, criterion, alpha=np.linspace(0,2,num=num_buckets).tolist(), train_mb_size=train_mb_size, 
                        train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, temperature=1, 
                        plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
        elif strat == "ER":
            buffer_size = 8
            strategy = Replay(model, optimizer, criterion, buffer_size, train_mb_size=train_mb_size, 
                      train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                      plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
        elif strat == "SI":
            strategy = SynapticIntelligence(model, optimizer, criterion, si_lambda=0.0001, train_mb_size=train_mb_size, 
                                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                                    plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
        elif strat == "CWR":
            strategy = CWRStar(model, optimizer, criterion, cwr_layer_name=None, train_mb_size=train_mb_size, 
                       train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                       plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
        else:
            raise ValueError("Invalid strategy.")    
        
        # TODO: verify experience id / task id, differ by 1? 
        print('Starting experiment...')
        train_stream = scenario.train_stream
        test_stream = scenario.test_stream
        for exp_id in range(len(train_stream)):
            print("Start of experience: ", train_stream[exp_id].current_experience)
            print("Current Classes: ", train_stream[exp_id].classes_in_this_experience)

            current_training_set = train_stream[exp_id].dataset
            print('This task contains', len(current_training_set), 'training examples')

            strategy.train(train_stream[exp_id])
            print('Training completed')

            current_test_set = test_stream[train_stream[exp_id].current_experience].dataset
            print('This task contains', len(current_test_set), 'test examples')
        
            if mode == "offline":
                print('Computing accuracy on each bucket')
                for j in range(len(test_stream)):
                    expid = format(test_stream[exp_id].current_experience,'03d')
                    taskid = format(test_stream[exp_id].task_label,'03d')
                    strategy.eval(test_stream[exp_id])[f'Top1_Acc_Exp/eval_phase/test_stream/Task{taskid}/Exp{expid}']   

            else: # online
                print('Computing accuracy on future buckets') # train test id offset by 1 in two streams
                for j in range(len(test_stream[exp_id:])): # evaluate on all future buckets to get forward transfer
                    expid = format(test_stream[exp_id].current_experience,'03d')
                    taskid = format(test_stream[exp_id].task_label,'03d')
                    strategy.eval(test_stream[exp_id])[f'Top1_Acc_Exp/eval_phase/test_stream/Task{taskid}/Exp{expid}'] 

    strat_lst = ["LwF","ER","SI","CWR"]
    
    # Test cases, is_pretrained = False
    train_transform, eval_transform = get_transforms(is_pretrained=False)

    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=True, mode="online", \
                                train_transform=train_transform, eval_transform=eval_transform, debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="online", strat=strat, is_pretrained=False)
    
    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=False, mode="online", \
                                train_transform=train_transform, eval_transform=eval_transform, debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="online", strat=strat, is_pretrained=False)
        
    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=True, mode="offline", split_ratio=0.7, \
                                train_transform=train_transform, eval_transform=eval_transform, debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="offline", strat=strat, is_pretrained=False)

    benchmark_instance = mkCLEAR(["/data3/zhiqiul/CLEAR-10-PUBLIC","/data3/zhiqiul/clear_datasets/CLEAR10-TEST-CLEANED"], return_task_id=True, mode="offline", \
                                train_transform=train_transform, eval_transform=eval_transform, debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="offline", strat=strat, is_pretrained=False)



    # Test cases, is_pretrained = True
    train_transform, eval_transform = get_transforms(is_pretrained=True)

    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=True, mode="online", \
                                train_transform=train_transform, eval_transform=eval_transform, pretrained=True, feature_type="moco_b0", debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="online", strat=strat, is_pretrained=True, in_features=2048, num_classes=11)

    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=False, mode="online", \
                                train_transform=train_transform, eval_transform=eval_transform, pretrained=True, feature_type="moco_b0", debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="online", strat=strat, is_pretrained=True, in_features=2048, num_classes=11)

    benchmark_instance = mkCLEAR("/data3/zhiqiul/CLEAR-10-PUBLIC", return_task_id=True, mode="offline", split_ratio=0.7, \
                                train_transform=train_transform, eval_transform=eval_transform, pretrained=True, feature_type="moco_b0", debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="offline", strat=strat, is_pretrained=True, in_features=2048, num_classes=11)

    # There are 3300 input features in /data3/zhiqiul/clear_datasets/CLEAR10-TEST. Will not be available in public version.
    benchmark_instance = mkCLEAR(["/data3/zhiqiul/CLEAR-10-PUBLIC","/data3/zhiqiul/clear_datasets/CLEAR10-TEST"], return_task_id=True, mode="offline", \
                                train_transform=train_transform, eval_transform=eval_transform, pretrained=True, feature_type="moco_b0", debug=args.debug)
    for strat in strat_lst:
        print("Current strategy:", strat)
        train_eval(num_buckets=11, train_epochs=3, train_mb_size=10, eval_mb_size=10, device=device, 
                    scenario=benchmark_instance, mode="offline", strat=strat, is_pretrained=True, in_features=2048, num_classes=11)
    
    print("All CLEAR benchmark successfully loaded!")
    sys.exit(0)

