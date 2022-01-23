import torch
import torchvision.transforms as transforms

from avalanche.benchmarks.generators import filelist_benchmark, tensors_benchmark 


def get_transforms(pretrain_feature):
    # copied from https://github.com/ElvishElvis/Continual-Learning/blob/master/load_dataset.py
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
        if pretrain_feature != "None":
            train_transform, eval_transform = None, None
        return train_transform, eval_transform

def get_data_set_offline(root, train_list, eval_list, train_transform, eval_transform, pretrain_feature):
    if pretrain_feature == "None":
            generic_scenario = filelist_benchmark(
            root,  
            train_list, 
            eval_list, 
            task_labels=range(1,len(train_list)+1), # starts from index 1 because the 0th bucket is used for pretrain
            train_transform=train_transform,
            eval_transform=eval_transform
            )
    else:
        generic_scenario = tensors_benchmark(
            train_tensors = [torch.load(file) for file in train_list], 
            test_tensors = [torch.load(file) for file in eval_list],
            task_labels = range(1,len(train_list)+1)
        )
    return generic_scenario

def get_data_set_online(root, train_list, train_transform, eval_transform, pretrain_feature):
    if pretrain_feature == "None":
        generic_scenario = filelist_benchmark(
            root,  
            train_list,
            train_list,
            task_labels=range(1,len(train_list)+1), # starts from index 1 because the 0th bucket is used for pretrain
            train_transform=train_transform,
            eval_transform=eval_transform
            )
    else:
        generic_scenario = tensors_benchmark(
                train_tensors = [torch.load(file) for file in train_list], 
                test_tensors = [torch.load(file) for file in train_list],
                task_labels = range(1,len(train_list)+1)
            )

    return generic_scenario

