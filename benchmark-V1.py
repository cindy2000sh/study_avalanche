# benchmark-V1.py

import torch
import torchvision.transforms as transforms

from avalanche.benchmarks.generators import filelist_benchmark, tensors_benchmark 


def get_transforms(is_pretrained):
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
        if is_pretrained:
            train_transform, eval_transform = None, None
        return train_transform, eval_transform

# TODO: change load_dataset.py here: https://github.com/ElvishElvis/Continual-Learning/blob/master/load_dataset.py
# TODO: input == args in original code
def get_data_set_offline(root, train_list, eval_list, train_transform, eval_transform, pretrained):
    if pretrained == True:
            generic_scenario = filelist_benchmark(
            root,  # /data3/zhiqiul/clear_datasets/CLEAR10-TEST-CLEANED/labeled_images
            train_list, # ["/1/train.txt", ....]
            eval_list, # ["/1/test.txt",...]
            task_labels=range(1,len(train_list)+1), # starts from index 1 because the 0th bucket is used for pretrain
            train_transform=train_transform,
            eval_transform=eval_transform
            )
    else:
        # pretrained == True
        # root : "/data3/siqiz/clear_datasets/CLEAR-10-PUBLIC/training_folder/testset_ratio_0.3/seed_0/features/moco_b0"
        # train_list: ["/1/train.pth", ...]
        train_feat_lst = list(map(lambda x: root + x, train_list))
        eval_feat_lst = list(map(lambda x: root + x, eval_list))
        generic_scenario = tensors_benchmark(
            train_tensors = [torch.load(file) for file in train_feat_lst], 
            test_tensors = [torch.load(file) for file in eval_feat_lst],
            task_labels = range(1,len(train_list)+1)
        )
    return generic_scenario

def get_data_set_online(root, train_list, train_transform, eval_transform, pretrained):
    if pretrained == False:
        generic_scenario = filelist_benchmark(
            root,  
            train_list,
            train_list,
            # default evaluation is to evaluate the whole test stream but 
            # we can only keep the future bucket result during post processing
            task_labels=range(1,len(train_list)+1), # starts from index 1 because the 0th bucket is used for pretrain
            train_transform=train_transform,
            eval_transform=eval_transform
            )
    else:
        train_feat_lst = list(map(lambda x: root + x, train_list))
        generic_scenario = tensors_benchmark(
                train_tensors = [torch.load(file) for file in train_feat_lst], 
                test_tensors = [torch.load(file) for file in train_feat_lst],
                task_labels = range(1,len(train_list)+1)
            )
    # TODO: need to save current model and call eval again for loading private test set [whole test set] on jiashi's train.py
    # TODO: ask what to compare, since plotting is unavailable
    # jia's output path: /data/jiashi
    return generic_scenario

