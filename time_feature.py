import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr  
from PIL import Image


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
import torch.nn.functional as F


from tqdm import tqdm
import os


def load_images(images_dir):
    train_df = pd.read_csv(images_dir + "/1/train.txt",sep=" ",names=["image_path","target"], header=None)
    train_path = train_df["image_path"].apply(lambda x: image_header + x)
    train_target = train_df["target"]
    d_train = {k: [] for k in range(0,11)} 
    for i, cls in enumerate(train_target):
        img = Image.open(train_path[i])
        if img.mode != "RGB":
            img = img.convert("RGB")
        transformed = transform(img)
        d_train[cls].append(transformed.numpy().flatten()) #[R*row*col, G.., B..]
    
    test_dicts = []
    for bucket in range(1,11):
        d = {k: [] for k in range(0,11)} 
        test_df = pd.read_csv(images_dir + f"/{bucket}/test.txt",sep=" ",names=["image_path","target"], header=None)
        test_path = test_df["image_path"].apply(lambda x: image_header + x)
        test_target = test_df["target"]
        for i, cls in enumerate(test_target):
            img = Image.open(test_path[i])
            if img.mode != "RGB":
                img = img.convert("RGB")
            transformed = transform(img)
            d[cls].append(transformed.numpy().flatten())
        test_dicts.append(d)

    return d_train, test_dicts

def load_features(data_dir):
    # train on bucket 1
    train_feat = torch.load(data_dir + f'/1/train.pth')[0].numpy() 
    train_target = torch.load(data_dir + f'/1/train.pth')[1] # list
    d_train = {k: [] for k in range(0,11)} 
    for i, cls in enumerate(train_target):
        d_train[cls].append(train_feat[i])

    test_dicts = []
    for bucket in range(1,11):
        d = {k: [] for k in range(0,11)} 
        b_feat = torch.load(data_dir + f'/{bucket}/test.pth')[0].numpy() 
        b_target = torch.load(data_dir + f'/{bucket}/test.pth')[1] # list
        for i, cls in enumerate(b_target):
            d[cls].append(b_feat[i])
        test_dicts.append(d)
        
    return d_train, test_dicts

def cosine(x,y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

def dot(x,y):
    return np.dot(x,y)

def L2(x,y):
    return np.linalg.norm(x-y)

def plot_distance_matrix(train, test, normalized=False, pretrained=False):
    if pretrained:
        exp_name = feature_name
    else:
        exp_name = "images"
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 17))
    newsize = ((buckets+1)*2, (buckets+1)*2)
    eps = 2 # black line width
    ## all class average
    M00R = np.zeros((224*224,(buckets+1)*2, 3))
    M00G = np.zeros((224*224,(buckets+1)*2, 3))
    M00B = np.zeros((224*224,(buckets+1)*2, 3))
    M00all = np.zeros((224*224,(buckets+1)*2, 3))
    for cls in train:
        M0R = np.zeros((224*224,(buckets+1)*2, 3))
        M0G = np.zeros((224*224,(buckets+1)*2, 3))
        M0B = np.zeros((224*224,(buckets+1)*2, 3))
        M0all = np.zeros((224*224,(buckets+1)*2, 3))
        # train b1
        M0R[:,0,0] = np.mean(train[cls],axis=0)[0:224*224]
        M0R[:,1,:] = np.ones((224*224,3))
        M0G[:,0,1] = np.mean(train[cls],axis=0)[224*224:2*224*224]
        M0G[:,1,:] = np.ones((224*224,3))
        M0B[:,0,2] = np.mean(train[cls],axis=0)[2*224*224:3*224*224]
        M0B[:,1,:] = np.ones((224*224,3))
        for bucket in range(1,buckets+1):
            # test bucket 1 to 10 from left to right
            M0R[:,bucket*2,0] = np.mean(test[bucket-1][cls],axis=0)[0:224*224]
            M0R[:,bucket*2+1,:] = np.ones((224*224,3))
            M0G[:,bucket*2,1] = np.mean(test[bucket-1][cls],axis=0)[224*224:2*224*224]
            M0G[:,bucket*2+1,:] = np.ones((224*224,3))
            M0B[:,bucket*2,2] = np.mean(test[bucket-1][cls],axis=0)[224*224:2*224*224]
            M0B[:,bucket*2+1,:] = np.ones((224*224,3))
        # for r in range(224):
        #     if r != 0:
        #         M0R[r*224-eps:r*224,:,:] = np.zeros((eps,(buckets+1)*2,3))
        #         M0G[r*224-eps:r*224,:,:] = np.zeros((eps,(buckets+1)*2,3))
        #         M0B[r*224-eps:r*224,:,:] = np.zeros((eps,(buckets+1)*2,3))
        M0all[:,:,0] = M0R[:,:,0]
        M0all[:,:,1] = M0G[:,:,1]
        M0all[:,:,2] = M0B[:,:,2]
        Image.fromarray((M0R*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/{classes[cls]}_R.png")
        Image.fromarray((M0G*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/{classes[cls]}_G.png")
        Image.fromarray((M0B*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/{classes[cls]}_B.png")
        Image.fromarray((M0all*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/{classes[cls]}_all.png")
        
        M00R[:,:,0] += M0R[:,:,0]
        M00G[:,:,1] += M0G[:,:,1]
        M00B[:,:,2] += M0B[:,:,2]

        
        img0_all = np.array(Image.fromarray((M0all*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR),dtype=np.uint8)
        for bucket in range(0,buckets+1):
            img0_all[:,bucket*2+1,:] = 128
        Image.fromarray(img0_all).convert('RGB').save(save_dir+f"/RGB/{classes[cls]}_all_grey.png")
    
    M00R /= len(classes)
    M00G /= len(classes)
    M00B /= len(classes)
    for bucket in range(0,buckets+1):
        M00R[:,bucket*2+1,:] = np.ones((224*224,3))
        M00G[:,bucket*2+1,:] = np.ones((224*224,3))
        M00B[:,bucket*2+1,:] = np.ones((224*224,3))
    M00all[:,:,0] = M00R[:,:,0]
    M00all[:,:,1] = M00G[:,:,1]
    M00all[:,:,2] = M00B[:,:,2]
    Image.fromarray((M00R*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/avg_R.png")
    Image.fromarray((M00G*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/avg_G.png")
    Image.fromarray((M00B*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/avg_B.png")
    Image.fromarray((M00all*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR).save(save_dir+f"/RGB/avg_all.png")
    Image.fromarray((M00all*255).astype(np.uint8)).convert('RGB').save(save_dir+f"/RGB/avg_all_long.png")
    
    img00_all = np.array(Image.fromarray((M00all*255).astype(np.uint8)).convert('RGB').resize(newsize, resample=Image.Resampling.BILINEAR),dtype=np.uint8)
    for bucket in range(0,buckets+1):
        img00_all[:,bucket*2+1,:] = 128
    Image.fromarray(img00_all).convert('RGB').save(save_dir+f"/RGB/avg_all_grey.png")
    img00_all_long = np.array(Image.fromarray((M00all*255).astype(np.uint8)).convert('RGB'),dtype=np.uint8)
    for bucket in range(0,buckets+1):
        img00_all_long[:,bucket*2+1,:] = 128
    Image.fromarray(img00_all_long).convert('RGB').save(save_dir+f"/RGB/avg_all_grey_long.png")
    '''
    M1 = np.zeros((len(classes)+1,buckets)) 
    for cls in train:
        for bucket in range(1,buckets+1):
            M1[cls, bucket-1] = cosine(np.mean(train[cls],axis=0),
                                        np.mean(test[bucket-1][cls],axis=0))
    M2 = np.zeros((len(classes)+1,buckets)) 
    # for cls in train:
    #     for bucket in range(1,buckets+1):
    #         M2[cls-1, bucket-1] = dot(np.mean(train[cls],axis=0),
    #                                     np.mean(test[bucket-1][cls],axis=0))
    M3 = np.zeros((len(classes)+1,buckets)) 
    for cls in train:
        for bucket in range(1,buckets+1):
            M3[cls, bucket-1] = L2(np.mean(train[cls],axis=0),
                                        np.mean(test[bucket-1][cls],axis=0))
    if normalized:
        M1[-1,:] = np.mean(M1[:-1,:],axis=0)
        s = sns.heatmap(M1, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[0],\
                        xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        s.axes.set_title(f"Cosine", fontsize=20)

        
        # M2[-1,:] = np.mean(M2[:-1,:],axis=0)
        # s = sns.heatmap(M2, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[1],\
        #                 xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        # s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        # s.axes.set_title(f"Dot Product", fontsize=20)
        
        M3[-1,:] = np.mean(M3[:-1,:],axis=0)
        s = sns.heatmap(M3, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[1],\
                        xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        s.axes.set_title(f"Euclidean Distance", fontsize=20)

        plt.suptitle(f"Similarity Matrix for {bucket_name} w/ {exp_name}", fontsize = 22) 
        plt.savefig(save_dir+f"/distM/distance_matrix_{bucket_name}_{exp_name}.jpg")
        plt.clf()
    else:
        # normalized by the distance of b1 train vs. b1 test
        for i in range(len(classes)):
            M1[i,:] /= M1[i,0]
        M1[-1,:] = np.mean(M1[:-1,:],axis=0)
        s = sns.heatmap(M1, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[0],\
                        xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        s.axes.set_title(f"Cosine", fontsize=20)

        
        # for i in range(len(classes)):
        #     M2[i,:] /= M2[i,0]
        # M2[-1,:] = np.mean(M2[:-1,:],axis=0)
        # s = sns.heatmap(M2, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[1],\
        #                 xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        # s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        # s.axes.set_title(f"Dot Product", fontsize=20)

        for i in range(len(classes)):
            M3[i,:] /= M3[i,0]
        M3[-1,:] = np.mean(M3[:-1,:],axis=0)
        s = sns.heatmap(M3, annot=True, fmt=".4g", cmap="YlGnBu", ax=axes[1],\
                        xticklabels=range(1,11),yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets')
        s.axes.set_title(f"Euclidean Distance", fontsize=20)

        plt.suptitle(f"Normalized Similarity Matrix for {bucket_name} w/ {exp_name}", fontsize = 22) 
        plt.savefig(save_dir+f"/distM/norm_distance_matrix_{bucket_name}_{exp_name}.jpg")
        plt.clf()
    
    return M1, M2, M3    
    '''



# TODO: 
# 1. rewrite the whole training process without avalanche, 
# (actually don't need that, we only need the first step train on B1 and eval on others)
# and plot conflict matrix
# output: a csv, # of (gradient step-1) * ((11 classes delta test acc) * 10 buckets 
# + binary feature that shows whether a class is in the current training batch or not)

# load raw images
class CLEAR10IMG(Dataset):

    def __init__(self, images_dir, bucket, form="all", transform=None):
        self.transform = transform
        self.bucket = bucket
        self.form = form
        df = pd.read_csv(images_dir + f"/{self.bucket}/{form}.txt",sep=" ",names=["image_path","target"], header=None)
        self.img_paths = df["image_path"].apply(lambda x: image_header + x)
        self.targets = df["target"]

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self,idx):
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is None:
            return img, torch.tensor(self.targets[idx])
        else:
            return self.transform(img), torch.tensor(self.targets[idx])

class CLEAR10MOCO(Dataset):
    def __init__(self, feat_dir, bucket, form="all"): 
        self.feat_dir = feat_dir
        self.bucket = bucket
        self.form = form
        self.feats, self.targets = torch.load(feat_dir + f'/{bucket}/{form}.pth')
    
    def __len__(self): 
        return len(self.targets)

    def __getitem__(self,idx):
        return self.feats[idx], torch.tensor(self.targets[idx])

def train_test_conflicts(device, binary=False, pretrained=False, normalized=False, redo=False):
    plt.rcParams["figure.figsize"] = (120,16)
    if pretrained:
        exp_name = feature_name
        model = nn.DataParallel(nn.Linear(2048,len(classes))).to(device)
        train_set = CLEAR10MOCO(feature_dir, 1, form="train")
        test_sets = [CLEAR10MOCO(feature_dir, i, form="test")
                     for i in range(1,11)]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loaders = [DataLoader(test_sets[i-1], batch_size=1, shuffle=False)
                        for i in range(1,11)]
    else:
        exp_name = "images"
        model = nn.DataParallel(resnet18(pretrained=False)).to(device)
        model.module.fc = nn.Linear(512, len(classes)).to(device)
        train_set = CLEAR10IMG(image_name_dir, 1, form="train", transform=transform)
        test_sets = [CLEAR10IMG(image_name_dir, i, form="test", transform=transform)
                    for i in range(1,11)]
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        test_loaders = [DataLoader(test_sets[i-1], batch_size=1, shuffle=False)
                    for i in range(1,11)]
    if os.path.exists(save_dir + f"/conflict/conflict_{exp_name}.csv") and redo == False:
        df = pd.read_csv(save_dir + f"/conflict/conflict_{exp_name}.csv")
    else:
        # train on b1
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        existence = [] # 10 classes existence * gradient steps
        accs = [] # (11 classes * 10 buckets) * gradient steps
        for _ in tqdm(range(epochs)):
            scheduler = StepLR(optimizer,step_size=60,gamma=0.1)
            for idx, (img, target) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                img, target = img.to(device), target.to(device)
                loss = nn.CrossEntropyLoss()(model(img), target)     
                loss.backward() # one gradient step
                optimizer.step()
                scheduler.step()
                class_count = [(target.detach().cpu().numpy() == i).sum() for i in range(0,11)]
                existence.append(class_count)
                # eval immediately
                model.eval()
                with torch.no_grad():
                    dict_tests = {k: [] for k in range(1,11)}
                    for bucket in range(1, 11):
                        b_test = {k: [] for k in range(0,11)} 
                        for idx, (img, target) in enumerate(test_loaders[bucket-1]):
                            img, target = img.to(device), target.to(device)
                            output = model(img)
                            prob = F.softmax(output, dim=1)
                            loss = F.nll_loss(torch.log(prob), target)
                            y_pred = torch.argmax(prob).detach().cpu().numpy()
                            target = target.detach().cpu().numpy()[0]
                            b_test[target].append(y_pred)
                        for cls in b_test:
                            y_preds = list(b_test[cls])
                            b_test[cls] = sum(np.array(y_preds) == cls)/len(y_preds)
                        dict_tests[bucket] = np.array(list(b_test.values()))
                    accs.append(np.array(list(dict_tests.values())).flatten())
        accs = np.array(accs)
        accs_delta = accs[1:,:] - accs[:-1,:]
        colnames = [f"{classes[i]}_freq" for i in range(0,len(classes))] + \
                    [f"b{j}_{classes[i]}_delta" for j in range(1,11) for i in range(0,len(classes))]
        df = pd.DataFrame(np.concatenate([existence[1:][:], accs_delta], axis=1),columns=colnames)
        df.to_csv(save_dir + f"/conflict/conflict_{exp_name}.csv",index=False)

    M = np.ones((len(classes)+1,len(classes)*buckets))
    if binary:
        for i in range(len(classes)):
            for j in range(len(classes)*buckets): 
                df[f"{classes[i]}_freq"] = [1 if x > 0 else 0 for x in df[f"{classes[i]}_freq"] ]
                M[i,j] = pearsonr(df[f"{classes[i]}_freq"],df[f"b{j % buckets + 1}_{classes[j // buckets]}_delta"])[0]
        exp_name += "_binary"
    else:
        for i in range(len(classes)):
            for j in range(len(classes)*buckets): 
                M[i,j] = pearsonr(df[f"{classes[i]}_freq"],df[f"b{j % buckets + 1}_{classes[j // buckets]}_delta"])[0]
    if normalized:
        for i in range(len(classes)):
            for j in range(len(classes)):
                M[i,j*buckets:(j+1)*buckets] /= M[i,j*buckets]
                
        M[-1,:] = np.mean(M[:-1,:],axis=0)
        s = sns.heatmap(M, annot=True, fmt=".4g", cmap="YlGnBu",\
                        xticklabels=[f"b{j}_{classes[i]}" for i in range(0,len(classes)) for j in range(1,11)],yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets Acc Delta')
        s.set_yticklabels(s.get_yticklabels(), rotation = -45)
        s.axes.set_title(f"Normalized Correlation between Frequency of Classes in {bucket_name} batches vs. Delta Acc of Classes in Test w/ {exp_name}", 
                        fontsize=18)
        plt.savefig(save_dir + f"/conflict/norm_conflict_{exp_name}.jpg")
        plt.clf()
    else:
        M[-1,:] = np.mean(M[:-1,:],axis=0)
        s = sns.heatmap(M, annot=True, fmt=".4g", cmap="YlGnBu",\
                        xticklabels=[f"b{j}_{classes[i]}" for i in range(0,len(classes)) for j in range(1,11)],yticklabels=classes+["class_mean"])   
        s.axes.set(ylabel='Train Classes', xlabel='Test Buckets Acc Delta')
        s.set_yticklabels(s.get_yticklabels(), rotation = -45)
        s.axes.set_title(f"Correlation between Frequency of Classes in {bucket_name} batches vs. Delta Acc of Classes in Test w/ {exp_name}", 
                        fontsize=18)
        plt.savefig(save_dir + f"/conflict/conflict_{exp_name}.jpg")
        plt.clf()
    return

# TODO: 
# 2. use spectral embeddings to reduce dimension, 
# use cosine similarity as affinity measure
# find the optimal number of reduced dimensions so that using mocob0 feature has
# a similar performance on trainB1-valB1,
# and see how delta acc matrix behave. [Just try Naive FineTune]



if __name__ == '__main__':
    buckets = 10 # exclude b0
    bucket_name = "modelb1"
    save_header = "/data3/siqiz/clear10_outputs/visualization"
    save_dir = save_header + '/' + bucket_name

    feature_name = "moco_b0"
    feature_header = "/data3/zhiqiul/CLEAR-10-PUBLIC/training_folder/testset_ratio_0.3/seed_0/features" # moco_features
    feature_dir = feature_header + "/" + feature_name

    image_header = "/data3/zhiqiul/CLEAR-10-PUBLIC/"
    image_name_dir = image_header + "training_folder/testset_ratio_0.3/seed_0/filelists"
    

    classes = open("/data3/zhiqiul/CLEAR-10-PUBLIC/class_names.txt").read().split('\n')
    classes_id = range(0,1+len(classes))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(), # cancel randomness
        transforms.ToTensor(),
        #normalize, # don't want negative values
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    epochs = 70

    #train, test = load_features(feature_dir)
    #plot_distance_matrix(train, test, normalized=False, pretrained=True)
    #plot_distance_matrix(train, test, normalized=True, pretrained=True)
    train, test = load_images(image_name_dir)
    plot_distance_matrix(train, test, normalized=False, pretrained=False)
    #plot_distance_matrix(train, test, normalized=True, pretrained=False)
    #train_test_conflicts(device, binary=False, pretrained=True, normalized=False, redo=False)
    #train_test_conflicts(device, binary=False, pretrained=True, normalized=True, redo=False)
    #train_test_conflicts(device, binary=False, retrained=False, normalized=False, redo=False)
    #train_test_conflicts(device, binary=False, pretrained=False, normalized=True, redo=False)
    #train_test_conflicts(device, binary=True, pretrained=True, normalized=False, redo=False)
    #train_test_conflicts(device, binary=True, pretrained=True, normalized=True, redo=False)
    #train_test_conflicts(device, binary=True, pretrained=False, normalized=False, redo=False)
    #train_test_conflicts(device, binary=True, pretrained=False, normalized=True, redo=False)