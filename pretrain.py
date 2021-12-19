import datetime
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import copy
import lightly
from lightly.models.modules.heads import MoCoProjectionHead, SimCLRProjectionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, batch_shuffle, batch_unshuffle
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from dataset import CLEAR10IMG
from main import get_transforms

class CLEAR10IMGPL(CLEAR10IMG):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def __getitem__(self,idx):
        img,target = super().__getitem__(idx)
        if self.form == "all":
            return img,target,self.img_paths[idx]
        elif self.form == "train":
            return img,target,self.train_img_paths[idx]
        else:
            return img,target,self.test_img_paths[idx]

class BYOL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        update_momentum(model.backbone, model.backbone_momentum, m=0.99)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=0.99
        )
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)



class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]

# moco pretrain
# https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html


class MocoModel(pl.LightningModule): # Moco-V2
    def __init__(self,num_classes,memory_bank_size=None):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_classes=num_classes, num_splits=1)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 2048, 128) 
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, 0.99
        )

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]


# feature extract
# modified from linear classifier part of https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html

class FeatureExtractor(pl.LightningModule):
    def __init__(self, backbone, save_path, mode):
        super().__init__()
        # use the pretrained mode-ResNet backbone
        self.backbone = nn.Sequential(backbone,nn.Linear(1,4)) 
        # TODO: suspicious!!! Add one Linear layer to match with previous pretrained outcomes
        # freeze the backbone
        deactivate_requires_grad(backbone)
        self.save_path = save_path
        self.mode = mode

    def forward(self, x, file_name):
        file_name = file_name.split(".")[0]
        res_tensor = self.backbone(x).flatten(start_dim=1).cpu().detach()
        torch.save(res_tensor, save_path+f"/pretrained_models/{file_name}_{self.mode}.pth")
        return res_tensor

    def training_step(self, batch, batch_idx):
        x, _, file_name = batch
        self.forward(x, file_name[0].split("/")[-1])

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def custom_histogram_weights(self):
        return

    def configure_optimizers(self):
        return

    


if __name__ == '__main__':
    num_workers = 56
    batch_size = 8
    memory_bank_size = 4
    seed = 42
    max_epochs = 10
    root_dir = "/data3/zhiqiul/clear_datasets/CLEAR10-TEST"
    save_path = "/data3/siqiz/avalanche/outputs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modes = ["MOCO","SIMCLR","BYOL"] 
    curr_time = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
    num_classes = 11

    pl.seed_everything(seed)
    
    for mode in modes:
        if mode == "MOCO":
            # MoCo v2 uses SimCLR augmentations, additionally, disable blur
            collate_fn_pretrain = lightly.data.SimCLRCollateFunction(
                input_size=224,
                gaussian_blur=0.,
                normalize={'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]},
                hf_prob=0.5
            )

            # only use the 0th bucket to pretrain
            dataset_train_moco = CLEAR10IMGPL(root_dir,0,form="all",debug=True)
            dataset_ft_moco = CLEAR10IMGPL(root_dir,0,form="all",debug=True,transform=get_transforms(False)[0])
            
            dataloader_train_moco = torch.utils.data.DataLoader(
                                        dataset_train_moco,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn_pretrain,
                                        drop_last=True,
                                        num_workers=num_workers)
            dataloader_ft_moco = torch.utils.data.DataLoader(
                                        dataset_ft_moco,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=num_workers)
            
            print("Dataset successfully loaded.")

            # use GPU if available
            gpus = 1 if torch.cuda.is_available() else 0 

            model = MocoModel(num_classes,memory_bank_size)
            model = model.to(device)
            
            trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
            trainer.fit(
                model,
                dataloader_train_moco
            )

            model.eval()
            feature_extractor = FeatureExtractor(model.backbone,save_path,mode)
            trainer_ft = pl.Trainer(max_epochs=1, gpus=gpus)
            trainer_ft.fit(
                feature_extractor,
                dataloader_ft_moco
            )
        elif mode == "SIMCLR":
            collate_fn_pretrain = lightly.data.SimCLRCollateFunction(
                input_size=224,
                normalize={'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]},
                hf_prob=0.5
            )
            dataset_train_simclr = CLEAR10IMGPL(root_dir,0,form="all",debug=True)
            dataset_ft_simclr = CLEAR10IMGPL(root_dir,0,form="all",debug=True,transform=get_transforms(False)[0])
            
            dataloader_train_simclr = torch.utils.data.DataLoader(
                                        dataset_train_simclr,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn_pretrain,
                                        drop_last=True,
                                        num_workers=num_workers)
            dataloader_ft_simclr = torch.utils.data.DataLoader(
                                        dataset_ft_simclr,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=num_workers)
            
            print("Dataset successfully loaded.")

            # use GPU if available
            gpus = 1 if torch.cuda.is_available() else 0 

            model = SimCLRModel()
            model = model.to(device)
            
            trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
            trainer.fit(
                model,
                dataloader_train_simclr
            )

            model.eval()
            feature_extractor = FeatureExtractor(model.backbone,save_path,mode)
            trainer_ft = pl.Trainer(max_epochs=1, gpus=gpus)
            trainer_ft.fit(
                feature_extractor,
                dataloader_ft_simclr
            )
        else: # BYOL
            collate_fn_pretrain = lightly.data.SimCLRCollateFunction(
                input_size=224,
                normalize={'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]},
                hf_prob=0.5
            )
            dataset_train_byol = CLEAR10IMGPL(root_dir,0,form="all",debug=True)
            dataset_ft_byol = CLEAR10IMGPL(root_dir,0,form="all",debug=True,transform=get_transforms(False)[0])
            
            dataloader_train_byol = torch.utils.data.DataLoader(
                                        dataset_train_byol,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn_pretrain,
                                        drop_last=True,
                                        num_workers=num_workers)
            dataloader_ft_byol = torch.utils.data.DataLoader(
                                        dataset_ft_byol,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=num_workers)
            
            print("Dataset successfully loaded.")

            # use GPU if available
            # TODO: multi-gpu
            gpus = 1 if torch.cuda.is_available() else 0 

            model = BYOL()
            model = model.to(device)
            
            trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
            trainer.fit(
                model,
                dataloader_train_byol
            )

            model.eval()
            feature_extractor = FeatureExtractor(model.backbone,save_path,mode)
            trainer_ft = pl.Trainer(max_epochs=1, gpus=gpus)
            trainer_ft.fit(
                feature_extractor,
                dataloader_ft_byol
            )