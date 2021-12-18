import random
import argparse
import datetime
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
import torchvision.transforms as transforms
from dataset import CLEAR10IMG, CLEAR10MOCO
from train import *

def parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root_dir", type=str,
                    help="root directory that contains all image buckets",
                    default="/data3/zhiqiul/clear_datasets/CLEAR10-TEST")
  parser.add_argument("--save_path", type=str,
                    help="directory to save all outputs",
                    default="~/study_avalanche/outputs")
  parser.add_argument("--debug",type=str, choices=["True", "False"],
                    help="only load the first 25 features/images in the dataset for each bucket \
                          for each class to debug",default="False")
  parser.add_argument("--num_buckets", type=int,
                    help="number of buckets",default=11)
  parser.add_argument("--is_pretrained", type=str, choices=["True", "False"],
                    help="True if using raw image features, False if using pretrained features",
                    default="False")
  parser.add_argument("--pretrained_path", type=str,
                    help="directory of pretrained features",
                    default="/data/jiashi/moco_resnet50_clear_10_feature")
  parser.add_argument("--in_features",type=int,
                    help="size of in_features for linear classifier",default=2048)
  parser.add_argument("--num_classes",type=int,
                    help="number of classes for one single bucket",default=11)
  parser.add_argument("--buffer_size",type=str,
                    help="Buffer size for [iid, streaming]",default="2310")
  parser.add_argument("--train_mb_size", type=int,
                    help="train mini batch size",default=64)
  parser.add_argument("--train_epochs", type=int,
                    help="train epochs",default=70)
  parser.add_argument("--eval_mb_size", type=int,
                    help="test mini batch size",default=64)
  parser.add_argument("--lr", type=float,
                    help="initial learning rate",default=0.001)
  parser.add_argument("--strategies", type=str,
                    help="a list of all learning strategies: EWC, SI, LwF, CWR, GDumb, ER, AGEM, Naive, NaiveBiased")
  parser.add_argument("--alpha",type=float,
                    help="alpha in fixed/dynamic alpha biased reservoir sampling")
  parser.add_argument("--biased_mode",type=str, choices=["fixed", "dynamic"],
                    help="fixed or dynamic biased reservoir sampling")
  args = parser.parse_args()
  if args.debug == "False":
      args.debug = False
  else:
      args.debug = True
  if args.is_pretrained == "False":
      args.is_pretrained = False
  else:
      args.is_pretrained = True
  args.buffer_size = args.buffer_size.split(",")
  args.strategies = args.strategies.split(",")

  return args

def get_transforms(is_pretrained):
    # copied from https://github.com/ElvishElvis/Continual-Learning/blob/master/load_dataset.py
    # Note that this is not exactly imagenet transform/moco transform for val set
    # Because we resize to 224 instead of 256
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if is_pretrained:
        train_transform, test_transform = None, None
    return train_transform, test_transform

if __name__ == '__main__':
    random.seed(42)

    args = parser()

    root_dir = args.root_dir
    save_path = args.save_path
    num_buckets = args.num_buckets
    train_transform,test_transform = get_transforms(args.is_pretrained)
    exp_time = datetime.datetime.now().strftime("%m%d%Y%H%M%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.is_pretrained:
        # iid
        clear10_train = [(i,CLEAR10IMG(root_dir,i,form="train",debug=args.debug)) for i in range(num_buckets)]
        clear10_train = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_train))
        clear10_test = [(i,CLEAR10IMG(root_dir,i,form="test",debug=args.debug)) for i in range(num_buckets)]
        clear10_test = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_test))

        # streaming
        clear10_all = [(i,CLEAR10IMG(root_dir,i,form="all",debug=args.debug)) for i in range(num_buckets)]
        clear10_all = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_all))
    else: # moco, exclude first bucket used for pretrain
        # iid
        clear10_train = [(i,CLEAR10MOCO(args.pretrained_path,i,device,form="train",debug=args.debug)) for i in range(1,num_buckets)] 
        clear10_train = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_train))
        clear10_test = [(i,CLEAR10MOCO(args.pretrained_path,i,device,form="test",debug=args.debug)) for i in range(1,num_buckets)]
        clear10_test = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_test))

        # streaming
        clear10_all = [(i,CLEAR10MOCO(args.pretrained_path,i,device,form="all",debug=args.debug)) for i in range(1,num_buckets)]
        clear10_all = list(map(lambda x: AvalancheDataset(x[1], task_labels=x[0]), clear10_all))
    
    print(f"Dataset successfully loaded from {root_dir}")

    # from the paper: 
    # if offline: train/test0, train/test1, train/test2...
    # train0 -> h0, test on test0/1/2/....
    # train0+train1 -> h1, test on test0/1/2/....

    # if online 10 buckets: train0-test1-train0/train1-test2-train0/train1/train2-test3...
    # implement reservoir sampling for online training

    iid_scenario = dataset_benchmark(
        clear10_train,
        clear10_test,
        train_transform=train_transform,
        eval_transform=test_transform,
    ) 

    streaming_scenario = dataset_benchmark(
        clear10_all[:-1],
        clear10_all[1:],
        train_transform=train_transform,
        eval_transform=test_transform,
    ) 
    for strategy in args.strategies:
      train_eval(save_path, exp_time, num_buckets, iid_scenario, 
                 strategy, device, args.is_pretrained, args.train_mb_size, args.train_epochs, 
                 args.eval_mb_size, int(args.buffer_size[0]), args.lr, args.in_features, args.num_classes, 
                 alpha=args.alpha, 
                 setting="iid") 
      train_eval(save_path, exp_time, num_buckets, streaming_scenario, 
                 strategy, device, args.is_pretrained, args.train_mb_size, args.train_epochs, 
                 args.eval_mb_size, int(args.buffer_size[1]), args.lr, args.in_features, args.num_classes, 
                 alpha=args.alpha, 
                 setting="streaming")

