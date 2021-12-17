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
                    default="/data3/zhiqiul/clear_datasets/CLEAR10-TEST/")
  parser.add_argument("--save_path", type=str,
                    help="directory to save all outputs")
  parser.add_argument("--debug",type=bool,
                    help="only load the first 25 features/images in the dataset for each bucket \
                          for each class to debug",default=False)
  parser.add_argument("--num_buckets", type=int,
                    help="number of buckets",default=11)
  parser.add_argument("--is_pretrained", type=bool, choices=[True, False],
                    help="True if using raw image features, False if using pretrained features",
                    default=False)
  parser.add_argument("--pretrained_path", type=str,
                    help="directory of pretrained features",
                    default="/data/jiashi/moco_resnet50_clear_10_feature")
  parser.add_argument("--in_features",type=int,
                    help="size of in_features for linear classifier",default=2048)
  parser.add_argument("--num_classes",type=int,
                    help="number of classes for one single bucket",default=11)
  parser.add_argument("--buffer_size",type=list,
                    help="Buffer size for [iid, streaming]",default=2310)
  parser.add_argument("--train_mb_size", type=int,
                    help="train mini batch size",default=64)
  parser.add_argument("--train_epochs", type=int,
                    help="train epochs",default=70)
  parser.add_argument("--eval_mb_size", type=int,
                    help="test mini batch size",default=64)
  parser.add_argument("--lr", type=int,
                    help="initial learning rate",default=0.001)
  parser.add_argument("--strategies", type=list,
                    help="a list of all learning strategies: EWC, SI, LwF, CWR, GDumb, ER, AGEM, Naive, NaiveBiased")
  parser.add_argument("--alpha",type=int,
                    help="alpha in fixed/dynamic alpha biased reservoir sampling")
  parser.add_argument("--biased_mode",type=str, choices=["fixed", "dynamic"],
                    help="fixed or dynamic biased reservoir sampling")
  return parser.parse_args()

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
    if not is_pretrained:
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
        clear10_train = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_train))
        clear10_test = [(i,CLEAR10IMG(root_dir,i,form="test",debug=args.debug)) for i in range(num_buckets)]
        clear10_test = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_test))

        # streaming
        clear10_all = [(i,CLEAR10IMG(root_dir,i,form="all",debug=args.debug)) for i in range(num_buckets)]
        clear10_all = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_all))
    else: # moco, exclude first bucket used for pretrain
        # iid
        clear10_train = [(i,CLEAR10MOCO(root_dir,i,device,form="train",debug=args.debug)) for i in range(1,num_buckets)] 
        clear10_train = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_train))
        clear10_test = [(i,CLEAR10MOCO(root_dir,i,device,form="test",debug=args.debug)) for i in range(1,num_buckets)]
        clear10_test = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_test))

        # streaming
        clear10_all = [(i,CLEAR10MOCO(root_dir,i,device,form="all",debug=args.debug)) for i in range(1,num_buckets)]
        clear10_all = list(map(lambda x: AvalancheDataset(x[0], task_labels=x[1]), clear10_all))
    
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
                 args.eval_mb_size, args.buffer_size[0], args.lr, args.in_features, args.num_classes, 
                 alpha=args.alpha, 
                 setting="iid") 
      train_eval(save_path, exp_time, num_buckets, streaming_scenario, 
                 strategy, device, args.is_pretrained, args.train_mb_size, args.train_epochs, 
                 args.eval_mb_size, args.buffer_size[1], args.lr, args.in_features, args.num_classes, 
                 alpha=args.alpha, 
                 setting="streaming")





# Permute MNIST Example

'''
# creating the benchmark instance (scenario object)
perm_mnist = PermutedMNIST(
  n_experiences=3,
  seed=1234,
)

# recovering the train and test streams
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# iterating over the train stream
for experience in train_stream:
  print("Start of task ", experience.task_label)
  print('Classes in this task:', experience.classes_in_this_experience)

  # The current Pytorch training set can be easily recovered through the
  # experience
  current_training_set = experience.dataset
  # ...as well as the task_label
  print('Task {}'.format(experience.task_label))
  print('This task contains', len(current_training_set), 'training examples')

  # we can recover the corresponding test experience in the test stream
  current_test_set = test_stream[experience.current_experience].dataset
  print('This task contains', len(current_test_set), 'test examples')

'''

# full example
'''
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

scenario = SplitMNIST(n_experiences=5)

# MODEL CREATION
model = SimpleMLP(num_classes=scenario.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(scenario.test_stream))
'''