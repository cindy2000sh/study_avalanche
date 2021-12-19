import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchvision.models import resnet18
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb,  \
                                            LwF, AGEM, EWC, \
                                            SynapticIntelligence
from avalanche.evaluation.metrics import accuracy_metrics,timing_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.storage_policy import ExemplarsBuffer
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheConcatDataset, AvalancheSubset
from metric import *
from dataset import SubDataset



class BiasedReservoirSamplingBuffer(ExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, alpha, mode):
        super().__init__(max_size)
        self.alpha = alpha
        self.mode = mode

    def update(self, strategy):
        """ Update buffer. """
        self.update_from_dataset(strategy.experience.dataset, 
                                strategy.experience.task_label, 
                                self.alpha, self.mode)

    def update_from_dataset(self, new_data, time_stamp, alpha, mode): 
        """Update the buffer using the given dataset."""
        valid = []
        while(len(valid) == 0):
            new_weights = torch.rand(len(new_data))
            new_weights_enum = [(i,new_weights[i]) for i in range(len(new_weights))]
            if mode == "dynamic":
                if time_stamp == 0:
                    threshold = 1 # normal reservoir sampling. no need to bias threshold for the first bucket
                else:
                    threshold = alpha * self.max_size/time_stamp
            else: # fixed
                threshold = alpha
            valid = list(filter(lambda x: x[1] <= threshold, new_weights_enum))
            valid = list(map(lambda x: x[0], valid))
        if len(new_data) == 0:
            import pdb; pdb.set_trace()
        else:
            new_concat = AvalancheDataset(SubDataset(new_data, valid), task_labels=new_data[0][-1])
            if len(new_concat) + len(self.buffer) > self.max_size:
                max_tmp = self.max_size
                self.resize(len(new_concat) + len(self.buffer))
                self.buffer = AvalancheConcatDataset([new_concat, self.buffer])
                self.resize(max_tmp)
            else:
                self.buffer = AvalancheConcatDataset([new_concat, self.buffer])

    def resize(self, new_size):
        """ Update the maximum size of the buffer. """
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = AvalancheSubset(self.buffer, torch.arange(self.max_size))


class CustomReplay(StrategyPlugin):
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy,
                            num_workers = 0, shuffle = True):
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset, # data used to train
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        print("Buffer update.")
        self.storage_policy.update(strategy)


def create_strategy(name, model, optimizer, scheduler, criterion, eval_plugin, num_buckets, \
                    buffer_size, train_mb_size, train_epochs, eval_mb_size, device, \
                    alpha=None):
    if name == "EWC":
        return EWC(model, optimizer, criterion, ewc_lambda=0.001, train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                    plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "SI":
        return SynapticIntelligence(model, optimizer, criterion, si_lambda=0.0001, train_mb_size=train_mb_size, 
                                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                                    plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "LwF":
        return LwF(model, optimizer, criterion, alpha=np.linspace(0,2,num=num_buckets).tolist(), train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, temperature=1, 
                    plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "CWR":
        return CWRStar(model, optimizer, criterion, cwr_layer_name=None, train_mb_size=train_mb_size, 
                       train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                       plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "GDumb":
        return GDumb(model, optimizer, criterion, buffer_size, train_mb_size=train_mb_size, 
                     train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                     plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "ER":
        return Replay(model, optimizer, criterion, buffer_size, train_mb_size=train_mb_size, 
                      train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                      plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "AGEM": 
        # TODO: See: https://github.com/ElvishElvis/Continual-Learning/blob/master/train.py
        return AGEM(model, optimizer, criterion, buffer_size, buffer_size, train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                    plugins=[LRSchedulerPlugin(scheduler), CustomReplay(BiasedReservoirSamplingBuffer(max_size=buffer_size, alpha=alpha, mode="fixed"))], evaluator=eval_plugin)
    elif name == "Naive":
        return Naive(model, optimizer, criterion, train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                    plugins=[LRSchedulerPlugin(scheduler), CustomReplay(BiasedReservoirSamplingBuffer(max_size=buffer_size, alpha=alpha, mode="fixed"))], 
                    evaluator=eval_plugin)
    elif name == "NaiveBiased":
        if alpha is None:
            raise ValueError('Missing input alpha for biased reservoir sampling')
        else: 
            return Naive(model, optimizer, criterion, train_mb_size=train_mb_size, 
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                         plugins=[LRSchedulerPlugin(scheduler),
                          CustomReplay(BiasedReservoirSamplingBuffer(max_size=buffer_size, alpha=alpha, mode="dynamic"))], 
                         evaluator=eval_plugin)
    else:
        raise ValueError('Unknown Strategy provided. Strategies available: EWC,SI,LwF,CWR,GDumb,ER,AGEM,Naive,NaiveBiased')


class LinearReduce(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features,num_classes)
    
    def forward(self, x):
        return (self.linear(x).squeeze(1)) # remove 1 dimension

def train_eval(save_path, curr_time, num_buckets, scenario, strategy, device, is_pretrained,\
                train_mb_size, train_epochs, eval_mb_size, buffer_size, init_lr, \
                in_features, num_classes, alpha=None, setting="iid") -> None:
    
    if torch.cuda.device_count() > 1:
        if is_pretrained:
            model = nn.DataParallel(LinearReduce(in_features, num_classes))
        else:
            model = nn.DataParallel(resnet18(pretrained=False))
    else:
        if is_pretrained: 
            model = LinearReduce(in_features,num_classes)
        else:
            model = resnet18(pretrained=False)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CyclicLR(optimizer, init_lr, 0.1, cycle_momentum=False)

    if setting == "iid": 
        eval_plugin_iid = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                    timing_metrics(epoch=True),
                                    loggers=[InteractiveLogger(), TextLogger(open(save_path+f'/logs/{curr_time}_{strategy}.txt', 'a'))],
                                    benchmark=scenario,
                                    strict_checks=False)

        #iid_buffer_size = 2310
        iid_strat = create_strategy(strategy, model, optimizer, scheduler, criterion, eval_plugin_iid, num_buckets, 
                                    buffer_size, train_mb_size, train_epochs, eval_mb_size, device,
                                    alpha=alpha)
        
        print('Starting iid setting experiment...')
        results = np.zeros((num_buckets,num_buckets)) # acc matrix
        train_stream = scenario.train_stream
        test_stream = scenario.test_stream
        for exp_id in range(len(train_stream)):
            print("Start of experience: ", train_stream[exp_id].current_experience)
            print("Current Classes: ", train_stream[exp_id].classes_in_this_experience)

            current_training_set = train_stream[exp_id].dataset
            print('This task contains', len(current_training_set), 'training examples')

            iid_strat.train(train_stream[exp_id])
            print('Training completed')

            current_test_set = test_stream[train_stream[exp_id].current_experience].dataset
            print('This task contains', len(current_test_set), 'test examples')

            print('Computing accuracy on each bucket')
            for j in range(len(test_stream)):
                expid = format(test_stream[exp_id].current_experience,'03d')
                taskid = format(test_stream[exp_id].task_label,'03d')
                results[exp_id,j] = iid_strat.eval(test_stream[exp_id])[f'Top1_Acc_Exp/eval_phase/test_stream/Task{taskid}/Exp{expid}']
        
        visualize(results, save_path, curr_time, strategy) 
        print(f"strategy: {strategy}, in-domain: {in_domain(results)},  \
            backward_transfer: {backward_transfer(results)}, \
            forward_transfer: {forward_transfer(results)}")
    
    else:   
        eval_plugin_stm = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                    timing_metrics(epoch=True),
                                    loggers=[InteractiveLogger(), TextLogger(open(save_path+f'/logs/{curr_time}_{strategy}.txt', 'a'))],
                                    benchmark=scenario,
                                    strict_checks=False)
        #stm_buffer_size = 3300
        stm_strat = create_strategy(strategy, model, optimizer, scheduler, criterion, eval_plugin_stm, num_buckets, 
                                    buffer_size, train_mb_size, train_epochs, eval_mb_size, device,
                                    alpha=alpha)  

        print('Starting streaming setting experiment...')
        results = np.zeros((num_buckets,num_buckets))
        train_stream = scenario.train_stream
        test_stream = scenario.test_stream
        for exp_id in range(len(train_stream)):
            print("Start of experience: ", train_stream[exp_id].current_experience)
            print("Current Classes: ", train_stream[exp_id].classes_in_this_experience)

            current_training_set = train_stream[exp_id].dataset
            print('This task contains', len(current_training_set), 'training examples')

            stm_strat.train(train_stream[exp_id])
            print('Training completed')

            current_test_set = test_stream[train_stream[exp_id].current_experience].dataset
            print('This task contains', len(current_test_set), 'test examples')

            print('Computing accuracy on future buckets')
            for j in range(len(test_stream[exp_id:])): # train test id offset by 1
                expid = format(test_stream[exp_id].current_experience,'03d')
                taskid = format(test_stream[exp_id].task_label,'03d')
                results[exp_id,j] = stm_strat.eval(test_stream[exp_id])[f'Top1_Acc_Exp/eval_phase/test_stream/Task{taskid}/Exp{expid}']
        
        visualize(results, save_path, curr_time, strategy, partial=True)
        print(f"strategy: {strategy}, next-domain: {next_domain(results)},  \
            forward_transfer: {forward_transfer(results)}")
    
    torch.save(model.state_dict(), save_path+f"/models/{curr_time}_{strategy}_{setting}.pth")
    
    return