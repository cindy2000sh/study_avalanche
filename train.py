import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, DataParallel, Linear
from torchvision.models import resnet18
from avalanche.training.strategies import Naive, CWRStar, Replay, GDumb,  \
                                            LwF, AGEM, EWC, \
                                            SynapticIntelligence
from avalanche.evaluation.metrics import accuracy_metrics,timing_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.storage_policy import ExemplarsBuffer
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheSubset
from metric import *


class BiasedReservoirSamplingBuffer(ExemplarsBuffer):
    """ Buffer updated with reservoir sampling. """

    def __init__(self, max_size: int, alpha, mode):
        super().__init__(max_size)
        self.alpha = alpha
        self.mode = mode

    def update(self, strategy):
        """ Update buffer. """
        self.update_from_dataset(strategy.experience.dataset, strategy.experience.task_label, self.alpha)

    def update_from_dataset(self, new_data, time_stamp, alpha, mode): 
        """Update the buffer using the given dataset."""
        new_weights = torch.rand(len(new_data))
        for p_id in new_weights:
            if mode == "dynamic":
                threshold = alpha * self.max_size/time_stamp
            else: # fixed
                threshold = alpha
            if new_weights[p_id] <= threshold:
                self.buffer = AvalancheConcatDataset([new_data[p_id], self.buffer])
        self.buffer = AvalancheSubset(self.buffer, range(self.max_size))

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
        return CWRStar(model, optimizer, criterion, train_mb_size=train_mb_size, 
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
        # TODO: AGEM with reservoir sampling?
        # if adding plugin, will override agem plugin
        # still called agem?
        return AGEM(model, optimizer, criterion, buffer_size, buffer_size, train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                    plugins=[LRSchedulerPlugin(scheduler)], evaluator=eval_plugin)
    elif name == "Naive":
        return Naive(model, optimizer, criterion, train_mb_size=train_mb_size, 
                    train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                    plugins=[LRSchedulerPlugin(scheduler), CustomReplay(ReservoirSamplingBuffer(max_size=buffer_size))], 
                    evaluator=eval_plugin)
    elif name == "NaiveBiased":
        if alpha is None:
            raise ValueError('Missing input alpha for biased reservoir sampling')
        else: # TODO: default dynamic, should add one more input, or change Naive customreplay plugin
            return Naive(model, optimizer, criterion, train_mb_size=train_mb_size, 
                         train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device, 
                         plugins=[LRSchedulerPlugin(scheduler),
                          CustomReplay(BiasedReservoirSamplingBuffer(max_size=buffer_size, alpha=alpha, mode="dynamic"))], 
                         evaluator=eval_plugin)
    else:
        raise ValueError('Unknown Strategy provided. Strategies available: EWC, SI, LwF, CWR, GDumb, ER, AGEM, Naive, NaiveBiased')

def train_eval(save_path, curr_time, num_buckets, scenario, strategy, device, is_pretrained,\
                train_mb_size, train_epochs, eval_mb_size, buffer_size, init_lr, \
                in_features, num_classes, alpha=None, setting="iid") -> None:
    
    scenario.to(device) # TODO: or load data to device? 
    import pdb; pdb.set_trace()
    print(scenario.device)
    
    if torch.cuda.device_count() > 1:
        if is_pretrained:
            model = DataParallel(Linear(in_features, num_classes))
        else:
            model = DataParallel(resnet18(pretrained=False))
    else:
        if is_pretrained: 
            model = Linear(in_features,num_classes)
        else:
            model = resnet18(pretrained=False)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=init_lr, momentum=0.9)
    criterion = CrossEntropyLoss()
    scheduler = lr_scheduler.CyclicLR(optimizer, init_lr, 0.1)

    if setting == "iid": 
        eval_plugin_iid = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                    timing_metrics(epoch=True),
                                    loggers=[InteractiveLogger(), TextLogger(open(save_path+'/log.txt', 'a'))],
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
                import pdb; pdb.set_trace()
                results[exp_id,j] = iid_strat.eval(test_stream[exp_id])['accuracy_metrics']
        
        # TODO: check if it's really accuracy from the dict
        visualize(results, save_path) 
        print(f"in-domain: {in_domain(results)},  \
            backward_transfer: {backward_transfer(results)}, \
            forward_transfer: {forward_transfer(results)}")
    
    else:   
        eval_plugin_stm = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True),
                                    timing_metrics(epoch=True),
                                    loggers=[InteractiveLogger(), TextLogger(open(save_path+'/log.txt', 'a'))],
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
                results[exp_id,j] = stm_strat.eval(test_stream[exp_id])['accuracy_metrics'] 
        
        visualize(results, save_path, partial=True)
        print(f"in-domain: {next_domain(results)},  \
            forward_transfer: {forward_transfer(results)}")
    
    torch.save(model.state_dict(), save_path+f"/models/{curr_time}_{strategy}.pth")
    
    return