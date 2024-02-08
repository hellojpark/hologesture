from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

# Import Datasets
from datasets.Hololens import Hololens
from models.model_utilizer import ModuleUtilizer

# Import Model
from models.temporal import GestureTransoformer
from torch.optim.lr_scheduler import MultiStepLR

# Import loss

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter

import wandb



# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class GestureTrainer(object):
    """Gesture Recognition Train class

    Attributes:
        configer (Configer): Configer object, contains procedure configuration.
        train_loader (torch.utils.data.DataLoader): Train data loader variable
        val_loader (torch.utils.data.DataLoader): Val data loader variable
        test_loader (torch.utils.data.DataLoader): Test data loader variable
        net (torch.nn.Module): Network used for the current procedure
        lr (int): Learning rate value
        optimizer (torch.nn.optim.optimizer): Optimizer for training procedure
        iters (int): Starting iteration number, not zero if resuming training
        epoch (int): Starting epoch number, not zero if resuming training
        scheduler (torch.optim.lr_scheduler): Scheduler to utilize during training

    """

    def __init__(self, configer):
        self.configer = configer
        self.specific_path = self.configer.get("data", "specific_path")
        self.normal_type = self.configer.get("data", "normal_type")
        self.width = self.configer.get("data", "width")
        self.height = self.configer.get("data", "height")

        self.data_path = configer.get("data", "data_path")      #: str: Path to data directory
        self.crop_path = configer.get("data", "crop_path")

        # Losses
        self.losses = {
            'train': AverageMeter(),                      #: Train loss avg meter
            'val': AverageMeter(),                        #: Val loss avg meter
            'test': AverageMeter()                        #: Test loss avg meter
        }

        # Train val and test accuracy
        self.accuracy = {
            'train': AverageMeter(),                      #: Train accuracy avg meter
            'val': AverageMeter(),                        #: Val accuracy avg meter
            'test': AverageMeter()                        #: Test accuracy avg meter
        }

        # DataLoaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None
        self.lr = None

        # Training procedure
        self.optimizer = None
        self.iters = None
        self.epoch = 0
        self.train_transforms = None
        self.val_transforms = None
        self.loss = None

        self.save_iters = self.configer.get('checkpoints', 'save_iters')    #: int: Saving ratio

        # Other useful data
        self.backbone = self.configer.get("network", "backbone")     #: str: Backbone type
        self.in_planes = None                                       #: int: Input channels
        self.clip_length = self.configer.get("data", "n_frames")    #: int: Number of frames per sequence
        self.n_classes = self.configer.get("data", "n_classes")     #: int: Total number of classes for dataset
        self.data_type = self.configer.get("data", "type")          #: str: Type of data (rgb, depth, ir, leapmotion)
        self.dataset = self.configer.get("dataset").lower()         #: str: Type of dataset
        self.optical_flow = self.configer.get("data", "optical_flow")
        if self.optical_flow is None:
            self.optical_flow = True
        self.scheduler = None
        self.number = self.configer.get("number")
        self.hand_type = self.configer.get("hand_type")

    def init_model(self):
        """Initialize model and other data for procedure"""

        self.in_planes = 3

        self.loss = nn.CrossEntropyLoss().to(self.device)

        # Selecting correct model and normalization variable based on type variable
        self.net = GestureTransoformer(self.backbone, self.in_planes, self.n_classes,
                                       pretrained=self.configer.get("network", "pretrained"),
                                       n_head=self.configer.get("network", "n_head"),
                                       dropout_backbone=self.configer.get("network", "dropout2d"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module")
                                       )

        # Initializing training
        self.iters = 0
        self.epoch = None
        phase = self.configer.get('phase')

        # Starting or resuming procedure
        if phase == 'train':
            self.net, self.iters, self.epoch, optim_dict = self.model_utility.load_net(self.net)
        else:
            raise ValueError('Phase: {} is not valid.'.format(phase))

        if self.epoch is None:
            self.epoch = 0

        # ToDo Restore optimizer and scheduler from checkpoint
        self.optimizer, self.lr = self.model_utility.update_optimizer(self.net, self.iters)
        self.scheduler = MultiStepLR(self.optimizer, self.configer["solver", "decay_steps"], gamma=0.1)

        #  Resuming training, restoring optimizer value
        if optim_dict is not None:
            print("Resuming training from epoch {}.".format(self.epoch))
            self.optimizer.load_state_dict(optim_dict)

        # Selecting Dataset and DataLoader
       

        Dataset = Hololens
       
        # Setting Dataloaders
        self.train_loader = DataLoader(
            Dataset(self.configer, self.normal_type, self.specific_path, self.width, self.height, self.data_path, self.crop_path, split="train", data_type=self.data_type,
                   transforms=self.train_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=True, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        self.val_loader = DataLoader(
            Dataset(self.configer, self.normal_type, self.specific_path, self.width, self.height, self.data_path, self.crop_path, split="val", data_type=self.data_type,
                    transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
            batch_size=self.configer.get('data', 'batch_size'), shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        if self.dataset == "hololens":
            self.test_loader = None
        else:
            self.test_loader = DataLoader(
                Dataset(self.configer, self.normal_type, self.specific_path, self.width, self.height, self.data_path, self.crop_path, split="test", data_type=self.data_type,
                        transforms=self.val_transforms, n_frames=self.clip_length, optical_flow=self.optical_flow),
                batch_size=1, shuffle=False, drop_last=True,
                num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)
        
        wandb.login()
        wandb.init(project="save all random approach check", name="50 epoch randomseed frame-normal {} rgb_noaug {}".format(self.hand_type, self.number))

    def __train(self):
        """Train function for every epoch."""

        self.net.train()
        for data_tuple in tqdm(self.train_loader, desc="Train"):
            """
            input, gt
            """
            inputs = data_tuple[0].to(self.device)
            gt = data_tuple[1].to(self.device)

            output = self.net(inputs)

            self.optimizer.zero_grad()
            loss = self.loss(output, gt.squeeze(dim=1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            predicted = torch.argmax(output.detach(), dim=1)
            correct = gt.detach().squeeze(dim=1)

            self.iters += 1
            self.update_metrics("train", loss.item(), inputs.size(0),
                                float((predicted==correct).sum()) / len(correct))
        accuracy = self.accuracy["train"].avg
        print(f'train avg accruacy : {accuracy}')
        wandb.log({
            'train_acc': float(self.accuracy["train"].avg),
            'train_loss': float(self.losses["train"].avg)
        })
        self.accuracy["train"].reset()
        self.losses["train"].reset()


    def __val(self):
        """Validation function."""
        self.net.eval()

        with torch.no_grad():
            # for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=str(self.accuracy["val"].avg))):
            for i, data_tuple in enumerate(tqdm(self.val_loader, desc="Val", postfix=""+str(np.random.randint(200)))):
                """
                input, gt
                """
                inputs = data_tuple[0].to(self.device)
                gt = data_tuple[1].to(self.device)

                output = self.net(inputs)
                loss = self.loss(output, gt.squeeze(dim=1))

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                self.iters += 1
                self.update_metrics("val", loss.item(), inputs.size(0),
                                    float((predicted == correct).sum()) / len(correct))

        accuracy = self.accuracy["val"].avg
        print(f'valid avg accuracy : {accuracy}')
        wandb.log({
            'val_acc': float(self.accuracy["val"].avg),
            'val_loss': float(self.losses["val"].avg)
        })
        self.accuracy["val"].reset()
        self.losses["val"].reset()

        ret = self.model_utility.save(accuracy, self.net, self.optimizer, self.iters, self.epoch + 1)
        _ = self.model_utility.save_all(accuracy, self.net, self.optimizer, self.iters, self.epoch + 1)
        if ret < 0:
            return -1
        elif ret > 0 and self.test_loader is not None:
            self.__test()
        return ret

    def train(self):
        # for n in range(self.configer.get("epochs")):
        for n in range(self.epoch+1, self.configer.get("epochs")):
            print("Starting epoch {}".format(self.epoch + 1))
            self.__train()
            ret = self.__val()
            print('train accuracy : ', self.accuracy["train"].avg)
            if ret < 0:
                print("Got no improvement for {} epochs, current epoch is {}."
                      .format(self.configer.get("checkpoints", "early_stop"), n))
                # break
            self.epoch += 1

    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)