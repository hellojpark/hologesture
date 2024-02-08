import numpy as np
import torch

from torch.utils.data import DataLoader
import imgaug.augmenters as iaa

# Import Datasets
from datasets.Hololens import Hololens
from models.model_utilizer import ModuleUtilizer

# Import Model
from models.temporal import GestureTransoformer

# Import Utils
from tqdm import tqdm
from utils.average_meter import AverageMeter

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

class GestureTest(object):
    """Gesture Recognition Test class

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
        self.rgb_specific_path = self.configer.get("data", "rgb_specific_path")
        self.rgb_normal_type = self.configer.get("data", "rgb_normal_type")
        self.depth_normal_type = self.configer.get("data", "depth_normal_type")
        self.depth_specific_path = self.configer.get("data", "depth_specific_path")
        self.rgb_width = self.configer.get("data", "rgb_width")
        self.rgb_height = self.configer.get("data", "rgb_height")
        self.depth_width = self.configer.get("data", "depth_width")
        self.depth_height = self.configer.get("data", "depth_height")
        
        self.data_path = configer.get("data", "data_path")
        self.crop_path = configer.get("data", "crop_path")      #: str: Path to data directory

        # Train val and test accuracy
        self.accuracy = AverageMeter()

        # DataLoaders
        self.data_loader = None

        # Module load and save utility
        self.device = self.configer.get("device")
        self.model_utility = ModuleUtilizer(self.configer)      #: Model utility for load, save and update optimizer
        self.net = None

        # Training procedure
        self.transforms = None

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

    def init_model(self):
        """Initialize model and other data for procedure"""

        self.depth_in_planes = 1
        self.rgb_in_planes = 3

        # Selecting correct model and normalization variable based on type variable
        self.net = GestureTransoformer(self.configer, self.backbone, self.rgb_in_planes, self.depth_in_planes, self.n_classes,
                                       pretrained=self.configer.get("network", "pretrained"),
                                       n_head=self.configer.get("network", "n_head"),
                                       dropout_backbone=self.configer.get("network", "dropout2d"),
                                       dropout_transformer=self.configer.get("network", "dropout1d"),
                                       dff=self.configer.get("network", "ff_size"),
                                       n_module=self.configer.get("network", "n_module")
                                       )

        self.net, _, _, _ = self.model_utility.load_net(self.net)

        # Selecting Dataset and DataLoader
      
        Dataset = Hololens
           
        # Setting Dataloaders
        self.data_loader = DataLoader(
            Dataset(self.configer, self.rgb_normal_type, self.depth_normal_type, self.rgb_specific_path, self.depth_specific_path, self.rgb_width, self.rgb_height, self.depth_width, self.depth_height,
                    self.data_path, self.crop_path, split="test", data_type=self.data_type, transforms=self.transforms, n_frames=self.clip_length,
                    optical_flow=self.optical_flow),
            batch_size=1, shuffle=False, drop_last=True,
            num_workers=self.configer.get('solver', 'workers'), pin_memory=True, worker_init_fn=worker_init_fn)

    def __test(self):
        """Testing function."""
        self.net.eval()
        c = 0
        tot = 0
        # check_predicted = []
        with torch.no_grad():
            for rgb, depth, rgb_label, depth_label in tqdm(self.data_loader, desc="Test"):
                """
                input, gt
                """
                rgb_inputs = rgb.to(self.device)
                rgb_gt = rgb_label.to(self.device)

                depth_inputs = depth.to(self.device)
                gt = depth_label.to(self.device)

                output = self.net(rgb_inputs, depth_inputs)

                predicted = torch.argmax(output.detach(), dim=1)
                correct = gt.detach().squeeze(dim=1)

                if predicted == correct:
                    c += 1
                    # check_predicted.append(predicted)
                tot += 1

        accuracy = c / tot

        print("Accuracy: {}".format(accuracy))

        # print('predicted : ', check_predicted)

    def test(self):
        self.__test()


    def update_metrics(self, split: str, loss, bs, accuracy=None):
        self.losses[split].update(loss, bs)
        if accuracy is not None:
            self.accuracy[split].update(accuracy, bs)
        if split == "train" and self.iters % self.save_iters == 0:
            # self.tbx_summary.add_scalar('{}_loss'.format(split), self.losses[split].avg, self.iters)
            # self.tbx_summary.add_scalar('{}_accuracy'.format(split), self.accuracy[split].avg, self.iters)
            self.losses[split].reset()
            self.accuracy[split].reset()