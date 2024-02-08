import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from save_utilizer import ModuleUtilizer

from dataloaders.Hololens_data import Hololens
from network import C3D_model, R2Plus1D_model, R3D_model
from util.configer import Configer
import argparse
import imgaug.augmenters as iaa

# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
parser.add_argument('--hypes', default=None, type=str,
                    dest='hypes', help='The file of the hyper parameters.')
parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
parser.add_argument('--fusionkind', default=None, type=str)

args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
torch.autograd.set_detect_anomaly(True)
configer = Configer(args)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs


dataset = 'hololens'
num_classes=configer.get("data", "n_classes")

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


save_dir = configer.get("checkpoints", "save_dir")
# train_log_dir = configer.get("checkpoints", "tb_path")

modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    '''-------Args-------'''
    rgb_data_path = configer.get("data", "rgb_data_path")
    depth_data_path = configer.get("data", "depth_data_path")
    crop_data_path = configer.get("data", "crop_data_path")
    data_type = configer.get("data", "rgb_type")
    rgb_specific_path = configer.get("data", "rgb_specific_path")
    depth_specific_path = configer.get("data", "depth_specific_path")
    rgb_width = configer.get("data", "rgb_width")
    rgb_height = configer.get("data", "rgb_height")
    depth_width = configer.get("data", "depth_width")
    depth_height = configer.get("data", "depth_height")
    clip_length = configer.get("data", "n_frames")
    normal_type = configer.get("data", "normal_type")
    model_utilizer = ModuleUtilizer(configer)

    
    model = C3D_model.C3D(configer, num_classes=num_classes, pretrained=False)
    criterion = nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification

    model, _, _, _ = model_utilizer.load_net(model)
    # checkpoint = torch.load(os.path.join(save_dir + '/best_30odo_depth_train.pth'), map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    # print("Initializing weights from: {}...".format(
    #     os.path.join(save_dir + 'best_30odo_depth_train.pth')))
    # model.load_state_dict(checkpoint['state_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    # log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    
    # transforms = iaa.CenterCropToFixedSize(196, 132)
    
    test_dataloader  = DataLoader(Hololens(configer, normal_type, rgb_specific_path, depth_specific_path, rgb_width, rgb_height, depth_width, depth_height,
     rgb_data_path, depth_data_path, crop_data_path=crop_data_path, split='test', data_type=data_type, n_frames=clip_length),
       batch_size=2, drop_last=True, num_workers=configer.get('solver', 'workers'), worker_init_fn=worker_init_fn)

    test_size = len(test_dataloader.dataset)
            
    model.eval()
    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0.0
    num = 0

    for rgb_inputs, depth_inputs, rgb_labels, depth_labels in tqdm(test_dataloader):
        rgb_inputs = rgb_inputs.to(device)
        rgb_labels = rgb_labels.to(device)
        depth_inputs = depth_inputs.to(device)
        depth_labels = depth_labels.to(device)

        with torch.no_grad():
            outputs = model(rgb_inputs, depth_inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, depth_labels.data.squeeze())

        running_loss += loss.item() * depth_inputs.size(0)
        running_corrects += torch.sum(preds == depth_labels.data.squeeze())
        # print(preds.shape)
        # print(labels.data.squeeze().shape)
        # print('test : ', preds == labels.data)
        # print('preds num : ', preds)
        # print('label num : ', labels.data)
        # print('check : ', torch.sum(preds==labels.data))
        # print('matching num : ', len(preds==labels.data))
        num +=1
    # print('iter num : ', num)
    # print('corrects number : ', running_corrects)
    # print('test size : ', test_size)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects / test_size

    print("[test] Epoch: Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")



if __name__ == "__main__":
    test_model()