import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
# from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from save_utilizer import ModuleUtilizer

from dataloaders.Hololens_data import Hololens
from network import C3D_model, R2Plus1D_model, R3D_model
from util.configer import Configer
import argparse
# import imgaug.augmenters as iaa

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
train_log_dir = configer.get("checkpoints", "tb_path")

modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    '''-------Args-------'''
    data_path = configer.get("data", "data_path")
    normal_type = configer.get("data", "normal_type")
    crop_data_path = configer.get("data", "crop_data_path")
    data_type = configer.get("data", "type")
    specific_path = configer.get("data", "specific_path")
    width = configer.get("data", "width")
    height = configer.get("data", "height")
    clip_length = configer.get("data", "n_frames")
    model_utilizer = ModuleUtilizer(configer)

    
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
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
    log_dir = os.path.join(train_log_dir)
    # writer = SummaryWriter(log_dir=log_dir)

    # transforms = iaa.CenterCropToFixedSize(256, 192)
    
    test_dataloader  = DataLoader(Hololens(configer, normal_type, specific_path, width, height, data_path, crop_data_path=crop_data_path, split='test', data_type=data_type, n_frames=clip_length), batch_size=2, drop_last=True, num_workers=configer.get('solver', 'workers'), worker_init_fn=worker_init_fn)

    test_size = len(test_dataloader.dataset)
            
    model.eval()
    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0.0
    num = 0

    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        # print('1 : ', labels.data.shape)
        # print('2 : ', preds.shape)
        loss = criterion(outputs, labels.data.squeeze())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data.squeeze())
        # print(preds.shape)
        # print(labels.data.squeeze().shape)
        # print('test : ', preds == labels.data)
        # print('preds num : ', preds)
        # print('label num : ', labels.data)
        # print('check : ', torch.sum(preds==labels.data))
        # print('matching num : ', len(preds==labels.data))
        num +=1
    print('iter num : ', num)
    print('corrects number : ', running_corrects)
    print('test size : ', test_size)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects / test_size

    # writer.add_scalar('data/test_loss_epoch', epoch_loss)
    # writer.add_scalar('data/test_acc_epoch', epoch_acc)

    print("[test] Epoch: Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    # writer.close()


if __name__ == "__main__":
    test_model()