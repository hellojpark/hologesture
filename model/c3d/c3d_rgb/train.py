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
import pdb
import imgaug.augmenters as iaa
import wandb


# Setting seeds
def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)

parser = argparse.ArgumentParser()
parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
parser.add_argument('--hypes', default=None, type=str,
                    dest='hypes', help='The file of the hyper parameters.')
parser.add_argument('--number', default=None, type=str)

args = parser.parse_args()
args.device = None
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')
torch.autograd.set_detect_anomaly(True)
configer = Configer(args)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# nEpochs = 100  # Number of epochs for training
nEpochs = configer.get("epochs")
resume_epoch = 0  # Default is 0, change if want to resume
nTestInterval = 20 # Run on test set every nTestInterval epochs
lr = 3e-3 # Learning rate  # original lr = 1e-3


dataset = 'hololens' # Options: hmdb51 or ucf101

num_classes=configer.get("data", "n_classes")

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


save_dir = configer.get("checkpoints", "save_dir")
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset



def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr, num_epochs=nEpochs):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    '''-------Args-------'''
    normal_type = configer.get("data", "normal_type")
    data_path = configer.get("data", "data_path")
    crop_data_path = configer.get("data", "crop_data_path")
    data_type = configer.get("data", "type")
    specific_path = configer.get("data", "specific_path")
    width = configer.get("data", "width")
    height = configer.get("data", "height")
    clip_length = configer.get("data", "n_frames")
    model_utilizer = ModuleUtilizer(configer)

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss().to(device)  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=1e-3)   # original weight_decay = 5e-4(대략 40% val) second : 1e-3(대략 50% val)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    train_iters = 0
    val_iters = 0

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    # train_transforms = iaa.Sequential([
    #             # iaa.Resize((0.8, 1.2)),
    #             # iaa.CropToFixedSize(width=256, height=192),
    #             iaa.Rotate((-15, 15))
    #         ])
    # val_transforms = iaa.CenterCropToFixedSize(256, 192)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(Hololens(configer, normal_type, specific_path, width, height, data_path, crop_data_path=crop_data_path, split='train', data_type=data_type, transforms=None, n_frames=clip_length), batch_size=configer.get("data", "batch_size"), shuffle=True, drop_last = True, num_workers=configer.get('solver', 'workers'), worker_init_fn = worker_init_fn)
    val_dataloader   = DataLoader(Hololens(configer, normal_type, specific_path, width, height, data_path, crop_data_path=crop_data_path, split='val', data_type=data_type, n_frames=clip_length), batch_size=configer.get("data", "batch_size"), drop_last=True, num_workers=configer.get('solver', 'workers'), worker_init_fn=worker_init_fn)
    
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                # inputs = Variable(inputs, requires_grad=True).to(device)
                # labels = Variable(labels).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.squeeze())
                # print(loss)
                # print('loss : ', loss.item())

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    # scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.squeeze())
                if phase == 'train':
                    train_iters +=1
                elif phase =='val':
                    val_iters +=1

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects / trainval_sizes[phase]
            if phase == 'val':
                val_acc = running_corrects / trainval_sizes[phase]

            if phase == 'train':
                wandb.log({
                    'train_acc': float(epoch_acc),
                    'train_loss': float(epoch_loss)
                })                
            else:
                wandb.log({
                    'val_acc': float(epoch_acc),
                    'val_loss': float(epoch_loss)
                })

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        ret = model_utilizer.save(accuracy=val_acc, net=model, optimizer=optimizer, iters=train_iters, epoch = epoch + 1)
        _ = model_utilizer.save_all(accuracy=val_acc, net=model, optimizer=optimizer, iters=train_iters, epoch = epoch + 1)
        if ret < 0:
                print("Got no improvement for {} epochs, current epoch is {}."
                      .format(configer.get("checkpoints", "early_stop"), epoch))
                # break
        
if __name__ == "__main__":
    number = configer.get("number")
    hand_type = configer.get("hand color")
    wandb.login()
    wandb.init(project="c3d_lr_3e3_10step", name="c3d noaug frame-normal {} RGB pretrained_{}".format(hand_type, number))
    train_model()