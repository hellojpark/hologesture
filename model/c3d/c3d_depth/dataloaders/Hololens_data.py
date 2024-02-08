import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

from dataloaders.utils.read_data import load_split_nvgesture, load_data_from_file
from dataloaders.utils.normals import normals_multi
from dataloaders.utils.normalize import video_std, frame_std, video_min_max_norm, frame_min_max_norm

from pathlib import Path

class Hololens(Dataset):
    """Hololens2 Dataset class"""
    def __init__(self, configer, normal_type, specific_path, width, height, path, crop_data_path, split="train", data_type="depth", transforms=None, n_frames=40, optical_flow=False):
        """Constructor method for NVGesture Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()

        print("Loading Hololens {} dataset...".format(split.upper()), end=" ")

        # self.configer = configer
        self.specific_path = specific_path
        self.normal_type = normal_type
        self.dataset_path = Path(path)
        self.crop_dataset_path = Path(crop_data_path)
        self.split = split
        self.data_type = data_type
        self.transforms = transforms
        self.optical_flow = optical_flow
        self.image_width = width
        self.image_height = height

        file_lists = self.crop_dataset_path / \
                     "hololens_{}_correct_cvpr2016_v2.lst".format(self.split)

        self.data_list = list()
        load_split_nvgesture(file_with_split=str(file_lists), specific_path=self.specific_path, list_split=self.data_list)

        self.sensor = "depth"
        
        print("done.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, label, offsets = load_data_from_file(self.dataset_path, example_config=self.data_list[idx], specific_path=self.specific_path, sensor=self.sensor,
                                          image_width=self.image_width, image_height=self.image_height)
        # (288, 320, 1, 40)
       
        data = data.astype(np.float32)
        
        data = data[..., [*range(0, data.shape[-1], 2)]]  # Our settings is working with static clip containing 40 frames
        # (288, 320, 1, 20)

        # import matplotlib.pyplot as plt
        # plt.title('dataset class level before normalization')
        # plt.imshow(data[:,:,:,0])
        # plt.show()

        # data shape : (112, 112, 1, 20)
        if self.normal_type=='video_normal':
            data=video_min_max_norm(data)
        elif self.normal_type=='frame_normal':
            data=frame_min_max_norm(data)
        elif self.normal_type == 'video_stand':
            data=video_std(data)
        elif self.normal_type == 'frame_stand':
            data=frame_std(data)
        # if self.transforms is not None:
        #     aug_det = self.transforms.to_deterministic()
        #     data = np.array([aug_det.augment_image(data[..., i]) for i in range(data.shape[-1])]).transpose(1, 2, 3, 0)
            
        # plt.title('after normalization')
        # plt.imshow(data[:,:,:,0])
        # plt.show()
        
        data = data.transpose(3, 0, 1, 2)   # (20, 288, 320, 1)
        data = data.astype(np.float32)
        # for frm in range(data.shape[0]):
        #     print('data mean : ', np.array([np.mean(data[frm,...])], np.float32))
        # print('data shape 1 : ', data.shape)
        data = np.concatenate((data, data, data), axis=3).transpose(3,0,1,2)   # (20, 112, 112, 1) => (3, 20, 112, 112)
        # print('data shape 2 : ', data.shape)
       
        # for frm in range(data.shape[1]):
        #     print('data mean1 : ', np.array([np.mean(data[:, frm, ...])], np.float32))
        #     print('data mean2 : ', np.array([np.mean(data[0, frm, ...])], np.float32))
        #     print('data mean3 : ', np.array([np.mean(data[1, frm, ...])], np.float32))


        # data = np.concatenate(data.transpose(3, 0, 1, 2), axis=2).transpose(2, 0, 1)    # (288, 320, 3, 20) => (288, 320, 60) => (60, 288, 320)
        data = torch.from_numpy(data)
        label = torch.LongTensor(np.asarray([label]))
        
        # print('data dtype : ', data.dtype)
        # plt.title('before return')
        # plt.imshow(data[:,0,:,:].transpose(0,2).transpose(0,1))
        # plt.show()

        return data, label