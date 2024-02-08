import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

from datasets.utils.read_data import load_split_nvgesture, load_data_from_file
from datasets.utils.normals import normals_multi
from datasets.utils.normalize import video_std, frame_std, video_min_max_norm, frame_min_max_norm, video_img_net_std, frame_img_net_std

from pathlib import Path

class Hololens(Dataset):
    """Hololens2 Dataset class"""
    def __init__(self, configer, rgb_normal_type, depth_normal_type, rgb_specific_path, depth_specific_path, rgb_width, rgb_height, depth_width, depth_height, data_path, crop_path, split="train", data_type="depth", transforms=None, n_frames=40, optical_flow=False):
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

        self.rgb_specific_path = rgb_specific_path
        self.rgb_normal_type = rgb_normal_type
        self.depth_normal_type = depth_normal_type
        self.depth_specific_path = depth_specific_path
        self.data_path = Path(data_path)
        self.crop_path = Path(crop_path)
        self.split = split
        self.data_type = data_type
        self.transforms = transforms
        self.optical_flow = optical_flow
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.depth_width = depth_width
        self.depth_height = depth_height

        rgb_file_lists = self.crop_path / \
                     "hololens_{}_correct_cvpr2016_v2.lst".format(self.split)
        
        depth_file_lists = self.crop_path / \
                     "hololens_{}_correct_cvpr2016_v2.lst".format(self.split)

        self.rgb_data_list = list()
        load_split_nvgesture(file_with_split=str(rgb_file_lists), specific_path=self.rgb_specific_path, list_split=self.rgb_data_list)
        self.depth_data_list = list()
        load_split_nvgesture(file_with_split=str(depth_file_lists), specific_path=self.depth_specific_path, list_split=self.depth_data_list)

        print("done.")

    def __len__(self):
        return len(self.rgb_data_list)

    def __getitem__(self, idx):
        
        rgb_data, rgb_label, offsets = load_data_from_file(self.data_path, self.rgb_specific_path, self.depth_specific_path, example_config=self.rgb_data_list[idx], sensor="color",
                                          image_width=self.rgb_width, image_height=self.rgb_height)
       
        rgb_data = rgb_data[..., [*range(0, rgb_data.shape[-1], 2)]]  # Our settings is working with static clip containing 40 frames

        if self.rgb_normal_type=='video_normal':
            rgb_data=video_min_max_norm(rgb_data)
        elif self.rgb_normal_type=='frame_normal':
            rgb_data=frame_min_max_norm(rgb_data)
        elif self.rgb_normal_type == 'video_std':
            rgb_data=video_std(rgb_data)
        elif self.rgb_normal_type == 'frame_std':
            rgb_data=frame_std(rgb_data)
        elif self.rgb_normal_type == 'video_imgnet_std':
            rgb_data=video_img_net_std(rgb_data)
        elif self.rgb_normal_type == 'frame_imgnet_std':
            rgb_data=frame_img_net_std(rgb_data)

        # if self.transforms is not None:
        #     aug_det = self.transforms.to_deterministic()
        #     rgb_data = np.array([aug_det.augment_image(rgb_data[..., i]) for i in range(rgb_data.shape[-1])]).transpose(1, 2, 3, 0)

        rgb_data = np.concatenate(rgb_data.transpose(3, 0, 1, 2), axis=2).transpose(2, 0, 1)
        rgb_data = torch.from_numpy(rgb_data)
        rgb_label = torch.LongTensor(np.asarray([rgb_label]))

        
        depth_data, depth_label, offsets = load_data_from_file(self.data_path, self.rgb_specific_path, self.depth_specific_path, example_config=self.depth_data_list[idx], sensor="depth",
                                          image_width=self.depth_width, image_height=self.depth_height)
       
        depth_data = depth_data[..., [*range(0, depth_data.shape[-1], 2)]]  # Our settings is working with static clip containing 40 frames

        if self.depth_normal_type=='video_normal':
            depth_data=video_min_max_norm(depth_data)
        elif self.depth_normal_type=='frame_normal':
            depth_data=frame_min_max_norm(depth_data)
        elif self.depth_normal_type == 'video_std':
            depth_data=video_std(depth_data)
        elif self.depth_normal_type == 'frame_std':
            depth_data=frame_std(depth_data)

        # if self.transforms is not None:
        #     aug_det = self.transforms.to_deterministic()
        #     depth_data = np.array([aug_det.augment_image(depth_data[..., i]) for i in range(depth_data.shape[-1])]).transpose(1, 2, 3, 0)
        
        depth_data = np.concatenate(depth_data.transpose(3, 0, 1, 2), axis=2).transpose(2, 0, 1)
        depth_data = torch.from_numpy(depth_data)
        depth_label = torch.LongTensor(np.asarray([depth_label]))

        return rgb_data.float(), depth_data.float(), rgb_label, depth_label