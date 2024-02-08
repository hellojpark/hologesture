import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from mypath import Path


def DataProcess(dataset='ucf101', split='train', clip_len=16, preproc=False):
    
    root_dir, output_dir = Path.db_dir(dataset)           # root_dir : ../UCF-101_data    /   output_dir : ../ucf101
    folder = os.path.join(output_dir, split)                   # folder : ../ucf101/train
    clip_len = clip_len
    split = split

    # The following three parameters are chosen as described in the paper section 4.1
    resize_height = 128
    resize_width = 171
    crop_size = 112

    if not check_integrity(root_dir):
        raise RuntimeError('Dataset not found or corrupted.' +
                            ' You need to download it from official website.')

    if (not check_preprocess(output_dir)) or preproc:
        print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        preprocess(output_dir, root_dir, resize_height, resize_width)


def check_integrity(root_dir):
    if not os.path.exists(root_dir):
        return False
    else:
        return True

def check_preprocess(output_dir):
    # TODO: Check image size in output_dir
    if not os.path.exists(output_dir):
        return False
    elif not os.path.exists(os.path.join(output_dir, 'train')):
        return False

    for ii, video_class in enumerate(os.listdir(os.path.join(output_dir, 'train'))):
        for video in os.listdir(os.path.join(output_dir, 'train', video_class)):
            video_name = os.path.join(os.path.join(output_dir, 'train', video_class, video),
                                sorted(os.listdir(os.path.join(output_dir, 'train', video_class, video)))[0])
            image = cv2.imread(video_name)
            if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                return False
            else:
                break

        if ii == 10:
            break

    return True

def preprocess(output_dir, root_dir, resize_height, resize_width):
    if not os.path.exists(output_dir):               # output_dir : ../ucf101
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'train'))
        os.mkdir(os.path.join(output_dir, 'val'))
        os.mkdir(os.path.join(output_dir, 'test'))

    # Split train/val/test sets
    for file in os.listdir(root_dir):                # root_dir : ../UCF-101_data    /   file : [ApplyEyeMakeup, ApplyLipstic, ...]
        file_path = os.path.join(root_dir, file)       # file_path : ../UCF-101_data/ApplyEyeMakeup
        video_files = [name for name in os.listdir(file_path)]      # name : ../UCF-101_data/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi    /    video_files : [v_ApplyEyeMakeup_g01_c01.avi, ...]

        train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
        train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

        train_dir = os.path.join(output_dir, 'train', file)        # train_dir : ../ucf101/train/ApplyEyeMakeup
        val_dir = os.path.join(output_dir, 'val', file)
        test_dir = os.path.join(output_dir, 'test', file)

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        for video in train:
            process_video(video, file, train_dir, root_dir, resize_height, resize_width)

        for video in val:
            process_video(video, file, val_dir, root_dir, resize_height, resize_width)

        for video in test:
            process_video(video, file, test_dir, root_dir, resize_height, resize_width)

    print('Preprocessing finished.')

def process_video(video, action_name, save_dir, root_dir, resize_height, resize_width):      # action_name : [ApplyEyeMakeup, ApplyLipstic, ...]     /     save_dir : ../ucf101/train/ApplyEyeMakeup
    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]                    # video_filename : v_ApplyEyeMakeup_g01_c01
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(save_dir, video_filename))

    capture = cv2.VideoCapture(os.path.join(root_dir, action_name, video))

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sure splited video has at least 16 frames
    EXTRACT_FREQUENCY = 4
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True

    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1

    # Release the VideoCapture once it is no longer needed
    capture.release()



if __name__ == "__main__":
    DataProcess(dataset='ucf101', split='train', clip_len=8, preproc=False)
    DataProcess(dataset='ucf101', split='val', clip_len=8, preproc=False)
    DataProcess(dataset='ucf101', split='test', clip_len=8, preproc=False)
