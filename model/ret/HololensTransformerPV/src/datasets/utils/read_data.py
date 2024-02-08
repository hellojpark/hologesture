import cv2
import numpy as np
from PIL import Image
import os

from pathlib import Path

def load_split_nvgesture(file_with_split='./hololens_test_correct.lst', specific_path = 'PV_aligned', list_split=list()):
    with open(file_with_split, 'rt') as f:
        dict_name = file_with_split[file_with_split.rfind('\\') + 1:]
        dict_name = dict_name[:dict_name.find('_')] #PV

        for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name #PV

            path = params[0].split(':')[1] +'/'+specific_path # path = .yoon/blue/bare/class1/r0/PV_segment
            for param in params[1:]:        # params[1:] = [depth:sk_depth:48:88, label:1]
                parsed = param.split(':')   # parsed = [depth, sk_depth, 48, 88] 아니면 [label, 1]
                key = parsed[0]             # key = depth 아니면 label
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth'):
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]     # params_dictionary[depth] = .yoon/blue/bare/class1/r0/PV_segment/sk_depth
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])  # params_dictionary[depth_start] = 48

                    params_dictionary[key + '_end'] = int(parsed[3])    # params_dictionary[depth_start] = 88

            list_split.append(params_dictionary)

    return list_split


def load_data_from_file(data_path, example_config, specific_path, sensor, image_width, image_height, nogesture = False):
    path = example_config['depth']                        #exmaple_config[depth] = ./subject2/bare/class1/r3/depth/sk_depth
    path = Path(data_path) / path[path.find('/') + 1:]   #c:/Users/User/Desktop/hololens_gesture_t/subject2/bare/class1/r3/depth/sk_depth
  
    start_frame = example_config['depth_start']
    end_frame = example_config['depth_end']
    label = example_config['label']

    chnum = 3

    # print('height : ', image_width)

    video_container = np.zeros((image_height, image_width, chnum, 40), dtype=np.uint8)
    if specific_path == 'depth_based_rgb':
        video_container = np.zeros((int(152), int(320), chnum, 40), dtype=np.uint8)
   

    if end_frame - start_frame > 40:
        new_start = (end_frame - start_frame) // 2 - 20 + start_frame
        new_end = (end_frame - start_frame) // 2 + 20 + start_frame
        start_frame = new_start
        end_frame = new_end
        frames_to_load = range(start_frame, end_frame)  
    elif end_frame - start_frame < 40:
        diff = end_frame-start_frame
        new_start = 20 - diff//2
        new_end = 20 + diff//2
        frames_to_load = range(0,40)
    else:
        frames_to_load = range(start_frame, end_frame)
    
    
    path1 = str(path)
   
    

    j=0

    for indx in frames_to_load:     # 시작 frame부터 끝 frame까지 for문 돌아가면서 각 frame을 resize & video_container에 대입  
        if end_frame - start_frame<40:
            if indx < new_start:
                indx = start_frame
            elif indx > new_end:
                indx = end_frame
            else:
                indx = indx-(new_start-start_frame)
            
        
        path2 = path1 + str(indx) + '.png'
      
        # print('path : ', path2)
        depth_img = Image.open(path2)
        if specific_path == 'depth_based_rgb':
            frame = np.array(depth_img)
            # print('shape : ', frame.shape)
            frame = frame[:152,:,:]
        else:
            depth_img = depth_img.resize((int(image_width),int(image_height)))
            frame = np.array(depth_img)
                    
        video_container[..., int(j)] = frame
      
        j+=1

    return video_container, label, None