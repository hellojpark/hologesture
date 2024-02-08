import cv2
import numpy as np
from PIL import Image
import os

from pathlib import Path

def load_split_nvgesture(file_with_split='./hololens_test_correct.lst', specific_path='PV_aligned', list_split=list()):
    with open(file_with_split, 'rt') as f:
        dict_name = file_with_split[file_with_split.rfind('\\') + 1:]
        dict_name = dict_name[:dict_name.find('_')]

        for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1] + '/' +  specific_path
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth'):
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]  #params_dictionary[depth] = ./subject2/bare/class1/r3/depth / parsed[1] = sk_depth
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])

                    params_dictionary[key + '_end'] = int(parsed[3])

            list_split.append(params_dictionary)

    return list_split


def load_data_from_file(data_path, example_config, specific_path, sensor, image_width, image_height, nogesture = False):
    path = example_config['depth']                        #exmaple_config[depth] = ./subject2/bare/class1/r3/depth/sk_depth
    path = Path(data_path) / path[path.find('/') + 1:]   #c:/Users/User/Desktop/hololens_gesture_t/subject2/bare/class1/r3/depth/sk_depth
  
    start_frame = example_config['depth_start']
    end_frame = example_config['depth_end']
    label = example_config['label']

    chnum = 3

    # video_container = np.zeros((image_height, image_width, chnum, 40), dtype=np.uint8)
    video_container = np.zeros((112, 112, chnum, 40), dtype=np.uint8)

    
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
       
        depth_img = Image.open(path2)
        if specific_path == 'depth_based_rgb':
            frame = depth_img.crop((0,0,320, 152))
            frame = frame.resize((int(112), int(112)))
            frame = np.array(frame)
        else:
            # depth_img = depth_img.resize((int(image_width),int(image_height)))
            depth_img = depth_img.resize((int(112),int(112)))
            frame = np.array(depth_img)
        
        frame.astype(np.float32)
        # import matplotlib.pyplot as plt     
        # plt.imshow(frame)
        # plt.show()
        
        video_container[..., int(j)] = frame
      
        j+=1

        video_container = video_container.astype(np.float32)

    return video_container, label, None