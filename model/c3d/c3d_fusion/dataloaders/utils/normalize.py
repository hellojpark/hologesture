import numpy as np

def video_std(tensor: np.ndarray):
    """Normalize function for a single tensor.

    Args:
        block (np.ndarray): input tensor
    Returns:
        np.ndarray: normalized tensor

    """
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=2)
    mean = np.array([tensor[..., chn, :].mean() for chn in range(tensor.shape[2])])
    std = np.array([tensor[..., chn, :].std() for chn in range(tensor.shape[2])])
    return (tensor - mean[:, np.newaxis]) / std[:, np.newaxis]

def frame_std(tensor: np.ndarray):
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=2)
    for frm in range(tensor.shape[3]):
        for chn in range(tensor.shape[2]):
            mean = np.array([tensor[..., chn, frm].mean()], np.float32)
            std = np.array([tensor[..., chn, frm].std()], np.float32)
  
        tensor = np.array(tensor, np.float32)
        tensor[...,frm] = ((tensor[...,frm]-mean) / std)
  
    return tensor

def video_min_max_norm(tensor: np.ndarray):
    # tensor shape : (288, 320, 1, 20)
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=2)
    data_max = np.array([np.max(tensor[..., chn, :]) for chn in range(tensor.shape[2])])
    data_min = np.array([np.min(tensor[..., chn, :]) for chn in range(tensor.shape[2])])
    
    return ((tensor-data_min[:, np.newaxis]) / (data_max[:, np.newaxis]-data_min[:, np.newaxis]))

def frame_min_max_norm(tensor: np.ndarray):
    # print('input tensor shape : ', tensor.shape)
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=2)
    # tensor shape : (288, 320, 1, 20)
    # tensor.shape[2] : 1
    # print('tensor.shape[3] : ', tensor.shape[3])
    for frm in range(tensor.shape[3]):
        for chn in range(tensor.shape[2]):
            data_max = np.array([np.max(tensor[..., chn, frm])], np.float32)
            data_min = np.array([np.min(tensor[..., chn, frm])], np.float32)
        
        # print('data max : ', data_max)
        # print('data min : ', data_min)
        # print('data mean1 : ', np.array([np.mean(tensor[..., chn, frm])], np.float32))
  
        tensor = np.array(tensor, np.float32)
        # print('data mean2 : ', np.array([np.mean(tensor[..., chn, frm])], np.float32))
        tensor[...,frm] = ((tensor[...,frm]-data_min) / (data_max-data_min))
        # print('data mean3 : ', np.array([np.mean(tensor[...,frm])], np.float32))
        # print('data mean4 : ', np.array([np.mean(tensor[..., 0, frm])], np.float32))
  
    return tensor

def img_net_std(tensor: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406])
    # print('shape : ', mean.shape)
    std = np.array([0.229, 0.224, 0.225])
    if len(tensor.shape) <4:
        tensor = np.expand_dims(tensor, axis=2)
    return (tensor - mean[:, np.newaxis]) / std[:, np.newaxis]