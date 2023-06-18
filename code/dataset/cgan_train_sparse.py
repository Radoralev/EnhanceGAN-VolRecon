import torch
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode



def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=InterpolationMode.BICUBIC, normalize=True, isTrain=False):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop and isTrain:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class CustomDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, scan_list, opt, isTrain):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.opt = opt
        self.isTrain = isTrain

        # Map from rgb images to corresponding reference views
        self.rgb_to_ref = {'00000000.jpg': '23', '00000001.jpg': '24', '00000002.jpg': '33'}
        
        self.scan_list = [name for name in os.listdir(os.path.join(root_dir2, 'rgb')) if os.path.isdir(os.path.join(root_dir2, 'rgb', name)) and int(name[-2:]) in scan_list]

        self.flattened_list = [(scan_id, img_name) for scan_id in self.scan_list for img_name in self.rgb_to_ref.keys()]

    def __len__(self):
        return len(self.flattened_list)

    def __getitem__(self, idx):
        scan_id, img_name = self.flattened_list[idx]

        # Load input image
        img_path = os.path.join(self.root_dir2, 'rgb', scan_id, img_name)
        input_img = Image.open(img_path).convert('RGB')

        # Load corresponding ground truth image
        ref_view = self.rgb_to_ref[img_name]
        img_path = os.path.join(self.root_dir1, scan_id, 'image', '0000' + ref_view + '.png')
        ground_truth_img = Image.open(img_path).convert('RGB')

        # Calculate transformation parameters once for both images
        params = get_params(self.opt, input_img.size)
        
        # Apply the same transformations to both images
        transform = get_transform(self.opt, params, isTrain=self.isTrain)

        input_img = transform(input_img)
        ground_truth_img = transform(ground_truth_img)

        return {'input': input_img, 'ground_truth': ground_truth_img}
