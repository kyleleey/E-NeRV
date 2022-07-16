import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, train=True):
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0 
        for img_id in all_imgs:
            self.frame_path.append(img_id)
            frame_idx.append(num_frame)  # if 135 frames in total, this list will store 0, 1, 2, ..., 133, 134
            num_frame += 1          

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        # the id for first frame is 0 and the id for last is 1
        self.frame_idx = []
        for i in range(len(frame_idx)):
            x = frame_idx[i]
            self.frame_idx.append(float(x) / (len(frame_idx) - 1))
        self.accum_img_num = np.asfarray(accum_img_num)

        self.height = 720
        self.width = 1280

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, idx):
        valid_idx = int(idx)
        img_id = self.frame_path[valid_idx]
        img_name = os.path.join(self.main_dir, img_id)
        image = Image.open(img_name).convert("RGB")
        
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height))

        tensor_image = self.transform(image)
        if tensor_image.size(1) > tensor_image.size(2):
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[idx])

        data_dict = {
            "img_id": frame_idx,
            "img_gt": tensor_image,
        }
        return data_dict