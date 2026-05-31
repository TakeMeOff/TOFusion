import random
import numpy
from torch.utils.data import Dataset
import torch
import cv2
from tools import *
import clip
# Global Parameters
re_size = 600

class FusionDatasets(Dataset):

    def __init__(self, img_vi_path, img_if_path, text_path, mask_path=None, train='train', text=None, parser=None):
        if train == 'train':
            self.classes = parser.text_type
            self.train = train
            self.vi_path, self.names = prepare_data_path(img_vi_path)
            self.if_path, _ = prepare_data_path(img_if_path)
            self.text_path = text_path
            self.mask_path = mask_path
            self.length = len(self.names)
            self.resize = parser.resize
        else:
            self.train = train
            self.vi_path, self.names = prepare_data_path(img_vi_path)
            self.if_path, _ = prepare_data_path(img_if_path)
            self.length = len(self.names)
            self.text = text

    def __len__(self):
        return self.length

    def __process__(self, img):
        '''
        :param img: Numpy Img(H,W,C)    0-255
        :return: Tensor Img(C,H',W')  0-1.0
        '''
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if img.ndim < 3:
            img = numpy.expand_dims(img, axis=-1)
        img = torch.tensor(img / 255.0)
        img = torch.permute(img, (2, 0, 1)) # (H,W,C) -> (C,H,W)
        img = img.type(torch.float32)
        return img

    def __getitem__(self, index):
        '''
        :return: RGB Visible Images, Gray Infrared Images, Gray Mask
        :Range: 0 - 1.0
        :Shape: (C,H,W)
        :Type: FloatTensor
        '''
        if self.train == 'train':
            type_fusion = random.choice(self.classes)
            mask_paths = os.path.join(self.mask_path, type_fusion)
            mask_paths, _ = prepare_data_path(mask_paths)

            vi_path = self.vi_path[index]
            if_path = self.if_path[index]
            mask_path = mask_paths[index]
            file_name = self.names[index]

            img_vi_h = cv2.imread(vi_path)
            img_if_h = cv2.imread(if_path, cv2.IMREAD_GRAYSCALE)
            mask_h = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if self.resize:
                img_vi_h = cv2.resize(img_vi_h, (re_size, re_size))
                img_if_h = cv2.resize(img_if_h, (re_size, re_size))
                mask_h = cv2.resize(mask_h, (re_size, re_size))

                img_vi_l = cv2.resize(img_vi_h, (re_size // 2, re_size // 2))
                img_if_l = cv2.resize(img_if_h, (re_size // 2, re_size // 2))
                mask_l = cv2.resize(mask_h, (re_size // 2, re_size // 2))

            else:
                h,w,_ = img_vi_h.shape
                img_vi_l = cv2.resize(img_vi_h, (w // 2, h // 2))
                img_if_l = cv2.resize(img_if_h, (w // 2, h // 2))
                mask_l = cv2.resize(mask_h, (w // 2, h // 2))

            img_vi_h = self.__process__(img_vi_h)
            img_if_h = self.__process__(img_if_h)
            mask_h = self.__process__(mask_h)

            img_vi_l = self.__process__(img_vi_l)
            img_if_l = self.__process__(img_if_l)
            mask_l = self.__process__(mask_l)

            return (img_vi_h, img_vi_l, img_if_h, img_if_l, mask_h, mask_l, file_name, type_fusion)

        else:
            vi_path = self.vi_path[index]
            if_path = self.if_path[index]
            file_name = self.names[index]

            img_vi_h = cv2.imread(vi_path)
            img_if_h = cv2.imread(if_path, cv2.IMREAD_GRAYSCALE)

            h,w,c = img_vi_h.shape
            img_vi_l = cv2.resize(img_vi_h, (w // 2, h // 2))
            img_if_l = cv2.resize(img_if_h, (w // 2, h // 2))

            img_vi_h = self.__process__(img_vi_h)
            img_if_h = self.__process__(img_if_h)
            img_vi_l = self.__process__(img_vi_l)
            img_if_l = self.__process__(img_if_l)

            return (img_vi_h, img_vi_l, img_if_h, img_if_l, file_name, self.text)