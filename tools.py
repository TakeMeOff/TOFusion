from tempfile import tempdir
import numpy
import cv2
import os
import glob
import matplotlib.pyplot as plt
import random
import torch
from tensorboardX import SummaryWriter

def prepare_data_path(dataset_path):
    # 获取文件夹下的所有文件路径和名字
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

def prepare_path(data_root, mode):
    data_root = os.path.join(data_root, mode)
    img_vi_path = os.path.join(data_root, "img_vi")
    img_if_path = os.path.join(data_root, "img_if")
    mask_path = os.path.join(data_root, "mask")
    text_path = os.path.join(data_root, "text")
    return img_vi_path, img_if_path, mask_path, text_path

def show_tensor(img):
    '''
    :param img: Tensor (C,H,W) 0-1.0
    '''
    img = img.numpy().transpose(1, 2, 0)
    img = img * 255
    img = img.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    plt.imshow(img)
    plt.show()

def check_img_range(img):
    ones = torch.ones_like(img)
    zeros = torch.zeros_like(img)
    img = torch.where(img > ones, ones, img)
    img = torch.where(img < zeros, zeros, img)
    return img

def save_img_tensor(img, name, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = numpy.round(img * 255).astype(numpy.uint8)
    for k in range(img.shape[0]):
        img[k, :, :, :] = cv2.cvtColor(img[k,:,:,:], cv2.COLOR_YCrCb2BGR)

    for k in range(len(name)):
        img_name = name[k]
        save_path = os.path.join(result_path, img_name)
        cv2.imwrite(save_path, img[k, :, :, :])
        # print("Fusion {0} Successfully!".format(img_name))

def save_img_tensor_gray(img, name, result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = numpy.round(img * 255).astype(numpy.uint8)

    for k in range(len(name)):
        img_name = name[k]
        save_path = os.path.join(result_path, img_name)
        cv2.imwrite(save_path, img[k, :, :, :])
        # print("Fusion {0} Successfully!".format(img_name))

def write_image(img, name, path):
    img = check_img_range(img)
    if img.shape[1] == 1:
        save_img_tensor_gray(img, name, path)
    elif img.shape[1] == 3:
        save_img_tensor(img, name, path)
    else:
        print("The input channel(s) should be 1 or 3")

def save_weight(model, optimizer, scheduler, epoch_cur, epochs, weight_path):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    path = os.path.join(weight_path, "checkpoint")
    state = {'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'epoch_cur': epoch_cur,
              'epochs': epochs}
    torch.save(state, f'{path}_epoch_{epoch_cur}.pth')
    print("---------Save {}th Path---------".format(epoch_cur))

def save_loss(write, data, title, x_label):
    write.add_scalar(title, sum(data)/len(data), x_label)
    data.clear()

def save_img(write, img, title, x_label):
    write.add_images(title, img, x_label)

def threshold_mask(mask, threshold):
    temp = torch.where(mask > 0.1, torch.tensor(1.0), mask)
    return temp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_text(file):
    random_line = random.choice(file)
    random_line = random_line.strip()
    return random_line


