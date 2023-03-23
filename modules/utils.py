import os, shutil, random
from pathlib import Path
# from kaggle import api
import zipfile
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
# from fastdownload import FastDownload
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os
import torchio as tio
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import numpy as np


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def one_batch(dl):
    return next(iter(dl))
        

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).type(torch.uint8)
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()

    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.train_folder), transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=val_transforms)
    
    if args.slice_size>1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataset


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)



# %%
def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        # print(img['image'].data[0,:,:,1].squeeze().min(), img['image'].data[...,1].squeeze().max() )
        plt.imshow(img['image'].data[0,:,:, 1].squeeze(), cmap='gray')
# https://github.com/Project-MONAI/tutorials/blob/main/modules/TorchIO_MONAI_PyTorch_Lightning.ipynb

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def plot_batch(dataloader):

    batch = tio.utils.get_first_item(dataloader)

    fig, axes = plt.subplots(batch['image']['data'].shape[0]//2, 2, figsize=(12, 10))
    for ax, im in zip(axes.flatten(), batch['image']['data']):
        ax.imshow(im.squeeze(), cmap='gray')
    # plt.suptitle(sampler.__class__.__name__)
    plt.tight_layout()



def cine_to_3D(data_dir):
    # converting 4D cine images (x, y, z, t) to 3D volumes per cardiac phase (x, y, z)
    # input path : /data/MMs2/dataset
    # output path : /data/MMs2/dataset_3D
    import nibabel as nib
    from tqdm import tqdm
    cine_list = sorted(os.listdir(data_dir))
    print('loading cine data and converting to 3D volumes...')
    pbar = tqdm(cine_list)
    
    for i, file in enumerate(pbar):
        cine_image = nib.load(os.path.join(data_dir, file, file + '_SA_CINE.nii.gz'))
        for phase in range(cine_image.shape[-1]):
            image_3D = cine_image.get_fdata()[..., phase]
            os.makedirs(os.path.join(data_dir + '_3D', file), exist_ok=True)
            out_dir = os.path.join(data_dir + '_3D', file, file + '_SA_P{}.nii.gz'.format(phase))
            nifti_out_image = nib.Nifti1Image(image_3D, cine_image.affine, cine_image.header)
            nib.save(nifti_out_image, out_dir)

import nibabel as nib
from tqdm import tqdm
import numpy as np    
class bbox():
    
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        
    def return_mask(self, mask, extra_dim):

        mask[mask==2]=1
        mask[mask==3]=1

        xx,yy=np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))
        xx2, yy2=np.meshgrid(range(mask.shape[1]), range(mask.shape[0]))
        xx=xx[..., None] * np.ones((1, 1, mask.shape[2]))
        yy=yy[..., None]* np.ones(( 1, 1, mask.shape[2]))
        xx2=xx2[..., None] * np.ones((1, 1, mask.shape[2]))
        yy2=yy2[..., None]* np.ones(( 1, 1, mask.shape[2]))

        ymin=int(np.min(yy[np.transpose(mask, axes=[1,0,2])==1]))
        ymax=int(np.max(yy[np.transpose(mask, axes=[1,0,2])==1]))
        ymin2=int(np.min(yy2[mask==1]))
        ymax2=int(np.max(yy2[mask==1]))

        center_point_x=ymin-extra_dim+(ymax+extra_dim-ymin+extra_dim)/2
        center_point_y=ymin2-extra_dim+(ymax2+extra_dim-ymin2+extra_dim)/2
        anchor_point=(ymin-extra_dim, ymin2-extra_dim)
        width=ymax+extra_dim-ymin+extra_dim
        height=ymax2+extra_dim-ymin2+extra_dim

        #return center_point_x, center_point_y, anchor_point, width, height
        mask[ymin2-extra_dim:(ymax2+1+extra_dim),ymin-extra_dim:(ymax+extra_dim+1),: ]=1

        return mask
    
    def return_bbox(self, mask, extra_dim):
	
        mask[mask==2]=1
        mask[mask==3]=1
        
        xx,yy=np.meshgrid(range(mask.shape[0]), range(mask.shape[1]))
        xx2, yy2=np.meshgrid(range(mask.shape[1]), range(mask.shape[0]))
        xx=xx[..., None] * np.ones((1, 1, mask.shape[2]))
        yy=yy[..., None]* np.ones(( 1, 1, mask.shape[2]))
        xx2=xx2[..., None] * np.ones((1, 1, mask.shape[2]))
        yy2=yy2[..., None]* np.ones(( 1, 1, mask.shape[2]))
        
        ymin=int(np.min(yy[np.transpose(mask, axes=[1,0,2])==1]))
        ymax=int(np.max(yy[np.transpose(mask, axes=[1,0,2])==1]))
        ymin2=int(np.min(yy2[mask==1]))
        ymax2=int(np.max(yy2[mask==1]))
        
        
        center_point_x=ymin-extra_dim+(ymax+extra_dim-ymin+extra_dim)/2
        center_point_y=ymin2-extra_dim+(ymax2+extra_dim-ymin2+extra_dim)/2
        
        anchor_point=(ymin-extra_dim, ymin2-extra_dim)
        width=ymax+extra_dim-ymin+extra_dim
        height=ymax2+extra_dim-ymin2+extra_dim
        
        return center_point_x, center_point_y, anchor_point, width, height, ymin, ymax,ymin2, ymax2
    
    def return_crop_points(self, xmin, xmax, ymin,ymax, margin):

        diff_x=(xmax+margin)-(xmin-margin)
        diff_y=(ymax+margin)-(ymin-margin)

        abs_diff=np.abs(diff_x-diff_y)

        if diff_x<diff_y:
            if abs_diff%2==0:
                xmin_new=xmin-margin-int(abs_diff/2)
                xmax_new=xmax+margin+int(abs_diff/2)
                ymin_new=ymin-margin 
                ymax_new=ymax+margin
            else:
                xmin_new=xmin-margin-int(abs_diff/2)
                xmax_new=xmax+margin+int(abs_diff/2)+1
                ymin_new=ymin-margin
                ymax_new=ymax+margin 

        elif diff_x>diff_y:
            if abs_diff%2==0:
                xmin_new=xmin-margin
                xmax_new=xmax+margin
                ymin_new=ymin-margin-int(abs_diff/2) 
                ymax_new=ymax+margin+int(abs_diff/2)
            else:
                xmin_new=xmin-margin
                xmax_new=xmax+margin
                ymin_new=ymin-margin-int(abs_diff/2) 
                ymax_new=ymax+margin+int(abs_diff/2)+1
                
        elif diff_x==diff_y:

            xmin_new=xmin-margin
            xmax_new=xmax+margin
            ymin_new=ymin-margin
            ymax_new=ymax+margin

        return xmin_new, xmax_new, ymin_new, ymax_new
    
    def crop_image(self):
        data_list = sorted(os.listdir(self.data_dir))
        pbar = tqdm(data_list)

        for i, data in enumerate(pbar):

            label = nib.load(os.path.join(self.data_dir, data, data + '_SA_ED_gt.nii.gz'))
            label_array = np.round(label.get_fdata()).astype(np.int64)
            bbox_array = self.return_mask(label_array, 1)
            margin=15
            center_point_x, center_point_y, anchor_point, width, height, xmin, xmax, ymin, ymax = self.return_bbox(bbox_array, margin)
            xmin_new, xmax_new, ymin_new, ymax_new = self.return_crop_points(xmin, xmax, ymin, ymax, margin=margin)
            if xmin_new < 0:
                xmin_new = 0
            elif ymin_new < 0:
                ymin_new = 0

            image_3D_list = sorted(os.listdir(os.path.join(self.data_dir + '_3D', data )))           
            for image_3D in image_3D_list:
                image_3D_array = nib.load(os.path.join(self.data_dir + '_3D', data, image_3D ))
                cropped_img = image_3D_array.get_fdata()[ymin_new:ymax_new, xmin_new:xmax_new, ...]


                print("original image shape:", image_3D_array.shape)
                print("Cropped image shape:", cropped_img.shape)
                os.makedirs(os.path.join(self.data_dir + '_3D_crop', data), exist_ok=True)
                out_dir = os.path.join(self.data_dir + '_3D_crop', data, image_3D)
                header = image_3D_array.header
                header['dim'][0:4]=[3, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]]
                nifti_out_image = nib.Nifti1Image(cropped_img, image_3D_array.affine, header )
                nib.save(nifti_out_image, out_dir)

