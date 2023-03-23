import torch
import torchvision
import torchvision.transforms as T
import torchio as tio
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader
# dataloader 
class CMR2DDataModule(pl.LightningDataModule):
    def __init__(self, data_dir,image_size, batch_size, train_val_ratio, num_workers):
        super().__init__()
        # self.task = task
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size
        # self.base_dir = root_dir
        # self.dataset_dir = os.path.join(root_dir, task)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=2)

    def get_mean_res(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spacing for s in dataset])
        return shapes[:,0].mean()


    # def download_data(self):
    #     if not os.path.isdir(self.dataset_dir):
    #         url = f'https://msd-for-monai.s3-us-west-2.amazonaws.com/{self.task}.tar'
    #         monai.apps.download_and_extract(url=url, output_dir=self.base_dir)

    #     image_training_paths = sorted(glob(os.path.join(self.dataset_dir, 'imagesTr', "*.nii*")))
    #     label_training_paths = sorted(glob(os.path.join(self.dataset_dir, 'labelsTr', "*.nii*")))
    #     image_test_paths = sorted(glob(os.path.join(self.dataset_dir, 'imagesTs', "*.nii*")))
    #     return image_training_paths, label_training_paths, image_test_paths
    def list_data(self):
        image_test_list = []
        image_list = []
        label_list = []
        self.test_dir = self.data_dir.replace('dataset-sorted', 'dataset-sorted-test')
        if 'PerDisease' in self.data_dir:
            disease_list = sorted(os.listdir(self.data_dir))

            for dis in disease_list:
                image_list +=[os.path.join(self.data_dir, dis, 'Image', name) for name in sorted(os.listdir(os.path.join(self.data_dir, dis, 'Image')))]
                label_list +=[os.path.join(self.data_dir, dis, 'Label', name) for name in sorted(os.listdir(os.path.join(self.data_dir, dis, 'Label')))]
                image_test_list  +=[os.path.join(self.test_dir, dis, 'Label', name) for name in sorted(os.listdir(os.path.join(self.test_dir, dis, 'Label')))]
                # image_class.append(dis)

        elif 'PerScanner' in data_dir:
            vendor_list = sorted(os.listdir(data_dir))

            for ved in vendor_list:
                image_list +=[os.path.join(self.data_dir, ved, 'Image', name) for name in sorted(os.listdir(os.path.join(self.data_dir, ved, 'Image')))]
                label_list +=[os.path.join(self.data_dir, ved, 'Label', name) for name in sorted(os.listdir(os.path.join(self.data_dir, ved, 'Label')))]
                image_test_list  +=[os.path.join(self.test_dir, ved, 'Label', name) for name in sorted(os.listdir(os.path.join(self.test_dir, ved, 'Label')))]
        else:
            raise NotImplementedError
        return image_list, label_list, image_test_list

    def prepare_data(self):
        image_training_paths, label_training_paths, image_test_paths = self.list_data()

        self.subjects = []
        for image_path, label_path in zip(image_training_paths, label_training_paths):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)

        self.test_subjects = []
        for image_path in image_test_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.Resample((1.25, 1.25, 9.8), image_interpolation ='bspline'),
            tio.CropOrPad((self.image_size,self.image_size, 13)),
            tio.ZNormalization(),
            # tio.RescaleIntensity((-1, 1)),
            # tio.CropOrPad((self.image_size,self.image_size,self.get_max_shape(self.subjects))),
            # tio.Resample((self.get_mean_res(self.subjects), self.get_mean_res(self.subjects)), image_interpolation ='bspline'),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            # tio.RandomAffine(p=0.5),
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = torch.utils.data.random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess]) # add agument here for data augmentationd during trainig

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        patch_size = (self.image_size, self.image_size, 1)  # 2D slices
        max_queue_length = 1000
        patches_per_volume = 13
        sampler = tio.UniformSampler(patch_size)
        # sampler = tio.WeightedSampler(patch_size, probability_map='image') # makes it really slow
        queue = tio.Queue(self.train_set, max_queue_length, patches_per_volume, sampler)
        return DataLoader(queue, self.batch_size,shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        patch_size = (self.image_size, self.image_size, 1)  # 2D slices
        max_queue_length = 1000
        patches_per_volume = 13
        sampler = tio.UniformSampler(patch_size)
        # sampler = tio.WeightedSampler(patch_size, probability_map='image')
        queue = tio.Queue(self.val_set, max_queue_length, patches_per_volume, sampler)
        return DataLoader(queue, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)


class CMRCineDataModule(pl.LightningDataModule):
    def __init__(self, data_dir,image_size, batch_size, train_val_ratio, num_workers):
        super().__init__()
        # self.task = task
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size
        # self.base_dir = root_dir
        # self.dataset_dir = os.path.join(root_dir, task)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=2)

    def get_mean_res(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spacing for s in dataset])
        return shapes[:,0].mean()


    # def download_data(self):
    #     if not os.path.isdir(self.dataset_dir):
    #         url = f'https://msd-for-monai.s3-us-west-2.amazonaws.com/{self.task}.tar'
    #         monai.apps.download_and_extract(url=url, output_dir=self.base_dir)

    #     image_training_paths = sorted(glob(os.path.join(self.dataset_dir, 'imagesTr', "*.nii*")))
    #     label_training_paths = sorted(glob(os.path.join(self.dataset_dir, 'labelsTr', "*.nii*")))
    #     image_test_paths = sorted(glob(os.path.join(self.dataset_dir, 'imagesTs', "*.nii*")))
    #     return image_training_paths, label_training_paths, image_test_paths
    def list_data(self):

        images = []
        image_files = sorted(os.listdir(self.data_dir))
        
        for file in image_files:
            image_list = sorted(os.listdir(os.path.join(self.data_dir, file)))
            for image in image_list:
                images.append(os.path.join(self.data_dir, file, image))

        return images

    def prepare_data(self):
        image_training_paths = self.list_data()

        self.subjects = []
        for image_path in image_training_paths:
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(
                image=tio.ScalarImage(image_path)
                )
            self.subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.Resample((1.25, 1.25, 9.8), image_interpolation ='bspline'),
            tio.CropOrPad((self.image_size, self.image_size, 13)),
            tio.ZNormalization(),
            # tio.RescaleIntensity((-1, 1)),
            # tio.CropOrPad((self.image_size,self.image_size,self.get_max_shape(self.subjects))),
            # tio.Resample((self.get_mean_res(self.subjects), self.get_mean_res(self.subjects)), image_interpolation ='bspline'),
            # tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            # tio.RandomAffine(p=0.5),
            # tio.RandomGamma(p=0.5),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.1),
            # tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = torch.utils.data.random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess]) # add agument here for data augmentationd during trainig

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        # self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        patch_size = (self.image_size, self.image_size, 1)  # 2D slices
        max_queue_length = 1000
        patches_per_volume = 13
        sampler = tio.UniformSampler(patch_size)
        # sampler = tio.WeightedSampler(patch_size, probability_map='image') # makes it really slow
        queue = tio.Queue(self.train_set, max_queue_length, patches_per_volume, sampler)
        return DataLoader(queue, self.batch_size,shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        patch_size = (self.image_size, self.image_size, 1)  # 2D slices
        max_queue_length = 1000
        patches_per_volume = 13
        sampler = tio.UniformSampler(patch_size)
        # sampler = tio.WeightedSampler(patch_size, probability_map='image')
        queue = tio.Queue(self.val_set, max_queue_length, patches_per_volume, sampler)
        return DataLoader(queue, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # return DataLoader(self.test_set, self.batch_size,shuffle=False, num_workers=self.num_workers)
        return NotImplementedError
