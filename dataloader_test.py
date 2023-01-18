# %%
from utils import *
import argparse
import sys
sys.argv=['']
del sys

# set some options
default_config = {
    'dataset_path':  '/home/bme001/20180883/data/mnms2/sorted/SA/PerDisease',
    'run_name': "cmr_DDPM_Uncondtional_230128",
    'epochs': 50,
    'log_interval': 100,
    'batch_size' : 8,
    'image_size' : 256,
    'num_workers' : 8,
    'device' : "cuda:4",
    'lr' : 3e-4,
    'noise_steps' : 500,
    'beta_start':1e-4,
    'beta_end': 0.01,
    }

parser = argparse.ArgumentParser()
for keys in default_config:
    parser.add_argument('--'+keys, default=default_config[keys], type= type(default_config[keys]))
args = parser.parse_args()


data = CMRDataModule(
        data_dir=args.dataset_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_val_ratio=0.8,
        num_workers=args.num_workers,
    )
data.prepare_data()
data.setup()
print('batch_size {}'.format(args.batch_size))
dataloader = data.train_dataloader()
print('number of images is {}'.format(len(dataloader)))
# %%
# visualize a batch of images
plot_batch(dataloader)


# %%
# test the training loop
from tqdm import tqdm
import logging
for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            labels = images['location'][:,2].to(args.device)
            images = images['image']['data'].squeeze(dim=-1).to(args.device   )
            images = images.to(args.device   )
            print(images.shape)
            if len(images.shape)<4:
                print(images)
# %%
