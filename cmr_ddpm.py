import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, UNet128
import logging
import argparse
import wandb
import math
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
import sys
sys.argv=['']
del sys

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", schedule_name='cosine'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name = schedule_name

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    ### start schedulers
    def prepare_noise_schedule(self):
        if self.schedule_name == "linear":
        # Linear schedule from Ho
            return self.linear_beta_schedule()
        elif self.schedule_name == "cosine":
            # cosine scheduler for improved diffusion model from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
            return self.betas_for_alpha_bar(
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        elif self.schedule_name == 'quadratic':
            return self.quadratic_beta_schedule()
        elif self.schedule_name == 'sigmoid':
            return self.sigmoid_beta_schedule()
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.schedule_name}")
    def betas_for_alpha_bar(self, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        num_diffusion_timesteps = self.noise_steps
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.from_numpy(np.array(betas))
    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        https://huggingface.co/blog/annotated-diffusion
        """
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps , steps)
        alphas_cumprod = torch.cos(((x / self.noise_steps ) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):

        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def quadratic_beta_schedule(self):

        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.noise_steps) ** 2

    def sigmoid_beta_schedule(self):

        betas = torch.linspace(-6, 6, self.noise_steps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
    #### end schedulers
        
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.float32)
        # x = (x * 255).type(torch.uint8)
        return x


def train(default_config):
    # parse arguments
    parser = argparse.ArgumentParser()
    for keys in default_config:
        parser.add_argument('--'+keys, default=default_config[keys], type= type(default_config[keys]))
    args = parser.parse_args()

    setup_logging(args.run_name)
    device = args.device    
    data = CMRDataModule(
        data_dir=args.dataset_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_val_ratio=0.8,
        num_workers=args.num_workers,
    )
    data.prepare_data()
    data.setup()
    
    dataloader = data.train_dataloader()
    print('Training:  ', len(dataloader))
    # model = UNet().to(device)
    model = UNet128().to(device)
    if args.continue_train:
        # load model weights
        model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt")))
    print('loading model {}'.format(args.run_name))
    print('# of model parameters {} M'.format(int((10**(-6))*sum(p.numel() for p in model.parameters() if p.requires_grad))))
    # 1. Start a new run
    wandb.init(config = default_config , project='cmr_diffusion', name=args.run_name)

    # 2. Save model inputs and hyperparameters
    # config = wandb.config
    # config.dropout = 0.01
    # 3. Log gradients and model parameters
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.loss_type == 'mse':
        loss_fun = nn.MSELoss()
    elif args.loss_type == 'L1':
        loss_fun = nn.L1Loss()
    elif args.loss_type == 'huber':
        loss_fun = nn.SmoothL1Loss()
    else:
        raise NotImplementedError()
    # mse = nn.L1Loss()
    wandb.watch(model, loss_fun,  log='all', log_freq=100)
    ## seed everything
    set_seed(args.seed)
    diffusion = Diffusion(noise_steps = args.noise_steps, beta_start= args.beta_start, beta_end = args.beta_end, img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images['image']['data'].squeeze(dim=-1).to(device)
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = loss_fun(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            if i % args.log_interval == 0:      
                log_image_table(images, x_t, noise, predicted_noise, loss)
            # if i==0:
            #     torch.onnx.export(model, (x_t, t), os.path.join('./runs/', f"{args.run_name}.onnx"))
            #     wandb.save(os.path.join('./runs/', f"{args.run_name}.onnx"))

        if epoch%2==0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            wandb.log({"sampled image" : [wandb.Image(img.detach().cpu().numpy()) for img in sampled_images ]} )
        if epoch%50==0:
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}ckpt.pt"))
    
    wandb.finish()

def log_image_table(images, x_t, noise, predicted_noise, loss):
    # 4. Log metrics to visualize performance
    wandb.log({"MSE_loss": loss.item()})
    # wandb.log({"examples" : [wandb.Image(im) for im in images]})
    # my_table = {'image': [wandb.Image(im) for im in images] ,"noisy image": [wandb.Image(im) for im in x_t]}
    my_table = wandb.Table(columns=["image", "noisy image", "noise", "predicted noise"])
    for img, nimg, noi, pnoi in zip(images.to('cpu'), x_t.to('cpu'), noise.to('cpu'), predicted_noise.to('cpu')):
        my_table.add_data(wandb.Image(img.detach().cpu().numpy()), wandb.Image(nimg.detach().cpu().numpy()), wandb.Image(noi.detach().cpu().numpy()), wandb.Image(pnoi.detach().cpu().numpy()))
    wandb.log({"training procedure": my_table})

def launch(default_config):


    train(default_config)


default_config = {
    'dataset_path':  '/data/sina/dataset/MnMs2_full/MnM2/dataset-sorted/SA/PerDisease/',
    'run_name': "cmr_DDPM_Uncondtional_221207",
    'epochs': 100,
    'log_interval': 100,
    'batch_size' : 8,
    'image_size' : 256,
    'num_workers' : 8,
    'device' : "cuda:2",
    'lr' : 3e-4,
    'noise_steps' : 1000,
    'beta_start':1e-4,
    'beta_end': 0.02,
    'continue_train': False,
    'seed': 50,
    'loss_type': 'mse',




}

if __name__ == '__main__':
    launch(default_config)
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
