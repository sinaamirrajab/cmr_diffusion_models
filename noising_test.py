# %%
import torch
from torchvision.utils import save_image
from cmr_ddpm import Diffusion
from utils import get_data, CMRDataModule
import argparse
from matplotlib import pyplot as plt
# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# args.batch_size = 1  # 5
# args.image_size = 64
# args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"

# dataloader = get_data(args)

data_dir = '/home/bme001/20180883/data/mnms2/sorted/SA/PerDisease'
data = CMRDataModule(
    data_dir=data_dir,
    batch_size=1,
    train_val_ratio=0.8,
    image_size=256,
    num_workers=1
)
data.prepare_data()
data.setup()
print('number of training data:  ', len(data.train_set))
train_loader = data.train_dataloader()
# get a bach o images
image = next(iter(train_loader))['image']['data'].squeeze(dim=-1)
# %%
schedule_name = 'linear'
diff = Diffusion(noise_steps=1000, beta_start=1e-4,
beta_end=0.02, img_size=128, device="cpu", schedule_name=schedule_name)

t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()
# t = torch.Tensor([5, 20, 35, 70, 130, 260, 370, 499]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), f"scheduler_{schedule_name}.jpg")

# %%
def plot_batch(batch,how_many_image):

    # batch = tio.utils.get_first_item(dataloader)

    fig, axes = plt.subplots(int(1), how_many_image, figsize=(12, 10))
    for ax, im in zip(axes.flatten(), batch):
        ax.imshow(im.cpu().numpy().squeeze(), cmap='gray')
    # plt.suptitle(sampler.__class__.__name__)
    plt.tight_layout()
plot_batch(noised_image, len(t))

# %%
