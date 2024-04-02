"""
@Time: 01/04/2024 17:15 
@Author: lit_ruan
"""

import random
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST


def show_images(images, title=''):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                plt.imshow(images[idx][0], cmap='gray')
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.show()


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], 'images in the first batch')
        break


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=1e-4, max_beta=0.02, device=None, image_shape=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_shape = image_shape
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)

        self.alpha = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alpha[:i + 1]) for i in range(len(self.alpha))]).to(device)

    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy

    def backward(self, x, t):
        return self.network(x, t)


def sinusoidal_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape(1, d)
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()

        # batch_norm 强调不同样本之间同一纬度特征的分布是可以进行归一化的
        # layer_norm 强调同一样本之间不同纬度的特征分布是可以进行归一化的
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out


class MyUNet(nn.Module):
    def __init__(self, n_steps=200, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)

        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name='sampling.gif', c=1, h=28, w=28):
    # 16 x 1 x 28 x 28 x_T

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # estimate noise to be remove
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alpha[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # denoise the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 1:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z
    return x

def train_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path='ddpm_model.pt'):
    mse = nn.MSELoss()
    best_loss = float('inf')
    n_steps = ddpm.n_steps

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f'Epoch {epoch + 1}/{n_steps}', colour='#005500')):
            # loading data
            x0 = batch[0].to(device)
            n = len(x0)  # batchsize

            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            noisy_imgs = ddpm(x0, t, eta)  # x0.shape = [128,1,28,28]  eta : GT  前向生成nosiy_images

            eta_theta = ddpm.backward(noisy_imgs, t)  # the predicted noise   后向预测噪声

            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()


            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        if display:
            show_images(generate_new_images(ddpm, device=device), f'Images generated at epoch {epoch + 1}')

        log_string = f'\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}'

        # save the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += ' --> best model ever(stored)'
        print(log_string)


if __name__ == '__main__':

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    STORE_PATH_MNIST= f'ddpm_model_mnist.pt'
    STORED_PATH_FASHION= f'ddpm_model.pt'

    fashion = True
    train_flag = True
    batch_size = 128
    n_epochs = 20
    n_steps = 1000
    lr = 1e-3

    store_path = STORED_PATH_FASHION if fashion else STORE_PATH_MNIST
    device = torch.device("cpu")

    transform = Compose([
        ToTensor(),  # 将数据转换为张量
        Lambda(lambda x: (x - 0.5) * 2)
    ])

    ds_fn = FashionMNIST if fashion else MNIST
    dataset = ds_fn(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # show_first_batch(loader)

    if train_flag:
        unet = MyUNet(n_steps=1000).to(device)
        ddpm = MyDDPM(unet, device=device)
        # print(sum([p.numel() for p in ddpm.parameters()]))
        optimizer = Adam(ddpm.parameters(), lr=lr)
        train_loop(ddpm=ddpm, loader=loader, n_epochs=n_epochs, optim=optimizer, device=device)

    best_model = MyDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print('Model loaded')

    generated = generate_new_images(best_model)
    show_images(generated, title='')