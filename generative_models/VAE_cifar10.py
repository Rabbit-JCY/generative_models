"""
@Time: 25/03/2024 23:41 
@Author: lit_ruan
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sympy import Lambda

# 检查GPU是否可用
device = torch.device("mps")

# 定义变分自编码器（VAE）模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        )

        self.fc_mu = nn.Linear(64 * 4 * 4, 20)
        self.fc_logvar = nn.Linear(64 * 4 * 4, 20)

        self.fc_decoder = nn.Linear(20, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 64 * 4 * 4)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, -1e-7, 1e-7)
        # print(sum(sum(logvar)))
        z = self.reparameterize(mu, logvar)
        z = z.view(-1, 20)
        z = self.fc_decoder(z).view(-1, 64, 4, 4)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# 初始化模型并移动到 GPU
vae = VAE().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.01)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    vae.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        print(inputs[0])
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(inputs)
        BCE_loss = criterion(recon_batch, inputs)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE_loss + kl_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            print(BCE_loss.item(), kl_loss.item())
            running_loss = 0.0

    # 每轮训练后生成并展示 20 张图片
    if epoch % 2 == 0:
        vae.eval()
        with torch.no_grad():
            z = torch.randn(20, 20).to(device)
            z = vae.fc_decoder(z).view(20, 64, 4, 4)
            generated_images = vae.decoder(z).cpu().detach().numpy()
            generated_images = np.transpose(generated_images, (0, 2, 3, 1))

            fig = plt.figure(figsize=(8, 4))
            for i in range(20):
                ax = fig.add_subplot(4, 5, i + 1)
                ax.imshow(generated_images[i])
                ax.axis('off')
            plt.show()

print('Finished Training')
