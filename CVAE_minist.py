import torch
import torch.nn as nn
import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 定义 CVAE 模型
class CVAE(nn.Module):
    def __init__(self, input_dim, label_dim, latent_dim):
        super(CVAE, self).__init__()
        # 编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 28 * 28 + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 28 * 28 + label_dim, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim + label_dim, 128 * 28 * 28)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        xy = torch.cat((x, y), dim=1)
        return self.fc_mu(xy), self.fc_logvar(xy)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat((z, y), dim=1)
        x = self.decoder_fc(zy)
        x = x.view(-1, 128, 28, 28)
        return self.decoder_conv(x)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar


# 训练循环
def train(epoch: int):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
        data, label = data.to(device), F.one_hot(label, num_classes=10).float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        # BCE = F.binary_cross_entropy(recon_batch, data, reduction='mean')
        # BCE = torch.where(torch.isnan(BCE) | torch.isinf(BCE), torch.tensor(0.0, device=device), BCE)
        BCE = F.mse_loss(recon_batch, data, reduction='mean')

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('epoch: {} loss: {}'.format(epoch, train_loss / len(train_loader) / batch_size))
    # 保存模型
    if epoch % 1 == 0:
        torch.save(model.state_dict(), './ckpt/mnist_' + str(epoch) + '.pth')
        generate_images([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2)


def generate_images(label_indices):
    model.eval()  # 设置模型为评估模式
    latent = torch.randn(20, 20).to(device)  # 20 张图片
    conditions = torch.eye(10)[label_indices].float().to(device)  # 指定的条件
    with torch.no_grad():
        generated_imgs = model.decode(latent, conditions).cpu()
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(generated_imgs[i][0], cmap='gray')
        plt.axis('off')
    plt.show()


# 设置训练参数
batch_size = 128
epochs = 30
lr = 1e-3

# 加载和预处理 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器和损失函数
device = torch.device("mps")
model = CVAE(input_dim=1, label_dim=10, latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

if __name__ == '__main__':
    flag = True
    # flag = False
    if flag:
        print('start training...')
        for i in range(1, epochs + 1):
            train(i)
    else:
        # 加载模型
        model = CVAE(input_dim=1, label_dim=10, latent_dim=20).to(device)
        model.load_state_dict(torch.load('./ckpt/mnist_1.pth'))
        model.eval()  # 设置模型为评估模式
        generate_images([5] * 20)
