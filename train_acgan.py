"""
训练 AC-GAN 的判别器分类头，并将权重保存到 acgan_discriminator.pth
以后只需用 predict_digit.py 进行推断，无需再训练
"""
import time, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from acgan_models import Generator, Discriminator  # 见下方『模型文件』

# 配置
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs       = 80
batch_size   = 128
latent_dim   = 100
num_classes  = 10

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# 模型
G, D = Generator(latent_dim, num_classes).to(device), Discriminator(num_classes).to(device)
criterion_adv = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4)
opt_D = optim.Adam(D.parameters(), lr=2e-4)

for epoch in range(epochs):
    t0 = time.time()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake  = torch.zeros(imgs.size(0), 1, device=device)

        # ---- 训练 G ----
        opt_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),), device=device)
        gen_imgs   = G(z, gen_labels)

        v_fake, c_fake = D(gen_imgs)
        g_loss = criterion_adv(v_fake, valid) + criterion_cls(c_fake, gen_labels)
        g_loss.backward();  opt_G.step()

        # ---- 训练 D ----
        opt_D.zero_grad()
        v_real, c_real = D(imgs)
        v_fake, c_fake_det = D(gen_imgs.detach())

        d_real = criterion_adv(v_real, valid) + criterion_cls(c_real, labels)
        d_fake = criterion_adv(v_fake, fake)  + criterion_cls(c_fake_det, gen_labels)
        d_loss = (d_real + d_fake) / 2
        d_loss.backward();  opt_D.step()

    print(f"Epoch {epoch+1:02}/{epochs} | D:{d_loss.item():.3f} | G:{g_loss.item():.3f} | {time.time()-t0:.1f}s")

# 仅保存 D（识别用）
torch.save(D.state_dict(), "acgan_discriminator.pth")
print("判别器已保存：acgan_discriminator.pth")
