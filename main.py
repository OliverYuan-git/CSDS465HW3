import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from scipy.linalg import sqrtm
import numpy as np
from torchvision.transforms import Resize

def prepare_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# G
class Generator(nn.Module):
    def __init__(self, latent_dim=128, filters=128):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(filters * 4, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(filters * 2, filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.ConvTranspose2d(filters, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# D
class Discriminator(nn.Module):
    def __init__(self, filters=128):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters * 2, filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input).view(-1)

# Inception
class InceptionClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionClassifier, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Score
def calculate_inception_score(generator, classifier, num_samples, batch_size, device):
    generator.eval()
    classifier.eval()
    all_preds = []
    resize = Resize((299, 299))

    with torch.no_grad():
        for _ in tqdm(range(num_samples // batch_size)):
            z = torch.randn(batch_size, generator.latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            fake_images_resized = resize(fake_images)  # Resize images
            preds = torch.softmax(classifier(fake_images_resized), dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)
    mean_preds = np.mean(all_preds, axis=0)
    kl_div = all_preds * (np.log(all_preds + 1e-10) - np.log(mean_preds + 1e-10))
    inception_score = np.exp(np.mean(np.sum(kl_div, axis=1)))

    return inception_score

# FID
def extract_features(data_loader_or_generator, classifier, device, is_generator=False, generator=None, latent_dim=None, num_samples=None, batch_size=None):
    classifier.eval()
    features = []
    resize = Resize((299, 299))

    with torch.no_grad():
        if is_generator:
            for _ in tqdm(range(num_samples // batch_size)):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_images = generator(z)
                fake_images_resized = resize(fake_images)
                preds = classifier(fake_images_resized).detach().cpu().numpy()
                features.append(preds)
        else:
            for imgs, _ in tqdm(data_loader_or_generator):
                imgs = imgs.to(device)
                imgs_resized = resize(imgs)
                preds = classifier(imgs_resized).detach().cpu().numpy()
                features.append(preds)

    features = np.concatenate(features, axis=0)
    return features

def calculate_fid(real_features, fake_features):
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def save_samples(generator, latent_dim, epoch, device):
    z = torch.randn(100, latent_dim, 1, 1, device=device)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = (gen_imgs + 1) / 2  # Rescale to [0, 1]
    os.makedirs("samples", exist_ok=True)
    save_image(gen_imgs, f"samples/sample_epoch_{epoch}.png", nrow=10)

def train_gan(latent_dim, batch_size, epochs, lr, lambda_gp, n_critic, dataloader, device):
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0, 0.9))

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(tqdm(dataloader)):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = G(z).detach()
            real_validity = D(real_imgs)
            fake_validity = D(fake_imgs)
            gradient_penalty = compute_gradient_penalty(D, real_imgs, fake_imgs)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                gen_imgs = G(z)
                g_loss = -torch.mean(D(gen_imgs))
                g_loss.backward()
                optimizer_G.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_samples(G, latent_dim, epoch, device)

    return G

# MAIN
if __name__ == '__main__':
    batch_size = 128
    latent_dim = 128
    epochs = 500
    lr = 2e-4
    lambda_gp = 10
    n_critic = 5
    num_samples = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = prepare_data(batch_size)
    generator = train_gan(latent_dim, batch_size, epochs, lr, lambda_gp, n_critic, dataloader, device)
    classifier = InceptionClassifier(num_classes=10).to(device)
    inception_score = calculate_inception_score(generator, classifier, num_samples, batch_size, device)
    print(f"Inception Score: {inception_score}")
    real_features = extract_features(dataloader, classifier, device)
    fake_features = extract_features(None, classifier, device, is_generator=True, generator=generator, latent_dim=latent_dim, num_samples=num_samples, batch_size=batch_size)
    fid = calculate_fid(real_features, fake_features)
    print(f"FID: {fid}")
