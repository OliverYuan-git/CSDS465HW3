import os
import torch
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.models as models

class GeneratedImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('.png', '.jpg'))]
        self.transform = transform if transform else Compose([
            Resize((299, 299)),  # Resize for Inception
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img

def calculate_inception_score(img_dir, batch_size=32, device='cuda'):
    # Load Inception model
    classifier = models.inception_v3(pretrained=True, transform_input=False)
    classifier.fc = torch.nn.Identity()  # Remove the classification head
    classifier.to(device)
    classifier.eval()

    # Load images
    dataset = GeneratedImageDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []

    # Run images through the Inception model
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Calculating Inception Score"):
            imgs = imgs.to(device)
            preds = torch.softmax(classifier(imgs), dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)
    mean_preds = np.mean(all_preds, axis=0)
    kl_div = all_preds * (np.log(all_preds + 1e-10) - np.log(mean_preds + 1e-10))
    inception_score = np.exp(np.mean(np.sum(kl_div, axis=1)))

    return inception_score


# -------- Main --------
if __name__ == "__main__":
    # Set parameters
    img_dir = "samples"  # Folder containing saved images
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate and print Inception Score
    print("Calculating Inception Score...")
    inception_score = calculate_inception_score(img_dir, batch_size, device)
    print(f"Inception Score: {inception_score}")
