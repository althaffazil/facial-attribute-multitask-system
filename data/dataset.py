import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms


class CelebAMultiTaskDataset(Dataset):
    def __init__(self, split="train", image_size=128):
        self.dataset = load_dataset("abhiyanta/celebA-gender-smile", split=split)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample["image"].convert("RGB"))

        gender = torch.tensor(sample["gender"], dtype=torch.float32)
        smile = torch.tensor(sample["smiling"], dtype=torch.float32)

        return image, torch.stack([gender, smile])


def get_dataloader(split, config):
    dataset = CelebAMultiTaskDataset(split, config.IMAGE_SIZE)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
