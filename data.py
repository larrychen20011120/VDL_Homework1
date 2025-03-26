import os
from PIL import Image
from collections import Counter

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

## create the dataset class without any augmentation
class HundredClassDataset(Dataset):
    def __init__(self, label2paths=dict(), split="train"):
        self.paths = []
        self.labels = []

        if split == "train" or split == "val":
            for label, paths in label2paths.items():
                for path in paths:
                    self.paths.append(path)
                    self.labels.append(int(label))

        else:
            entry = os.path.join('hw1-data', 'data', 'test')
            self.labels = [0] * len(os.listdir(entry))
            self.paths = [os.path.join(entry, filename) for filename in os.listdir(entry)]

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((324, 324)),
                transforms.CenterCrop((300, 300)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.05, 0.15), value='random'),  # 隨機擦除
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((324, 324)),
                transforms.CenterCrop((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])
        # get item then load the image
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label
    
    def get_file_names(self):
        return [path.split("/")[-1].split(".")[0] for path in self.paths]
    
    def get_class_distribution(self):
        return dict(Counter(self.labels))
    
    def get_sample_weights(self):
        class_distribution = self.get_class_distribution()
        total_samples = len(self.labels)
        
        # Calculate class weights as inverse of class frequency
        class_weights = {k: total_samples / v for k, v in class_distribution.items()}
        
        # Now, assign weights to each sample
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights

if __name__ == "__main__":
    entry = os.path.join('hw1-data', 'data')
    train_path = os.path.join(entry, 'train')

    label2paths = {
        i: [ os.path.join(train_path, str(i), f) for f in os.listdir(os.path.join(train_path, str(i))) ]
        for i in range(100)
    }

    dataset = HundredClassDataset(label2paths)
    print(len(dataset))
    print(dataset[0][0].shape)