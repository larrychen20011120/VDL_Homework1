import os
import torch
from data import HundredClassDataset
from model import HundredClassResNet

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data.sampler import WeightedRandomSampler

from tqdm import tqdm
import random


def sample_n_per_class(label2paths, n_samples_per_class=3):

    sampled_label2paths = {}
    remaining_label2paths = {}

    for label, paths in label2paths.items():
        random.seed(0)
        # Ensure we don't sample more than available samples
        sampled_paths = random.sample(paths, min(n_samples_per_class, len(paths)))
        remaining_paths = [path for path in paths if path not in sampled_paths]

        sampled_label2paths[label] = sampled_paths
        remaining_label2paths[label] = remaining_paths
    
    return sampled_label2paths, remaining_label2paths


def train(dataloader_train, dataloader_val, model, optimizer, scheduler, criterion, device, epochs=50, accumulate_steps=1, log_name="weights_resnet50"):
    
    if not os.path.exists(log_name):
        os.makedirs(log_name)  

    # mix_aug = RandomMixAugment(num_classes=100, p_mix=0.5, alpha=1.0)
    cutmix = v2.CutMix(num_classes=100)
    mixup = v2.MixUp(num_classes=100)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    model.to(device)
    model.train()
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        
        for step, (images, labels) in enumerate(dataloader_train):

            # images, labels = mix_aug(images, labels)
            images, labels = cutmix_or_mixup(images, labels)
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if (step + 1) % accumulate_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += (loss.item() * accumulate_steps)

        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader_train):.4f}")
        accuracy = val(dataloader_val, model, criterion, device)
        torch.save(model.state_dict(), f"{log_name}/{epoch}_{int(accuracy*100)}.pth")

def val(dataloader, model, criterion, device):

    model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct/total
    print(f"Validation Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy


import argparse

if __name__ == "__main__":

    ### add arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="step")
    parser.add_argument("--weighted_sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--log_name", type=str, default="weights_resnet50")
    
    # parse the arguments
    args = parser.parse_args()
    
    # define the number of classes, the device and the entry directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 100
    entry = os.path.join('hw1-data', 'data')


    ### load the data
    train_path = os.path.join(entry, 'train')
    val_path = os.path.join(entry, 'val')

    label2paths_train = {
        i: [os.path.join(train_path, str(i), f) for f in os.listdir(os.path.join(train_path, str(i)))]
        for i in range(n_classes)
    }
    label2paths_val = {
        i: [os.path.join(val_path, str(i), f) for f in os.listdir(os.path.join(val_path, str(i)))]
        for i in range(n_classes)
    }
    
    # Combine train and val label2paths into one dictionary
    combined_label2paths = {i: label2paths_train[i] + label2paths_val[i] for i in range(n_classes)}

    # Sample 5 images per class
    sampled_label2paths, remaining_label2paths = sample_n_per_class(combined_label2paths, n_samples_per_class=5)


    dataset_train = HundredClassDataset(label2paths=remaining_label2paths, split="train")
    dataset_val = HundredClassDataset(label2paths=sampled_label2paths, split="val")

    if args.weighted_sampler:
        sample_weights = dataset_train.get_sample_weights()
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=sampler)
    else:
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)

    model = HundredClassResNet()
    
    # split the parameters into two groups: feature_params and fc_params
    feature_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            fc_params.append(param)
        else:
            feature_params.append(param)

    optimizer = optim.Adam([
        {'params': feature_params, 'lr': args.backbone_lr},
        {'params': fc_params, 'lr': args.head_lr}
    ], weight_decay=args.weight_decay)

    # 2. Scheduler: Cosine Annealing
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    train(dataloader_train, dataloader_val, model, optimizer, scheduler, criterion, device, args.epochs, args.accumulate_steps, args.log_name)