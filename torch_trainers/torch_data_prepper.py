from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import transforms
from pathlib import Path
from PIL import Image
import PIL
import torch
from tqdm import tqdm

def generate_dataloaders(train_path, 
                         test_path, 
                         train_transform,
                         test_transform,
                         batch_size,
                         num_workers=4):

    
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader



def augmented_transform(mean, std):
    train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224,224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.TrivialAugmentWide(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std)
])
    return train_transform


def offline_augmenter(src_dir, out_dir, num_augments=5, split='train'):
    transform = transforms.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.TrivialAugmentWide()
    ])
    out_path = Path(out_dir)
    image_paths = datasets.ImageFolder(src_dir)
    for i_path, label in tqdm(image_paths.samples, desc='Augmenting images'):
        class_name = image_paths.classes[label]
        
        save_dir = out_path / split / class_name
        save_dir.mkdir(exist_ok=True, parents=True)
        img_name = Path(i_path).name
    
        #Load
        img = Image.open(i_path)
        
        # resize
        img = img.resize((224,224), resample=PIL.Image.LANCZOS)
        img_save_dir = save_dir / (img_name.split('.jpg')[0]+".png")
        img.save(img_save_dir, format='PNG')
        # make 5 augmentations
        for i in range(num_augments):
            aug_img = transform(img)
            new_img_name = img_name.split('.jpg')[0] + f"_augmented_{i}.png"
            new_img_path = save_dir / new_img_name
            aug_img.save(new_img_path, format='PNG')