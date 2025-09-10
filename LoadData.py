from PIL import Image
import os
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    """Custom dataset for scene classification"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Office', 'Kitchen', 'LivingRoom', 'Bedroom', 'Store', 
                       'Industrial', 'TallBuilding', 'InsideCity', 'Street', 
                       'Highway', 'Coast', 'OpenCountry', 'Mountain', 'Forest', 'Suburb']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label