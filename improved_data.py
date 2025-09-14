import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import img_paths
import numpy as np


class ImprovedOAS2Data(Dataset):
    def __init__(self, converted=False, size=1, data_type='train', seed=310):
        # Paths to images
        self.train_path = img_paths.train_paths
        self.test_path = img_paths.test_paths
        self.eval_path = img_paths.eval_paths
        # Paths to labels
        self.train_lab_path = 'OAS2_labels_cleaned.csv'
        self.test_lab_path = 'OAS2_test_eval_dst.csv'
        # Loads data and labels
        self.labs = pd.read_csv(self.train_lab_path) if data_type == 'train' else pd.read_csv(self.test_lab_path).iloc[:50] if data_type == 'test' else pd.read_csv(self.test_lab_path).iloc[50:101]
        self.labels, self.data = self.create_tensor_labels(converted, size, data_type)
        self.classes = self.set_classes(converted)
        # Whether or not to converted patients
        self.conv = converted

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label
    
    def create_tensor_labels(self, converted, size, data_type):
        torch.manual_seed(310)
        labels = []
        img = []

        data_to_use = self.train_path if data_type == 'train' else self.test_path if data_type == 'test' else self.eval_path

        # Improved transformations for medical images
        # More conservative augmentations for training data
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # Gentle augmentations that preserve medical image characteristics
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
            transforms.RandomRotation(degrees=5),     # Reduced rotation
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.02, 0.02),  # Reduced translation
                scale=(0.95, 1.05),      # Reduced scaling
                shear=(-2, 2)            # Gentle shear
            ),
            # Better normalization for medical images
            transforms.Normalize(mean=[0.485], std=[0.229])  # Using ImageNet stats
        ])
        
        # Transformations for test and eval data
        test_eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        for _, row in self.labs.iterrows():
            mri_id = row['MRI ID']
            label = row['Group_Code']
            if label == 2 and converted == False:
                continue
            for x in range(23, 99, 2):
                image_path = []
                for i in range(size):
                    image_path.append(f'{data_to_use[i]}\\{mri_id}_{x}.png')
                    image = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Preprocessing: enhance contrast and normalize
                        image = self.preprocess_image(image)
                        
                        tensor = train_transform(image) if data_type == 'train' else test_eval_transform(image)
                        img.append(tensor)
                        labels.append(label)

        print(f'Loaded Labels and Images for {data_type}')

        return labels, img

    def preprocess_image(self, image):
        """Preprocess medical image for better training"""
        # Convert to float
        image = image.astype(np.float32)
        
        # Normalize to 0-1 range
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply((image * 255).astype(np.uint8))
        
        # Convert back to float and normalize
        image = image.astype(np.float32) / 255.0
        
        return image

    def set_classes(self, converted):
        if converted == False:
            return ['Nondemented', 'Demented']
        else:
            return ['Nondemented', 'Demented', 'Converted']


# Alternative data loading with more aggressive preprocessing
class AdvancedOAS2Data(Dataset):
    def __init__(self, converted=False, size=1, data_type='train', seed=310):
        # Paths to images
        self.train_path = img_paths.train_paths
        self.test_path = img_paths.test_paths
        self.eval_path = img_paths.eval_paths
        # Paths to labels
        self.train_lab_path = 'OAS2_labels_cleaned.csv'
        self.test_lab_path = 'OAS2_test_eval_dst.csv'
        # Loads data and labels
        self.labs = pd.read_csv(self.train_lab_path) if data_type == 'train' else pd.read_csv(self.test_lab_path).iloc[:50] if data_type == 'test' else pd.read_csv(self.test_lab_path).iloc[50:101]
        self.labels, self.data = self.create_tensor_labels(converted, size, data_type)
        self.classes = self.set_classes(converted)
        self.conv = converted

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label
    
    def create_tensor_labels(self, converted, size, data_type):
        torch.manual_seed(310)
        labels = []
        img = []

        data_to_use = self.train_path if data_type == 'train' else self.test_path if data_type == 'test' else self.eval_path

        # Advanced transformations with more sophisticated augmentations
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # More sophisticated augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Sometimes useful for medical images
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(-5, 5)
            ),
            # Add noise for robustness
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        test_eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        for _, row in self.labs.iterrows():
            mri_id = row['MRI ID']
            label = row['Group_Code']
            if label == 2 and converted == False:
                continue
            for x in range(23, 99, 2):
                image_path = []
                for i in range(size):
                    image_path.append(f'{data_to_use[i]}\\{mri_id}_{x}.png')
                    image = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Advanced preprocessing
                        image = self.advanced_preprocess_image(image)
                        
                        tensor = train_transform(image) if data_type == 'train' else test_eval_transform(image)
                        img.append(tensor)
                        labels.append(label)

        print(f'Loaded Labels and Images for {data_type}')

        return labels, img

    def advanced_preprocess_image(self, image):
        """Advanced preprocessing for medical images"""
        # Convert to float
        image = image.astype(np.float32)
        
        # Remove outliers (very bright or very dark pixels)
        p5, p95 = np.percentile(image, (5, 95))
        image = np.clip(image, p5, p95)
        
        # Normalize to 0-1 range
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image = clahe.apply((image * 255).astype(np.uint8))
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Convert back to float and normalize
        image = image.astype(np.float32) / 255.0
        
        return image

    def set_classes(self, converted):
        if converted == False:
            return ['Nondemented', 'Demented']
        else:
            return ['Nondemented', 'Demented', 'Converted']


# Example usage
if __name__ == "__main__":
    # Test the improved data loading
    print("Testing improved data loading...")
    
    # Test basic improved version
    train_data = ImprovedOAS2Data(size=1, data_type='train')
    test_data = ImprovedOAS2Data(size=1, data_type='test')
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Test advanced version
    train_data_adv = AdvancedOAS2Data(size=1, data_type='train')
    test_data_adv = AdvancedOAS2Data(size=1, data_type='test')
    
    print(f"Advanced train samples: {len(train_data_adv)}")
    print(f"Advanced test samples: {len(test_data_adv)}")

