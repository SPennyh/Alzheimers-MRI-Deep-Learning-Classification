import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import img_paths


class OAS2Data(Dataset):
    def __init__(self, converted=False, size=1, data_type='train', seed=310):
        # Paths to images
        self.train_path = img_paths.train_paths
        self.test_path = img_paths.test_paths
        self.eval_path = img_paths.eval_paths
        # Paths to labels
        self.train_lab_path = 'OAS2_labels_cleaned.csv'
        self.test_lab_path = 'OAS2_test_eval_dst.csv'
        # Loads data and labels
        self.labs = pd.read_csv(self.train_lab_path) if data_type == 'train' else pd.read_csv(self.test_lab_path).iloc[:50] if data_type == 'test' else pd.read_csv(self.test_lab_path).iloc[50:101]  # Assuming eval data is the second half of the test data
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

        # Transformations for training data
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Transformations for test and eval data
        test_eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        for _, row in self.labs.iterrows(): # Iterate through all rows of labels csv
            mri_id = row['MRI ID']
            label = row['Group_Code']
            if label == 2 and converted == False: # Remove the converted labels if these conditions are met
                continue
            for x in range(23, 99, 2):
                image_path = []
                for i in range(size):
                    image_path.append(f'{data_to_use[i]}\\{mri_id}_{x}.png')
                    image = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        tensor = train_transform(image) if data_type == 'train' else test_eval_transform(image) # Transforming and converting image to tensor
                        img.append(tensor)
                        labels.append(label)
                    

        print(f'Loaded Labels and Images for {data_type}')

        return labels, img

    def set_classes(self, converted):
        if converted == False:
            return ['Nondemented', 'Demented']
        else:
            return ['Nondemented', 'Demented', 'Converted']
    
# data = OAS2Data(size=1, data_type='test')
# image, label= data[0]
# print(len(data.data))
# print(len(data.labels))

# for i in range(len(data.data)):
#     image, label= data[i]
#     print(f'label: {label}')
