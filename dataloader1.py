import re
import os
import fnmatch
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from device import fetchDevice

def list_files_by_type(folder_path, file_type):
    filtered_files = []
    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, f"*.{file_type}"):
            filtered_files.append(os.path.join(folder_path, file))
    return filtered_files

def process_image(args):
    file_path, label, transform = args  # Unpack the tuple
    image = Image.open(file_path)
    default_transform = transforms.ToTensor()
    tensor = default_transform(image)
    if transform:
        tensor = transform(tensor)
    return tensor, torch.tensor(label, dtype=torch.long)

class CIFAKEDataset(Dataset):
    @staticmethod
    def extract_index_and_category(file_path):
        filename = os.path.basename(file_path)
        pattern = r"(\d+)(?: \((\d+)\))?\..+"
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            category = int(match.group(2)) if match.group(2) else 0
            return index, category
        else:
            return None
    
    @staticmethod
    def load_folder(folder_path, label, category=None, transform=None, num_processes=1):
        print(f"Loading folder: {folder_path}")
        files = list_files_by_type(folder_path, "jpg")
        if category is not None:
            files = [file for file in files if CIFAKEDataset.extract_index_and_category(file)[1] == category]

        # Use process_map from tqdm.contrib.concurrent for better tqdm updates
        results = thread_map(process_image, [(file, label, transform) for file in files], max_workers=num_processes, chunksize=1)

        x = torch.stack([result[0] for result in results])
        y = torch.stack([result[1] for result in results])
        return x, y
        
    def __init__(self, category=None, transform=None, num_processes=1):
        label_0_folders = [
            "outputJPG/SD21Airplane",
            "outputJPG/SD21Automobile",
            "outputJPG/SD21Bird",
            "outputJPG/SD21Cat",
            "outputJPG/SD21Deer",
            "outputJPG/SD21Dog",
            "outputJPG/SD21Frog",
            "outputJPG/SD21Horse",
            "outputJPG/SD21Ship",
            "outputJPG/SD21Truck" 
        ]
        label_1_folders = [
            "CIFAKE/test/REAL",
            "CIFAKE/train/REAL"
        ]
        x1, y1 = CIFAKEDataset.load_folder(label_1_folders[0], 1, category, transform, num_processes)
        x2, y2 = CIFAKEDataset.load_folder(label_1_folders[1], 1, category, transform, num_processes)

        xList = [0]*10
        yList = [0]*10
        for i in range(10):
            xi, yi = CIFAKEDataset.load_folder(label_0_folders[i], 0, category, transform, num_processes)
            xList[i] = xi
            yList[i] = yi

        
        self.x = torch.cat(tuple(xList) + (x1, x2))
        self.y = torch.cat(tuple(yList) + (y1, y2))
        self.x = self.x.to(fetchDevice())
        self.y = self.y.to(fetchDevice())

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def data_dim(self):
        return self.x[0].size()
