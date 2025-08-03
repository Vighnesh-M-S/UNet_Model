import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json

color_map = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0],
    "purple": [0.5, 0.0, 0.5],
}

class PolygonColoringDataset(Dataset):
    def __init__(self, json_path, inputs_dir, outputs_dir, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_img = Image.open(os.path.join(self.inputs_dir, item["input_polygon"])).convert("RGB")
        output_img = Image.open(os.path.join(self.outputs_dir, item["output_image"])).convert("RGB")

        input_tensor = self.transform(input_img)
        output_tensor = self.transform(output_img)

        color_vec = torch.tensor(color_map[item["colour"]]).view(3, 1, 1)
        color_tensor = color_vec.expand(-1, input_tensor.shape[1], input_tensor.shape[2])
        input_conditioned = torch.cat([input_tensor, color_tensor], dim=0)

        return input_conditioned, output_tensor