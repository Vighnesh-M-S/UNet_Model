import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

# Map color names to RGB tensors
COLOR_TO_RGB = {
    "red":     [1, 0, 0],
    "green":   [0, 1, 0],
    "blue":    [0, 0, 1],
    "yellow":  [1, 1, 0],
    "cyan":    [0, 1, 1],
    "magenta": [1, 0, 1],
    "orange":  [1, 0.5, 0],
    "purple":  [0.5, 0, 0.5],
}

class ColorPolygonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

        # Load data.json which contains list of {"input": "circle.png", "color": "blue", "output": "blue_circle.png"}
        with open(os.path.join(root_dir, "data.json")) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Load polygon outline
        input_path = os.path.join(self.root_dir, "inputs", item["input_polygon"])
        outline = Image.open(input_path).convert("L")
        outline_tensor = self.transform(outline)  # (1, H, W)

        # 2. Prepare color hint as RGB (3, H, W)
        color_rgb = COLOR_TO_RGB[item["colour"]]
        color_tensor = torch.tensor(color_rgb).view(3, 1, 1).expand(-1, *outline_tensor.shape[1:])

        # 3. Load output target image (RGB)
        output_path = os.path.join(self.root_dir, "outputs", item["output_image"])
        output = Image.open(output_path).convert("RGB")
        output_tensor = self.transform(output)  # (3, H, W)

        # 4. Combine outline + color hint (1 + 3 = 4 channels)
        input_tensor = torch.cat([outline_tensor, color_tensor], dim=0)  # (4, H, W)

        return input_tensor, output_tensor
