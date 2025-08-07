import os
import torch
from PIL import Image
import torchvision.transforms as T
from models.DoubleUnet import UNet

# --- Config ---
CHECKPOINT_PATH = "checkpoints/color_polygon_unet.pth"
INPUT_MASKS_DIR = "dataset/training/inputs"
SAVE_DIR = "predictions"
IMAGE_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Color Map ---
def color_to_rgb(color_name):
    COLORS = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128)
    }
    if color_name not in COLORS:
        raise ValueError(f"Color '{color_name}' not recognized.")
    return COLORS[color_name]

# --- Transform ---
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])

# --- Model ---
model = UNet(in_channels=4, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# --- Helper: predict given shape & color ---
def predict(shape_name: str, color_name: str):
    # Load and process mask
    mask_path = os.path.join(INPUT_MASKS_DIR, f"{shape_name}.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"No mask found for shape: {shape_name}")
    
    mask_tensor = transform(Image.open(mask_path).convert("L"))  # (1, H, W)

    # Prepare color tensor
    rgb = color_to_rgb(color_name)  # (R, G, B)
    color_tensor = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(-1, IMAGE_SIZE, IMAGE_SIZE) / 255.0

    # Combine inputs
    input_tensor = torch.cat([mask_tensor, color_tensor], dim=0).unsqueeze(0).to(DEVICE)  # (1, 4, H, W)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)[0].cpu()  # (3, H, W)

    # Save output
    os.makedirs(SAVE_DIR, exist_ok=True)
    output_image = T.ToPILImage()(output.clamp(0, 1))
    save_path = os.path.join(SAVE_DIR, f"{shape_name}_{color_name}_pred.png")
    output_image.save(save_path)
    print(f"✅ Saved prediction to {save_path}")

# --- Example Usage ---
if __name__ == "__main__":
    predict("circle", "blue")
    predict("star", "red")
