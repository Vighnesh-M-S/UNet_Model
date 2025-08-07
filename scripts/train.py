import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from my_datasets.color_polygon_dataset import ColorPolygonDataset
from models.DoubleUnet import UNet

# --- Config ---
DATA_DIR = "dataset/training"
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
SAVE_PATH = "checkpoints/color_polygon_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128  # for logging purposes

# --- wandb Init ---
wandb.init(
    project="color-polygon-unet",
    name="unet-colorization-run",
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "model": "UNet",
    }
)

# --- Dataset & Dataloader ---
dataset = ColorPolygonDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = UNet(in_channels=4, out_channels=3).to(DEVICE)

# --- Loss & Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    
    # Log metrics to wandb
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    # Log prediction image every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        sample_input, sample_target = next(iter(dataloader))
        sample_input = sample_input.to(DEVICE)
        sample_target = sample_target.to(DEVICE)

        with torch.no_grad():
            pred = model(sample_input)

        wandb.log({
            "prediction": [wandb.Image(pred[0].cpu().clamp(0, 1), caption="Prediction")],
            "target": [wandb.Image(sample_target[0].cpu(), caption="Target")],
            "input_mask": [wandb.Image(sample_input[0][0].unsqueeze(0).cpu(), caption="Input Mask")],
        })

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Model saved to {SAVE_PATH}")

# --- wandb Finish ---
wandb.finish()
