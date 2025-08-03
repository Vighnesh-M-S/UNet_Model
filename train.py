import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from model import UNet
from dataset import PolygonColoringDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="polygon-coloring")

    config = wandb.config
    config.epochs = 25
    config.batch_size = 8
    config.lr = 1e-3

    train_dataset = PolygonColoringDataset(
        "dataset/training/data.json",
        "dataset/training/inputs",
        "dataset/training/outputs"
    )
    val_dataset = PolygonColoringDataset(
        "dataset/validation/data.json",
        "dataset/validation/inputs",
        "dataset/validation/outputs"
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()

        avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    train()