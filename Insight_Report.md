
# 🧠 Insights Report

### Model Architecture

- **Conditional UNet** with 4-channel input (3 for color hint + 1 for shape mask).
- Output is a 3-channel RGB image.
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam

### Data

- 8 shapes × 8 colors
- Input: polygon mask + color hint (encoded as 3-channel RGB)
- Target: filled polygon image

### Training Results

- Training converges steadily, with loss decreasing to below 0.07.
- Final visual outputs are accurate in both **color matching** and **shape boundaries**.

### WandB Monitoring

- Training metrics like loss per epoch are logged.
- Sample predictions (input mask, color hint, output, and target) are visualized.

### Challenges & Fixes

- Desaturation issue: resolved by explicitly feeding color hint as additional input.
- `wandb.Image` error: fixed by expanding 2D tensors to 3D.

---

