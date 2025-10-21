import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, save_dir, title="Training and Validation Loss"):
    """
    Plots training and validation loss curves and saves the figure to the given directory.

    Args:
        train_losses (list[float]): List of training losses per epoch.
        val_losses (list[float]): List of validation losses per epoch.
        save_dir (str): Directory path where to save the figure (created if missing).
        title (str): Title of the plot (default: 'Training and Validation Loss').
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "loss_curve.png")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to: {save_path}")