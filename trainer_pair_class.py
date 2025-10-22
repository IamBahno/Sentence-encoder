import os
import torch
from utils.plot import plot_losses


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, batch_size,run_name="tmp"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.run_name=run_name

        # Save path and tracking
        self.best_val_loss = float('inf')

        os.makedirs("runs", exist_ok=True)
        self.target_folder = os.path.join("runs",run_name)
        if os.path.exists(self.target_folder):
            print(f"Folder '{self.target_folder}' already exists, and its content will be overwritten !!! — continuing run.")
        os.makedirs(self.target_folder, exist_ok=True)


    def train(self,epochs):
        self.model.train()
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            total_epoch_loss = 0
            for enc_premise, enc_hypothesis, targets in self.train_loader:
                # Extract input_ids and attention_mask for each and move to device
                input_ids_premise, attn_premise = enc_premise["input_ids"].to(self.device), enc_premise["attention_mask"].to(self.device)
                input_ids_hypothesis, attn_hypothesis = enc_hypothesis["input_ids"].to(self.device), enc_hypothesis["attention_mask"].to(self.device)
                targets = targets.to(self.device)

                # Forward pass through the model = (bert embedder -> classifier)
                logits = self.model(input_ids_premise, attn_premise, input_ids_hypothesis, attn_hypothesis)

                # Compute cross entropy loss
                loss = self.criterion(logits, targets)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_epoch_loss += loss.item()

            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            train_losses.append(avg_epoch_loss)
            val_losses.append(val_loss)

            # --- checkpoint saving ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path_embedder = f"{self.target_folder}/best_model.pt"
                best_path_whole_model = f"{self.target_folder}/best_whole_model.pt"
                torch.save(self.model.embedder.state_dict(), best_path_embedder)
                torch.save(self.model.state_dict(), best_path_whole_model)
                print(f"Saved best model (val loss {self.best_val_loss:.4f})")

        last_path_embedder = f"{self.target_folder}/last_model.pt"
        last_path_whole_model = f"{self.target_folder}/last_model_whole_model.pt"
        torch.save(self.model.embedder.state_dict(), last_path_embedder)
        torch.save(self.model.state_dict(), last_path_whole_model)
        print(f"Saved last model (val loss {val_loss:.4f})")

        plot_losses(train_losses, val_losses, save_dir=self.target_folder)


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for enc_premise, enc_hypothesis, target in self.val_loader:
                input_ids_premise, attn_premise = enc_premise["input_ids"].to(self.device), enc_premise["attention_mask"].to(self.device)
                input_ids_hypothesis, attn_hypothesis = enc_hypothesis["input_ids"].to(self.device), enc_hypothesis["attention_mask"].to(self.device)
                targets = targets.to(self.device)

                logits = self.model(input_ids_premise, attn_premise, input_ids_hypothesis, attn_hypothesis)
                loss = self.criterion(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.model.train()
        return avg_val_loss