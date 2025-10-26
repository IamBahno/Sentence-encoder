import os
import torch
from utils.plot import plot_losses


class Trainer:
    def __init__(self, model, train_loader,val_loader, criterion, optimizer, device, batch_size,run_name="tmp"):
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
            total_epoch_loss = 0
            for enc_anchor, enc_pos, enc_neg in self.train_loader:
                # Extract input_ids and attention_mask for each
                input_ids_a, attn_a = enc_anchor["input_ids"], enc_anchor["attention_mask"]
                input_ids_p, attn_p = enc_pos["input_ids"], enc_pos["attention_mask"]
                input_ids_n, attn_n = enc_neg["input_ids"], enc_neg["attention_mask"]

                # Concatenate input_ids and attention_mask for single forward pass
                all_input_ids = torch.cat([input_ids_a, input_ids_p, input_ids_n], dim=0).to(self.device)
                all_attention_mask = torch.cat([attn_a, attn_p, attn_n], dim=0).to(self.device)

                # Forward pass
                all_embeddings = self.model(all_input_ids, all_attention_mask)
                # Split embeddings back into anchor, positive, negative
                emb_a = all_embeddings[:self.batch_size]
                emb_p = all_embeddings[self.batch_size:2*self.batch_size]
                emb_n = all_embeddings[2*self.batch_size:]

                # Compute triplet loss
                loss = self.criterion(emb_a, emb_p, emb_n)

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
                best_path = f"{self.target_folder}/best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best model (val loss {self.best_val_loss:.4f})")

        last_path = f"{self.target_folder}/last_model.pt"
        torch.save(self.model.state_dict(), last_path)
        print(f"Saved last model (val loss {val_loss:.4f})")

        plot_losses(train_losses, val_losses, save_dir=self.target_folder)


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for enc_anchor, enc_pos, enc_neg in self.val_loader:
                input_ids_a, attn_a = enc_anchor["input_ids"], enc_anchor["attention_mask"]
                input_ids_p, attn_p = enc_pos["input_ids"], enc_pos["attention_mask"]
                input_ids_n, attn_n = enc_neg["input_ids"], enc_neg["attention_mask"]

                all_input_ids = torch.cat([input_ids_a, input_ids_p, input_ids_n], dim=0).to(self.device)
                all_attention_mask = torch.cat([attn_a, attn_p, attn_n], dim=0).to(self.device)

                all_embeddings = self.model(all_input_ids, all_attention_mask)
                emb_a = all_embeddings[:self.batch_size]
                emb_p = all_embeddings[self.batch_size:2*self.batch_size]
                emb_n = all_embeddings[2*self.batch_size:]

                loss = self.criterion(emb_a, emb_p, emb_n)
                total_val_loss += loss.item()


        avg_val_loss = total_val_loss / len(self.val_loader)
        self.model.train()
        return avg_val_loss
    