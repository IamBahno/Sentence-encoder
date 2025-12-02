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

    def forward_pass(self, enc_anchors, enc_candidates):
        # Extract input_ids and attention_mask for each and move to device
        input_ids_anchor, attn_anchor = enc_anchors["input_ids"], enc_anchors["attention_mask"]
        input_ids_anchor_pairs, attn_anchor_pairs = enc_candidates["input_ids"], enc_candidates["attention_mask"]
        # Concatenate input ids and attentions to compute embeddings in one forward pass:
        all_input_ids = torch.cat([input_ids_anchor, input_ids_anchor_pairs], dim=0).to(self.device)
        all_attention_mask = torch.cat([attn_anchor, attn_anchor_pairs], dim=0).to(self.device)
        # get anchor and anchor_pairs embeddings:
        all_embeddings = self.model(all_input_ids, all_attention_mask)
        # Split back:
        emb_anchors = all_embeddings[:self.batch_size]
        emb_anchor_pairs = all_embeddings[self.batch_size:]


        # NEW cosine similarity
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        # Expand emb_anchor_pairs to [1, batch_size, 768] for broadcasting over batch_size
        pairs_expanded = emb_anchor_pairs.unsqueeze(0)  # [1, batch_size, 768]

        # Expand emb_anchors to [batch_size, 1, 768] for broadcasting over batch_size pairs
        anchors_expanded = emb_anchors.unsqueeze(1)    # [batch_size, batch_size, 768]
        # Compute cosine similarity over last dim → result shape: [batch_size, batch_size]
        cos_sim_scores = cos_sim(anchors_expanded, pairs_expanded)


        return cos_sim_scores

    def train(self,epochs):
        self.model.train()
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            total_epoch_loss = 0
            for enc_anchors, enc_anchor_candidates in self.train_loader:
                cos_sim_scores = self.forward_pass(enc_anchors, enc_anchor_candidates)
                targets = torch.tensor(range(len(cos_sim_scores)), dtype=torch.long, device=cos_sim_scores.device)
                loss = self.criterion(cos_sim_scores, targets)
                
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
            for enc_anchor, enc_anchor_pairs in self.val_loader:
                cos_sim_scores = self.forward_pass(enc_anchor, enc_anchor_pairs)
                targets = torch.tensor(range(len(cos_sim_scores)), dtype=torch.long, device=cos_sim_scores.device)

                loss = self.criterion(cos_sim_scores, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.model.train()
        return avg_val_loss