import os
import torch
import mteb
import yaml
import argparse

from mteb.types import PromptType
from mteb.abstasks.task_metadata import TaskMetadata
from transformers import BertTokenizer
from pathlib import Path
from models.bert_sentence_embedder import BertSentenceEmbedder


class BenchmarkEncoder:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        inputs, # DataLoader[BatchedInput]
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
   
        """Encodes the given sentences using the encoder.

        Args:
            inputs: The inputs to encode.
            task_metadata: The name of the task.
            hf_subset: The subset of the dataset.
            hf_split: The split of the dataset.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        batch_size = kwargs['batch_size']
        
        all_embeddings = []

        for batch in inputs:
            enc = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            embeddings = self.model(enc["input_ids"], enc["attention_mask"])
            all_embeddings.append(embeddings)


        return torch.cat(all_embeddings,dim=0).cpu()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_and_evaluate_all(cfg):
    """
    Function used to find all trained model weights files in a specified directory
    and evaluate using mteb. 
    """
    
    models_root = cfg.get("models_root")
    task_list = cfg.get("tasks")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tasks = mteb.get_tasks(tasks=task_list, languages=['eng'])

    model_files = list(Path(models_root).rglob("best_model.pt"))
    
    if not model_files:
        print("No models found.")
        return

    print(f"Found {len(model_files)} models. Starting evaluation...\n")

    for model_path in model_files:
        model_dir = model_path.parent
        model_name = model_dir.name
        model_config_path = model_dir / "local_config.yaml"
        
        if not model_config_path.exists():
            print(f"No config.yaml found in {model_dir}")
            continue

        try:
            model_cfg = load_config(model_config_path)
            pooling_mode = model_cfg.get("embedder").get("pooling")

            model = BertSentenceEmbedder(pooling=pooling_mode, cfg=model_cfg)
            model.load_state_dict(torch.load(str(model_path), weights_only=True))
            
            wrapped = BenchmarkEncoder(model, tokenizer)            
            
            output_dir = os.path.join(cfg.get("output_folder", "mteb_results"), model_name)
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(wrapped, output_folder=output_dir)
            
        except Exception as e:
            print(f"FAILED: {e}\n")
            continue
        
        # clean:
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTEB Evaluation script")
    parser.add_argument("--config", default="config.yaml", 
                        help="Path to config YAML (default: config.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    find_and_evaluate_all(cfg)
    print("All done.")