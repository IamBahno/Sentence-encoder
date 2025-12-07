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
    def __init__(self, model, tokenizer, cfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        self.batch_size = self.cfg["training"]["batch_size"]
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        inputs,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type=None,
        **kwargs):
        
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
        
        all_embeddings = []

        for batch in inputs:
            enc = self.tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            embeddings = self.model(enc["input_ids"], enc["attention_mask"])
            all_embeddings.append(embeddings)


        return torch.cat(all_embeddings,dim=0).cpu()
    
    @torch.no_grad()
    def encode_queries(self, queries, **kwargs):
        """
        Encodes a list of query strings.
        MTEB calls this method for Retrieval tasks.
        """
        # Create batches of queries
        batched_inputs = []
        for i in range(0, len(queries), self.batch_size):
            batch_text = queries[i : i + self.batch_size]
            batched_inputs.append({"text": batch_text})

        return self.encode(inputs=batched_inputs)

    @torch.no_grad()
    def encode_corpus(self, corpus, **kwargs):
        """
        Encodes a list of corpus documents.
        MTEB calls this method for Retrieval tasks.
        """
        # Flatten corpus to a list of strings - concatenate title and text
        if isinstance(corpus[0], dict):
            sentences = [
                (doc.get("title", "") + " " + doc.get("text", "")).strip() 
                for doc in corpus
            ]
        else:
            sentences = corpus

        # Create batches
        batched_inputs = []
        for i in range(0, len(sentences), self.batch_size):
            batch_text = sentences[i : i + self.batch_size]
            batched_inputs.append({"text": batch_text})

        return self.encode(inputs=batched_inputs)

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
    # tasks = mteb.get_tasks(task_types=task_list, languages=['eng'])
    tasks = mteb.get_tasks(tasks=task_list, languages=['eng'])

    model_files = list(Path(models_root).rglob("best_model.pt"))
    
    if not model_files:
        print("No models found.")
        return

    print(f"Found {len(model_files)} models. Starting evaluation...\n")

    for idx, model_path in enumerate(model_files):
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
            
            wrapped = BenchmarkEncoder(model, tokenizer, cfg)            
            
            output_dir = os.path.join(cfg.get("output_folder", "mteb_results"), model_name)
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(wrapped, output_folder=output_dir)

            print(f"Finished testing {idx}. model. Saved to {output_dir}.")
            
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