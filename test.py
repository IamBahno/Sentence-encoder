import torch
import numpy as np
import mteb

from mteb.types import PromptType,BatchedInput
from mteb.abstasks.task_metadata import TaskMetadata
from transformers import BertTokenizer

from models import BertSentenceEmbedder

# MTEB expects model with 'encode' method, that takes in list of sentences, and returns numpy embeddings
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


if __name__ == "__main__":

    model = BertSentenceEmbedder(pooling="mean")
    model_name = "triplet_max_pooling_bert"
    model_path = "runs\\triplet_max_pooling_bert\\last_model.pt"
    model.load_state_dict(torch.load(model_path,weights_only=True))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    wrapped = BenchmarkEncoder(model, tokenizer)

    # task_types=["Clustering", "Retrieval"]
    # tasks = mteb.get_tasks(tasks=["STS16"])
    # tasks = mteb.get_tasks(task_types=["Clustering", "Classification","Reranking","Retrieval","STS","Sumarization"],languages=['eng'])
    # tasks = mteb.get_tasks(task_types=["Clustering", "Classification","STS",],languages=['eng'])
    tasks = mteb.get_tasks(task_types=['STS'],languages=['eng'])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(wrapped, output_folder=model_name)