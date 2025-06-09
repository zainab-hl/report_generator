import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer
from transformers.optimization import get_scheduler
from torch.optim import AdamW
import os
import sys
from tqdm.auto import tqdm
import json 
import warnings 
import math # Import math for ceiling division
import argparse # For parsing command-line arguments

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from models.trained_models.Q_former.q_former import BertConfig
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

# --- Configuration for Training ---
class TrainingConfig:
    def __init__(self):
        self.dataset_dir = "/content/drive/MyDrive/dataLLM2" 
        self.output_dir = "/content/drive/MyDrive/finetuned_report_generator_distributed"
        self.max_seq_length = 256
        self.train_batch_size = 4
        self.eval_batch_size = 8
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 0.1 
        self.gradient_accumulation_steps = 1
        # New: Parameters for data partitioning
        self.worker_id = 0  # Default to worker 0
        self.num_workers = 1 # Default to 1 worker (no partitioning)

class ReportGenerationDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: BioGptTokenizer, max_seq_length: int = 256, 
                 start_idx: int = None, end_idx: int = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = [] # This will store all loaded data from all JSON files

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        print(f"Loading data from directory: {data_dir}")
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')]) # Ensure consistent order
        
        # Apply slicing based on start_idx and end_idx
        files_to_load = all_files[start_idx:end_idx] if start_idx is not None and end_idx is not None else all_files

        print(f"Loading {len(files_to_load)} files from {data_dir} (indices {start_idx}-{end_idx})...")

        for filename in files_to_load:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    entry = json.load(f)
                    self.data.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filepath}. Skipping.")
            except Exception as e:
                print(f"Warning: Error loading {filepath}: {e}. Skipping.")
        
        if not self.data:
            raise ValueError(f"No data loaded from {data_dir} for the specified range. Check directory path, file formats, and indices.")
        print(f"Successfully loaded {len(self.data)} samples for this worker.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Image embeddings (convert list to tensor)
        image_embedding = torch.tensor(item["embedding"], dtype=torch.float32)

        # Report text tokenization
        tokenized_report = self.tokenizer(
            self.tokenizer.bos_token + item["report"] + self.tokenizer.eos_token,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = tokenized_report.input_ids.squeeze(0)
        attention_mask = tokenized_report.attention_mask.squeeze(0)
        
        return {
            "image_embedding": image_embedding,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

# --- Main Training Function ---
def train_model(worker_id: int = 0, num_workers: int = 1):
    config = TrainingConfig()
    config.worker_id = worker_id
    config.num_workers = num_workers
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create a worker-specific output directory to save models
    worker_output_dir = os.path.join(config.output_dir, f"worker_{worker_id}")
    os.makedirs(worker_output_dir, exist_ok=True)
    print(f"Saving worker {worker_id} outputs to: {worker_output_dir}")

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")

    # Determine the data slice for this worker
    all_json_files = sorted([f for f in os.listdir(config.dataset_dir) if f.endswith('.json')])
    total_files = len(all_json_files)
    
    if total_files == 0:
        raise ValueError(f"No JSON files found in {config.dataset_dir}. Please check your dataset directory.")

    files_per_worker = math.ceil(total_files / config.num_workers) # Use ceil to ensure all data is covered
    start_idx = config.worker_id * files_per_worker
    end_idx = min(start_idx + files_per_worker, total_files)

    print(f"Worker {config.worker_id}/{config.num_workers} will process data from index {start_idx} to {end_idx-1}.")

    train_dataset = ReportGenerationDataset(
        config.dataset_dir, tokenizer, config.max_seq_length, 
        start_idx=start_idx, end_idx=end_idx
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

    # Initialize Q-Former Config
    qformer_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        encoder_width=512, 
        num_query_tokens=32,
        add_cross_attention=True,
        cross_attention_freq=1,
    )

    # Initialize Model
    print("Initializing XrayReportGenerator for training...")
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        biogpt_weights_path=MODEL_WEIGHTS['biogpt'],
        qformer_config=qformer_config
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Learning Rate Scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_steps)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # --- Training Loop ---
    print("Starting training...")
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0

    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            # Move batch to device
            image_embedding = batch["image_embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            loss = model(
                image_path=None,  
                image_features=image_embedding,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (completed_steps + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps % 100 == 0:
                print(f"Step {completed_steps}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")

        print(f"Epoch {epoch+1} finished for worker {config.worker_id}. Saving model...")
        save_path = os.path.join(worker_output_dir, f"report_generator_epoch_{epoch+1}_worker_{config.worker_id}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    print(f"Training complete for worker {config.worker_id}!")

    final_save_path = os.path.join(worker_output_dir, f"report_generator_final_worker_{config.worker_id}.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

    # It's good practice to save the tokenizer in each worker's directory too,
    # or ensure it's saved once centrally. For simplicity, saving per worker.
    tokenizer.save_pretrained(worker_output_dir)
    print(f"Tokenizer saved to {worker_output_dir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_sys_path = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    if project_root_for_sys_path not in sys.path:
        sys.path.insert(0, project_root_for_sys_path)
        print(f"Added {project_root_for_sys_path} to sys.path")

    parser = argparse.ArgumentParser(description="Train Xray Report Generator with data partitioning.")
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Unique ID for this worker (0-indexed).')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Total number of workers participating in training.')
    
    args = parser.parse_args()

    train_model(worker_id=args.worker_id, num_workers=args.num_workers)