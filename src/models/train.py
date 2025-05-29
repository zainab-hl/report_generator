import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer
from transformers.optimization import get_scheduler 
from torch.optim import AdamW 
import os
import sys
from tqdm.auto import tqdm 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Added an extra '..'
sys.path.insert(0, project_root)

from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from models.trained_models.Q_former.q_former import BertConfig 
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS 

class TrainingConfig:
    def __init__(self):
        self.dataset_path = "/content/drive/MyDrive/dataLLM" 
        self.output_dir = "/content/drive/MyDrive/finetuned_report_generator" 
        self.max_seq_length = 256 
        self.train_batch_size = 4 
        self.eval_batch_size = 8
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 0.1 
        self.gradient_accumulation_steps = 1 

class ReportGenerationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BioGptTokenizer, max_seq_length: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_embedding = torch.tensor(item["embedding"], dtype=torch.float32)

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

def train_model():
    config = TrainingConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")

    train_dataset = ReportGenerationDataset(config.dataset_path, tokenizer, config.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

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

    print("Initializing XrayReportGenerator for training...")
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        biogpt_weights_path=MODEL_WEIGHTS['biogpt'], 
        qformer_config=qformer_config
    ).to(device)

    # BioMedClip is well fine-tuned and tested
    for param in model.biomedclip_encoder.parameters():
        param.requires_grad = False
    print("BiomedCLIPEncoder parameters frozen.")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_steps)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    print("Starting training...")
    model.train() 
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0

    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
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

        print(f"Epoch {epoch+1} finished. Saving model...")
        save_path = os.path.join(config.output_dir, f"xray_report_generator_epoch_{epoch+1}.pth") 
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    print("Training complete!")

    final_save_path = os.path.join(config.output_dir, "xray_report_generator_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

    tokenizer.save_pretrained(config.output_dir)
    print(f"Tokenizer saved to {config.output_dir}")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_sys_path = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root_for_sys_path not in sys.path:
        sys.path.insert(0, project_root_for_sys_path)
        print(f"Added {project_root_for_sys_path} to sys.path")

    train_model()