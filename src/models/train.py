# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BioGptTokenizer
# from transformers.optimization import get_scheduler
# from torch.optim import AdamW
# import os
# import sys
# from tqdm.auto import tqdm
# import json 
# import warnings 


# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, project_root)

# from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
# from models.trained_models.Q_former.q_former import BertConfig
# from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

# # --- Configuration for Training ---
# class TrainingConfig:
#     def __init__(self):
#         self.dataset_dir = "/content/drive/MyDrive/processed_dataset" 
#         self.output_dir = "/content/drive/MyDrive/finetuned_report_generator"
#         self.max_seq_length = 256
#         self.train_batch_size = 4
#         self.eval_batch_size = 8
#         self.learning_rate = 5e-5
#         self.num_epochs = 3
#         self.warmup_steps = 0.1 
#         self.gradient_accumulation_steps = 1

# class ReportGenerationDataset(Dataset):
#     def __init__(self, data_dir: str, tokenizer: BioGptTokenizer, max_seq_length: int = 256):
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#         self.data = [] # This will store all loaded data from all JSON files

#         if self.tokenizer.bos_token is None:
#             self.tokenizer.add_special_tokens({'bos_token': '<s>'})
#         if self.tokenizer.eos_token is None:
#             self.tokenizer.add_special_tokens({'eos_token': '</s>'})
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

#         print(f"Loading data from directory: {data_dir}")
#         for filename in os.listdir(data_dir):
#             if filename.endswith('.json'): 
#                 filepath = os.path.join(data_dir, filename)
#                 try:
#                     with open(filepath, 'r') as f:
#                         entry = json.load(f)
#                         self.data.append(entry)
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not decode JSON from {filepath}. Skipping.")
#                 except Exception as e:
#                     print(f"Warning: Error loading {filepath}: {e}. Skipping.")
        
#         if not self.data:
#             raise ValueError(f"No data loaded from {data_dir}. Check directory path and file formats.")
#         print(f"Successfully loaded {len(self.data)} samples.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         # Image embeddings (convert list to tensor)
#         image_embedding = torch.tensor(item["embedding"], dtype=torch.float32)

#         # Report text tokenization
#         tokenized_report = self.tokenizer(
#             self.tokenizer.bos_token + item["report"] + self.tokenizer.eos_token,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_seq_length,
#             return_tensors="pt"
#         )
        
#         input_ids = tokenized_report.input_ids.squeeze(0)
#         attention_mask = tokenized_report.attention_mask.squeeze(0)
        
#         return {
#             "image_embedding": image_embedding,
#             "input_ids": input_ids,
#             "attention_mask": attention_mask
#         }

# # --- Main Training Function ---
# def train_model():
#     config = TrainingConfig()
    
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Create output directory if it doesn't exist
#     os.makedirs(config.output_dir, exist_ok=True)

#     tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
#         warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")

#     train_dataset = ReportGenerationDataset(config.dataset_dir, tokenizer, config.max_seq_length) # <--- Changed
#     train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

#     # Initialize Q-Former Config
#     qformer_config = BertConfig(
#         hidden_size=768,
#         num_hidden_layers=6,
#         num_attention_heads=12,
#         intermediate_size=3072,
#         encoder_width=512, 
#         num_query_tokens=32,
#         add_cross_attention=True,
#         cross_attention_freq=1,
#     )

#     # Initialize Model
#     print("Initializing XrayReportGenerator for training...")
#     model = XrayReportGenerator(
#         biomedclip_model_name=MODEL_NAMES['biomedclip'],
#         biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
#         biogpt_weights_path=MODEL_WEIGHTS['biogpt'],
#         qformer_config=qformer_config
#     ).to(device)

#     # Optimizer
#     optimizer = AdamW(model.parameters(), lr=config.learning_rate)

#     # Learning Rate Scheduler
#     num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
#     num_warmup_steps = int(num_training_steps * config.warmup_steps)
#     lr_scheduler = get_scheduler(
#         name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
#     )

#     # --- Training Loop ---
#     print("Starting training...")
#     model.train()
#     progress_bar = tqdm(range(num_training_steps))
#     completed_steps = 0

#     for epoch in range(config.num_epochs):
#         for batch in train_dataloader:
#             # Move batch to device
#             image_embedding = batch["image_embedding"].to(device)
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)

#             loss = model(
#                 image_path=None,  
#                 image_features=image_embedding,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
            
#             loss = loss / config.gradient_accumulation_steps
#             loss.backward()

#             if (completed_steps + 1) % config.gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
            
#             progress_bar.update(1)
#             completed_steps += 1

#             if completed_steps % 100 == 0:
#                 print(f"Step {completed_steps}, Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")

#         print(f"Epoch {epoch+1} finished. Saving model...")
#         save_path = os.path.join(config.output_dir, f"report_generator_epoch__2_{epoch+1}.pth")
#         torch.save(model.state_dict(), save_path)
#         print(f"Model saved to {save_path}")

#     print("Training complete!")

#     final_save_path = os.path.join(config.output_dir, "report_generator_final_2_.pth")
#     torch.save(model.state_dict(), final_save_path)
#     print(f"Final model saved to {final_save_path}")

#     tokenizer.save_pretrained(config.output_dir)
#     print(f"Tokenizer saved to {config.output_dir}")


# if __name__ == "__main__":

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root_for_sys_path = os.path.abspath(os.path.join(current_dir, '..', '..')) 
#     if project_root_for_sys_path not in sys.path:
#         sys.path.insert(0, project_root_for_sys_path)
#         print(f"Added {project_root_for_sys_path} to sys.path")

#     train_model()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer, AutoConfig # Import AutoConfig for better config handling
from transformers.optimization import get_scheduler
from torch.optim import AdamW
import os
import sys
from tqdm.auto import tqdm
import json 
import warnings 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to sys.path")

from model.py import XrayReportGenerator, XrayReportGeneratorConfig, BertConfig
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

# --- Configuration for Training ---
class TrainingConfig:
    def __init__(self):
        self.dataset_dir = "/content/drive/MyDrive/processed_dataset" 
        self.output_dir = "/content/drive/MyDrive/finetuned_report_generator"
        self.max_seq_length = 256
        self.train_batch_size = 4
        self.eval_batch_size = 8 
        self.learning_rate = 5e-5
        self.num_epochs = 3
        self.warmup_steps = 0.1 
        self.gradient_accumulation_steps = 1
        # Add BiomedCLIP encoder_width for QFormer config consistency
        self.biomedclip_encoder_width = 512 # This should match BiomedCLIP's feature dimension

class ReportGenerationDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: BioGptTokenizer, max_seq_length: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = [] 

        # Add special tokens if they don't exist (important for BioGPT)
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
            logger.warning("Added '<s>' as BOS token to tokenizer.")
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            logger.warning("Added '</s>' as EOS token to tokenizer.")
        if self.tokenizer.pad_token is None:
            # It's common to use EOS token as PAD token for GPT-like models if PAD isn't defined
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            logger.warning(f"Tokenizer pad_token not set, using eos_token ('{self.tokenizer.eos_token}') as pad_token.")
        
        # Load data from JSON files
        logger.info(f"Loading data from directory: {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'): 
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        entry = json.load(f)
                    self.data.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {filepath}. Skipping.")
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}. Skipping.")
        
        if not self.data:
            raise ValueError(f"No data loaded from {data_dir}. Check directory path and file formats.")
        logger.info(f"Successfully loaded {len(self.data)} samples.")

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
def train_model():
    config = TrainingConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {config.output_dir}")

    # Initialize tokenizer
    tokenizer = BioGptTokenizer.from_pretrained(MODEL_NAMES['biogpt']) # Use BioGPT from MODEL_NAMES
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")
    logger.info("Tokenizer initialized.")

    # Prepare Dataset and DataLoader
    train_dataset = ReportGenerationDataset(config.dataset_dir, tokenizer, config.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    logger.info(f"DataLoader initialized with {len(train_dataloader)} batches.")

    # Initialize Q-Former Config (using BertConfig from your Q_former module)
    qformer_bert_config = BertConfig(
        # Parameters explicitly overridden from your BertConfig's defaults:
        num_hidden_layers=6,  # Your custom BertConfig defaults to 12, you're setting it to 6
        encoder_width=config.biomedclip_encoder_width, # Your custom BertConfig defaults to 768, you're setting it to 512 (from TrainingConfig)

        # Parameters that match your custom BertConfig's defaults, you can include them for clarity
        # or omit them if you're happy with the default:
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None, # Explicitly setting to None is fine
        num_query_tokens=32,
        cross_attention_freq=1,
        gradient_checkpointing=False, # Assuming you want this as False by default
    )
    # The XrayReportGeneratorConfig expects a dictionary for qformer_config
    qformer_config_dict = qformer_bert_config.to_dict()
    logger.info("Q-Former BertConfig prepared.")

    # Initialize XrayReportGeneratorConfig (for the overall model)
    xray_report_generator_config = XrayReportGeneratorConfig(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biogpt_base_model=MODEL_NAMES['biogpt'],
        qformer_config=qformer_config_dict, # Pass the dictionary
        # These paths are included in the config but will be explicitly loaded below for training
        biomedclip_finetuned_weights=MODEL_WEIGHTS['biomedclip'],
        biogpt_finetuned_weights=MODEL_WEIGHTS['biogpt'],
        max_seq_length=config.max_seq_length # Pass max_seq_length to the model config too
    )
    logger.info("XrayReportGeneratorConfig for the overall model created.")

    # Initialize Model with the new config object
    logger.info("Initializing XrayReportGenerator model...")
    model = XrayReportGenerator(xray_report_generator_config).to(device)
    logger.info("XrayReportGenerator model instantiated.")

    # --- Explicitly Load Fine-tuned Weights for Training ---
    # This step is critical because XrayReportGenerator's __init__ was modified
    # to NOT load external .pth files when initialized (for HF Hub deployment).
    # You need to load them here to start your training from the fine-tuned state.

    # Load BiomedCLIP fine-tuned weights
    if MODEL_WEIGHTS["biomedclip"] and os.path.exists(MODEL_WEIGHTS["biomedclip"]):
        try:
            logger.info(f"Attempting to load fine-tuned BiomedCLIP weights from {MODEL_WEIGHTS['biomedclip']}")
            # Use weights_only=True for security if source is untrusted, though usually safe for own files
            biomedclip_state_dict = torch.load(MODEL_WEIGHTS["biomedclip"], map_location=device, weights_only=True)
            # Load into the wrapped model's clip_model
            model.biomedclip_encoder.model_wrapper.clip_model.load_state_dict(biomedclip_state_dict)
            logger.info(f"Successfully loaded fine-tuned BiomedCLIP weights from {MODEL_WEIGHTS['biomedclip']}")
        except Exception as e:
            logger.error(f"Error loading BiomedCLIP weights for training from {MODEL_WEIGHTS['biomedclip']}: {e}")
            logger.warning("Proceeding with BiomedCLIP model as initialized (from Hugging Face Hub, not your local fine-tuned weights).")
    else:
        logger.warning(f"BiomedCLIP fine-tuned weights path '{MODEL_WEIGHTS['biomedclip']}' not found. Using model as initialized (from Hugging Face Hub).")
    
    # Freeze BiomedCLIP encoder parameters (common practice for this architecture)
    model.biomedclip_encoder.eval() # Set to eval mode, ensures dropout/batchnorm act as inference
    for param in model.biomedclip_encoder.parameters():
        param.requires_grad = False # Freeze parameters
    logger.info("BiomedCLIPEncoder set to eval mode and frozen (requires_grad=False).")

    # Load BioGPT fine-tuned weights
    if MODEL_WEIGHTS["biogpt"] and os.path.exists(MODEL_WEIGHTS["biogpt"]):
        try:
            logger.info(f"Attempting to load fine-tuned BioGPT weights from {MODEL_WEIGHTS['biogpt']}")
            biogpt_state_dict = torch.load(MODEL_WEIGHTS["biogpt"], map_location=device, weights_only=True)
            model.biogpt_decoder.load_state_dict(biogpt_state_dict)
            logger.info(f"Successfully loaded fine-tuned BioGPT weights from {MODEL_WEIGHTS['biogpt']}")
        except Exception as e:
            logger.error(f"Error loading BioGPT weights for training from {MODEL_WEIGHTS['biogpt']}: {e}")
            logger.warning("Proceeding with BioGPT model as initialized (from Hugging Face Hub, not your local fine-tuned weights).")
    else:
        logger.warning(f"BioGPT fine-tuned weights path '{MODEL_WEIGHTS['biogpt']}' not found. Using model as initialized (from Hugging Face Hub).")

    
    trainable_params = list(model.qformer.parameters())
    if model.qformer_output_to_biogpt_input_projection:
        trainable_params.extend(list(model.qformer_output_to_biogpt_input_projection.parameters()))
    
    
    trainable_params.extend(list(model.biogpt_decoder.parameters()))
    
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")

    # Optimizer
    optimizer = AdamW(trainable_params, lr=config.learning_rate) # Pass the specific trainable_params
    logger.info("Optimizer initialized with trainable parameters.")

    # Learning Rate Scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_steps)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    logger.info(f"Learning rate scheduler initialized. Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    # --- Training Loop ---
    logger.info("Starting training...")
    model.train() # Set model to training mode (Qformer, projection, and unfrozen BioGPT)
    progress_bar = tqdm(range(num_training_steps), desc="Training progress")
    completed_steps = 0

    for epoch in range(config.num_epochs):
        total_loss_epoch = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch tensors to device
            image_embedding = batch["image_embedding"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass: image_features is directly the pre-computed image_embedding
            loss = model(
                image_features=image_embedding, # Pass the pre-computed embedding
                input_ids=input_ids,
                attention_mask=attention_mask,
                # image_path=None because we're using pre-computed embeddings
            )
            
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress_bar.update(1) # Update progress bar only on optimizer step

            total_loss_epoch += loss.item() * config.gradient_accumulation_steps # Accumulate actual loss before division

            if (completed_steps % 100 == 0) and ((batch_idx + 1) % config.gradient_accumulation_steps == 0):
                logger.info(f"Step {completed_steps}, Batch {batch_idx+1}/{len(train_dataloader)}, Current Loss: {loss.item() * config.gradient_accumulation_steps:.4f}")

        avg_loss_epoch = total_loss_epoch / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss_epoch:.4f}")

        # Save model checkpoint per epoch (optional, but good practice)
        epoch_save_path_dict = os.path.join(config.output_dir, f"pytorch_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_path_dict)
        logger.info(f"Model checkpoint saved to {epoch_save_path_dict}")

    logger.info("Training complete!")

   
    final_model_bin_path = os.path.join(config.output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), final_model_bin_path)
    logger.info(f"Final model state_dict saved to {final_model_bin_path} for Hugging Face compatibility.")

    # Save the model config (which now includes QFormer details)
    # This saves the XrayReportGeneratorConfig which AutoModel can load
    model.config.save_pretrained(config.output_dir)
    logger.info(f"Model configuration saved to {config.output_dir}")

    # Save the tokenizer
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Tokenizer saved to {config.output_dir}")

if __name__ == "__main__":
    # Ensure sys.path is updated at the very beginning of execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_sys_path = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    if project_root_for_sys_path not in sys.path:
        sys.path.insert(0, project_root_for_sys_path)
        print(f"Added {project_root_for_sys_path} to sys.path") # Using print here as logging might not be fully set up yet

    train_model()