import torch
import os
from collections import OrderedDict
import sys

# Assume your project root is correctly set up for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import necessary model classes for loading a dummy model structure
# You need the same model architecture to load state_dicts
from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from models.trained_models.Q_former.q_former import BertConfig
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

def average_models(output_base_dir: str, num_workers: int, final_output_path: str):
    print(f"Starting model averaging for {num_workers} workers...")
    
    # Initialize a dummy model (e.g., worker 0's model) to get the structure
    # This model will hold the averaged weights later.
    # We need to initialize it with the same configuration as during training.
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
    # Ensure this matches the training configuration
    dummy_model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'], # These weights might be needed for initialization
        biogpt_weights_path=MODEL_WEIGHTS['biogpt'], # These weights might be needed for initialization
        qformer_config=qformer_config
    )
    
    # Initialize an OrderedDict to accumulate weights
    averaged_state_dict = OrderedDict()
    
    for worker_id in range(num_workers):
        worker_model_path = os.path.join(output_base_dir, f"worker_{worker_id}", f"report_generator_final_worker_{worker_id}.pth")
        
        if not os.path.exists(worker_model_path):
            print(f"Warning: Model for worker {worker_id} not found at {worker_model_path}. Skipping this worker.")
            continue
            
        print(f"Loading model from: {worker_model_path}")
        state_dict = torch.load(worker_model_path, map_location='cpu') # Load to CPU to avoid GPU memory issues
        
        for key, param in state_dict.items():
            if key not in averaged_state_dict:
                averaged_state_dict[key] = param.clone() # Clone to prevent in-place modification issues
            else:
                # Add the parameters. If they are on different devices, this might cause issues
                # Ensure they are on CPU or move them before adding
                if param.device != averaged_state_dict[key].device:
                    averaged_state_dict[key] += param.to(averaged_state_dict[key].device)
                else:
                    averaged_state_dict[key] += param

    # Divide by the number of workers to get the average
    valid_workers_count = num_workers # Assuming all workers produced models, adjust if warnings occurred
    if valid_workers_count == 0:
        raise ValueError("No models were successfully loaded for averaging.")

    for key in averaged_state_dict:
        averaged_state_dict[key] /= valid_workers_count

    # Load the averaged state dictionary into the dummy model
    dummy_model.load_state_dict(averaged_state_dict)
    
    # Save the final averaged model
    torch.save(dummy_model.state_dict(), final_output_path)
    print(f"Averaged model saved to: {final_output_path}")

if __name__ == "__main__":
    # --- Configuration for averaging ---
    # This should match your TrainingConfig's output_dir from training
    BASE_OUTPUT_DIR = "/content/drive/MyDrive/finetuned_report_generator" 
    NUM_WORKERS = 2 # IMPORTANT: Set this to the total number of Colab instances you used
    FINAL_MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, "report_generator_final_averaged.pth")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_sys_path = os.path.abspath(os.path.join(current_dir, '..', '..')) 
    if project_root_for_sys_path not in sys.path:
        sys.path.insert(0, project_root_for_sys_path)
        print(f"Added {project_root_for_sys_path} to sys.path")

    average_models(BASE_OUTPUT_DIR, NUM_WORKERS, FINAL_MODEL_SAVE_PATH)