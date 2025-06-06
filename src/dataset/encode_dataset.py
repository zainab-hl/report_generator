import os
import json
import torch
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# --- Configuration ---
PROJECT_PATH = '/content/report_generator'
DATASET_PATH = '/content/drive/MyDrive/dataSET' 
OUTPUT_DATASET_PATH = '/content/drive/MyDrive/processed_dataset' 

import sys
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
    print(f"Added '{PROJECT_PATH}' to sys.path for module import.")

# Change current working directory to project_path for relative imports within the project
os.chdir(PROJECT_PATH)
print(f"Current working directory set to: {os.getcwd()}")

# Import the encoder class
try:
    from models.tained_models.BioMedClip.encoder import BiomedCLIPEncoder
    from configs.constants import MODEL_NAMES, MODEL_WEIGHTS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'encoder.py' and 'configs/constants.py' are correctly located within 'PROJECT_PATH' and the necessary dependencies are installed.")
    sys.exit(1)

os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)
print(f"Output directory '{OUTPUT_DATASET_PATH}' ensured.")

# --- Initialize the Encoder ---
print("Initializing BiomedCLIPEncoder...")
try:
    encoder = BiomedCLIPEncoder(model_name=MODEL_NAMES['biomedclip'],
                                weights_path=MODEL_WEIGHTS['biomedclip'])
    print("BiomedCLIPEncoder initialized successfully.")
except Exception as e:
    print(f"Failed to initialize BiomedCLIPEncoder: {e}")
    print("Please check your model and weights paths in configs/constants.py and ensure the model files are accessible.")
    sys.exit(1)


# --- Process the Dataset ---
print(f"Starting dataset processing from: {DATASET_PATH}")
processed_count = 0
for entry_name in os.listdir(DATASET_PATH):
    entry_path = os.path.join(DATASET_PATH, entry_name)

    if os.path.isdir(entry_path):
        image_file = None
        caption_file = None

        # Find image and caption files in the current directory
        for item in os.listdir(entry_path):
            if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
                image_file = os.path.join(entry_path, item)
            elif item == 'caption.json':
                caption_file = os.path.join(entry_path, item)

        if image_file and caption_file:
            print(f"Processing: {entry_path}")
            try:
                # 1. Encode the image
                image_features = encoder.encode_image(image_file)
                # Convert tensor to list for JSON serialization
                image_embedding = image_features.squeeze(0).tolist()

                # 2. Read the caption
                with open(caption_file, 'r') as f:
                    caption_data = json.load(f)
                    caption_text = caption_data.get('caption', '')

                if not caption_text:
                    print(f"Warning: 'caption' field not found in {caption_file}. Skipping this entry.")
                    continue

                # 3. Create the output dictionary
                output_data = {
                    "embedding": image_embedding,
                    "report": caption_text
                }

                # 4. Save the output to a new JSON file
                output_filename = os.path.basename(entry_path) + '.json' # Use directory name as filename
                output_filepath = os.path.join(OUTPUT_DATASET_PATH, output_filename)

                with open(output_filepath, 'w') as f:
                    json.dump(output_data, f, indent=4)
                
                processed_count += 1
                print(f"Saved: {output_filepath}")

            except Exception as e:
                print(f"Error processing {entry_path}: {e}")
        else:
            print(f"Skipping {entry_path}: Image or caption.json not found.")

print(f"\nFinished processing. Total entries processed: {processed_count}")
print(f"Processed dataset saved to: {OUTPUT_DATASET_PATH}")