import torch
import os
import sys
from typing import Optional
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from models.trained_models.Q_former.q_former import BertConfig # BertConfig is defined in q_former.py
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS # To get model names and weights paths

def generate_xray_report(
    image_path: str,
    prompt_text: Optional[str] = None,
    max_new_tokens: int = 100,
    num_beams: int = 4, # Use beam search for better quality generation
    do_sample: bool = False, # Set to True for more diverse outputs (e.g., for exploration)
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Generates an X-ray report using the multimodal model.

    Args:
        image_path (str): Path to the input X-ray image.
        prompt_text (Optional[str]): Initial prompt for the report generation (e.g., "The patient shows").
        max_new_tokens (int): Maximum number of tokens to generate.
        num_beams (int): Number of beams for beam search.
        do_sample (bool): Whether to use sampling.
        top_k (Optional[int]): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (Optional[float]): The cumulative probability for nucleus sampling.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        str: The generated X-ray report.
    """
    print(f"Using device: {device}")
    qformer_config = BertConfig(
        hidden_size=768,            # Q-Former's internal hidden size
        num_hidden_layers=6,        # Number of Q-Former Transformer layers
        num_attention_heads=12,     # Number of attention heads
        intermediate_size=3072,     # Feed-forward intermediate size
        encoder_width=512,          # Must match BiomedCLIPEncoder.feature_dim
        num_query_tokens=32,        # Number of learnable query tokens
        add_cross_attention=True,   # Essential for Q-Former's vision-language connection
        cross_attention_freq=1,     # Cross-attention in every layer (common for Q-Former)
    )

    print("Initializing XrayReportGenerator...")
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        qformer_config=qformer_config
    ).to(device)
    model.eval() 

    print("Model initialized. Generating report...")

    with torch.no_grad(): 
        generated_report = model(
            image_path=image_path,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p
        )

    print("\n--- Generated X-ray Report ---")
    print(generated_report)
    print("----------------------------")

    return generated_report

if __name__ == "__main__":
    example_image_path = "/content/bone-xray-hands.jpg" 
    
    if not os.path.exists(example_image_path):
        print(f"Error: Image file not found at {example_image_path}")
        print("Please replace 'path/to/your/xray_image.png' with a valid path to an X-ray image.")
        sys.exit(1)


    example_prompt = "describe the image that you will see in the prompt"

    generate_xray_report(
        image_path=example_image_path,
        prompt_text=example_prompt,
        max_new_tokens=80,
        num_beams=5,
        do_sample=False, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPrediction script finished.")