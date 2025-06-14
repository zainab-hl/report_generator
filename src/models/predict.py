import torch
import os
import sys
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from models.trained_models.biogpt.biogpt_model import XrayReportGenerator
from models.trained_models.Q_former.q_former import BertConfig
from configs.constants import MODEL_NAMES, MODEL_WEIGHTS

def generate_xray_report(
    image_path: str,
    prompt_text: Optional[str] = None,
    max_new_tokens: int = 100,
    num_beams: int = 4,
    do_sample: bool = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    print(f"Using device: {device}")
    
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

    print("Initializing XrayReportGenerator (architecture only)...")
    
    model = XrayReportGenerator(
        biomedclip_model_name=MODEL_NAMES['biomedclip'],
        biomedclip_weights_path=MODEL_WEIGHTS['biomedclip'],
        qformer_config=qformer_config,
        biogpt_weights_path=None 
    ).to(device)

    FINE_TUNED_MODEL_PATH = "/content/drive/MyDrive/finetuned_report_generator/xray_report_generator_final.pth" 

    print(f"Loading fine-tuned model weights from: {FINE_TUNED_MODEL_PATH}")
    full_model_state_dict = torch.load(FINE_TUNED_MODEL_PATH, map_location=device)
    
    model.load_state_dict(full_model_state_dict)
    
    model.eval() 

    print("Model initialized and fine-tuned weights loaded. Generating report...")

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
        print("Please replace '/content/bone-xray-hands.jpg' with a valid path to an X-ray image.")
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