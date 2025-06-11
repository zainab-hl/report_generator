MODEL_NAMES = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "llm": "microsoft/biogpt",
}
MODEL_WEIGHTS = {
    "biomedclip": "/biomedclip_finetuned.pth", 
    "biogpt": "/biogpt_finetuned.pth"         
}