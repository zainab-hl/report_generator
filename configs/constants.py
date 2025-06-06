MODEL_NAMES = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "llm": "microsoft/biogpt",
}
# these weights are fine tuned , biomedclip should be freezed in a future end-to-end-training !!
MODEL_WEIGHTS = {
    "biomedclip": '/content/drive/MyDrive/biomedclip_finetunedtry3.pth',
    "biogpt" : '/content/drive/MyDrive/biogpt_finetuned1.pth'
}