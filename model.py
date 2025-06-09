import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import warnings
from typing import Optional, Tuple, Dict, Any

# Standard Hugging Face and external library imports
from transformers import BioGptForCausalLM, BioGptTokenizer, AutoModel, AutoConfig
from transformers.utils import logging
from transformers.models.auto.modeling_auto import AutoModelForCausalLM # Keep if used, otherwise remove
from transformers.activations import ACT2FN

# External library for CLIP models (ensure it's pip installable in your environment)
import open_clip
from PIL import Image # For image loading in BiomedCLIPEncoder

logger = logging.get_logger(__name__)

# --- Helper Classes for Q-Former (from your previous model.py content) ---
# Copy and paste ModelOutput, BaseModelOutputWithPastAndCrossAttentions, apply_chunking_to_forward here.

# --- BertConfig Class (from your previous model.py content) ---
@AutoConfig.register("qformer_bert_config")
class BertConfig(AutoConfig):
    """
    Configuration for the Q-Former's internal BERT-like layers.
    ... (rest of BertConfig definition) ...
    """
    model_type = "qformer_bert_config" # Ensure this is present
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# --- Q-Former Sub-components (from your previous model.py content) ---
# Copy and paste BertSelfAttention, BertSelfOutput, BertAttention, BertIntermediate, BertOutput, BertLayer, BertEncoder here.

# --- Qformer Class (from your previous model.py content) ---
class Qformer(nn.Module):
    """
    The Querying Transformer (Q-Former) module.
    ... (rest of Qformer definition) ...
    """
    def __init__(self, config: BertConfig): # Ensure it expects BertConfig directly
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        self.bert_encoder = BertEncoder(config)
        self._init_weights()

    def _init_weights(self):
        # ... (your _init_weights method) ...
        pass

    def forward(self, image_features: torch.Tensor, image_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ... (your Qformer forward method) ...
        pass

# --- BiomedCLIPEncoder Class (copied from your original 'models.trained_models.BioMedClip.encoder' file) ---
# You need to define this class directly in model.py
class BiomedCLIPEncoder(nn.Module):
    def __init__(self, model_name, weights_path, img_size=224): # Added img_size as it's common for CLIP
        super().__init__()
        # This part assumes open_clip is installed and the weights_path points to a compatible file
        model, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=weights_path, img_size=img_size)
        self.model = model
        self.preprocess = preprocess
        # Ensure feature_dim matches your actual CLIP model's output dimension
        self.feature_dim = model.visual.output_dim # Common attribute for CLIP models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # Move model to device

    def encode_image(self, image_path: str) -> torch.Tensor:
        # Load image using PIL and preprocess
        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(processed_image)
            features = features / features.norm(p=2, dim=-1, keepdim=True) # Normalize features
        return features

# --- XrayReportGenerator Class (copied from your biogpt_model.py) ---
@AutoModel.register(model_type="xray_report_generator")
class XrayReportGenerator(nn.Module):
    def __init__(self, biomedclip_model_name, biomedclip_weights_path, qformer_config,
                 biogpt_weights_path: Optional[str] = None):
        super().__init__()
        # BiomedCLIPEncoder is now defined in this file
        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name=biomedclip_model_name,
            weights_path=biomedclip_weights_path
        )

        assert qformer_config.encoder_width == self.biomedclip_encoder.feature_dim, \
            "Q-Former encoder_width must match BiomedCLIP feature_dim"
        
        # Define device once, it's already done in BiomedCLIPEncoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Keep this for main model ops
        
        # Qformer is now defined in this file
        self.qformer = Qformer(qformer_config) # Pass the Qformer config directly

        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.biogpt_decoder = BioGptForCausalLM.from_pretrained("microsoft/biogpt") 
        self.biogpt_decoder.to(self.device) # Move decoder to device

        if biogpt_weights_path:
            print(f"Loading fine-tuned BioGPT weights from: {biogpt_weights_path}")
            state_dict = torch.load(biogpt_weights_path, map_location='cpu') 
            self.biogpt_decoder.load_state_dict(state_dict) 
            print("Fine-tuned BioGPT weights loaded successfully.")
        else:
            print("No fine-tuned BioGPT weights file provided, using default pre-trained BioGPT.")

        biogpt_hidden_size = self.biogpt_decoder.config.hidden_size

        if qformer_config.hidden_size != biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                qformer_config.hidden_size, biogpt_hidden_size
            )
        else:
            self.qformer_output_to_biogpt_input_projection = None

        self.eos_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            import warnings
            warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")

    # Remove the redundant class-level device definition here: device = torch.device(...)

    def forward(self,
        image_path: Optional[str] = None, 
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        image_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        ):
        # ... (your forward method logic from biogpt_model.py) ...
        # Ensure image_features and other tensors are moved to self.device if they're not already.
        
        is_training = image_features is not None and input_ids is not None and attention_mask is not None 
        if image_path is not None and not is_training:
            image_features = self.biomedclip_encoder.encode_image(image_path)
            # image_features = image_features.to(self.device) # Already done in encode_image if BiomedCLIPEncoder moves it
        elif image_features is None and not is_training: # Added 'and not is_training' to avoid raising error if image_features are not needed for training
            raise ValueError("Either image_path or image_features must be provided for inference.")
        
        if image_features is not None: # Only process if image_features are available
            if image_features.ndim == 1:
                image_features = image_features.unsqueeze(0)
            
            image_features_expanded = image_features.unsqueeze(1) 
            query_embeddings = self.qformer(image_features_expanded)

            if self.qformer_output_to_biogpt_input_projection:
                query_embeddings = self.qformer_output_to_biogpt_input_projection(query_embeddings)
            
            # Ensure query_embeddings are on the correct device for concatenation/LM input
            query_embeddings = query_embeddings.to(self.device)


        # training mode 
        if is_training:
            report_embeddings = self.biogpt_decoder.get_input_embeddings()(input_ids)
            decoder_input_embeddings = torch.cat([query_embeddings, report_embeddings], dim=1)
            query_attention_mask = torch.ones(
                query_embeddings.shape[0], 
                query_embeddings.shape[1], 
                dtype=torch.long, 
                device=self.device
            )
            decoder_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)
            labels = input_ids.clone()
            ignored_labels_for_query = torch.full(
                (query_embeddings.shape[0], query_embeddings.shape[1]),
                -100,
                dtype = torch.long,
                device = self.device
            )
            decoder_labels = torch.cat([ignored_labels_for_query, labels], dim=1)
            biogpt_decoder_kwargs = {
                "inputs_embeds": decoder_input_embeddings,
                "attention_mask": decoder_attention_mask,
                "labels": decoder_labels,
                "return_dict": True
            }

            outputs = self.biogpt_decoder(**biogpt_decoder_kwargs) 
            return outputs.loss
        else: # Inference mode
            input_embeddings_list = [query_embeddings]
            input_attention_mask_list = [torch.ones(query_embeddings.shape[0], query_embeddings.shape[1], dtype=torch.long, device=self.device)]

            if prompt_text:
                prompt_token_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
                prompt_token_ids = prompt_token_ids.to(self.device)
                text_embeddings = self.biogpt_decoder.get_input_embeddings()(prompt_token_ids)
                
                input_embeddings_list.append(text_embeddings)
                input_attention_mask_list.append(torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], dtype=torch.long, device=self.device))
            
            input_embeddings = torch.cat(input_embeddings_list, dim=1)
            input_attention_mask = torch.cat(input_attention_mask_list, dim=1)

            generated_output = self.biogpt_decoder.generate(
                inputs_embeds=input_embeddings,
                attention_mask=input_attention_mask,
                max_new_tokens=max_new_tokens, 
                num_beams=num_beams, # Use parameter passed to forward, not fixed 1
                do_sample=do_sample, # Use parameter passed to forward
                temperature=kwargs.get('temperature', 1.0), # Pass through kwargs or default
                top_k=top_k, # Use parameter passed to forward
                top_p=top_p, # Use parameter passed to forward
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Ensure generated_output is correctly handled for decode (it's usually 2D [batch, sequence])
            generated_report = self.tokenizer.decode(generated_output[0], skip_special_tokens=True) # Always decode the first sequence for single-batch

            return generated_report