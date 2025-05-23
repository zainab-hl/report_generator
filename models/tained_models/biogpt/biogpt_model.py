import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptTokenizer
from ..BioMedClip import BiomedCLIPEncoder 
from ..Q_former.q_former import Qformer

class XrayReportGenerator(nn.Module):
    def __init__(self, biomedclip_model_name, biomedclip_weights_path, qformer_config):
        super().__init__()
        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name=biomedclip_model_name,
            weights_path=biomedclip_weights_path
        )

        assert qformer_config.encoder_width == self.biomedclip_encoder.feature_dim, \
            "Q-Former encoder_width must match BiomedCLIP feature_dim"

        self.qformer = Qformer(qformer_config)

        # --- ACTUAL BIOGPT LOADING ---
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.biogpt_decoder = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

        # Get BioGPT's hidden size for potential projection
        biogpt_hidden_size = self.biogpt_decoder.config.hidden_size

        # If Q-Former output hidden_size doesn't match BioGPT's input embedding size,
        # you need a projection layer. This is common.
        if qformer_config.hidden_size != biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                qformer_config.hidden_size, biogpt_hidden_size
            )
        else:
            self.qformer_output_to_biogpt_input_projection = None

        # Store the ID for the end-of-sentence token for generation stopping criteria
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # IMPORTANT: Ensure pad_token_id is set for generation, as some tokenizers might not have it by default.
        # This is crucial for batch inference and consistent generation behavior.
        if self.tokenizer.pad_token_id is None:
            # A common practice is to set pad_token_id to eos_token_id if it's not defined.
            # Or, you can add a specific pad token if your tokenizer allows.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
            import warnings
            warnings.warn("Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")


    # --- FIX: forward method must be a direct member of the class, not nested ---
    def forward(self, image_path, prompt_text: Optional[str] = None, max_new_tokens=50, num_beams=1, do_sample=False, top_k=None, top_p=None):
        # 1. Encode Image with BiomedCLIP
        image_features = self.biomedclip_encoder.encode_image(image_path)
        # BiomedCLIP typically outputs (batch_size, feature_dim).
        # We need it as (batch_size, sequence_length, feature_dim) for Q-Former.
        # Here, sequence_length is 1 for a global image feature.
        image_features_expanded = image_features.unsqueeze(1) # (batch_size, 1, 512)

        # 2. Process Image Features with Q-Former
        # `query_embeddings` will be (batch_size, num_query_tokens, qformer_hidden_size)
        query_embeddings = self.qformer(image_features_expanded)

        # 3. Project Q-Former output if necessary to match BioGPT's hidden size
        if self.qformer_output_to_biogpt_input_projection:
            query_embeddings = self.qformer_output_to_biogpt_input_projection(query_embeddings)
            # Now: (batch_size, num_query_tokens, biogpt_hidden_size)

        # 4. Prepare Textual Prompt Embeddings (if any)
        input_embeddings_list = [query_embeddings]
        input_attention_mask_list = [torch.ones(query_embeddings.shape[0], query_embeddings.shape[1], dtype=torch.long, device=query_embeddings.device)]

        if prompt_text:
            prompt_token_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
            prompt_token_ids = prompt_token_ids.to(query_embeddings.device)
            # Get embeddings for the prompt text from BioGPT's embedding layer
            text_embeddings = self.biogpt_decoder.get_input_embeddings()(prompt_token_ids)
            
            input_embeddings_list.append(text_embeddings)
            input_attention_mask_list.append(torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], dtype=torch.long, device=text_embeddings.device))
        
        # Concatenate query embeddings and text embeddings
        input_embeddings = torch.cat(input_embeddings_list, dim=1)
        input_attention_mask = torch.cat(input_attention_mask_list, dim=1)


        # 5. Generate Report with BioGPT
        # The key is to pass inputs_embeds to the generate method.
        generated_output = self.biogpt_decoder.generate(
            inputs_embeds=input_embeddings,
            attention_mask=input_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams, # For beam search
            do_sample=do_sample, # For sampling-based generation
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.eos_token_id, # Stop generation when EOS token is generated
            pad_token_id=self.tokenizer.pad_token_id, # Required for batch inference
        )

        # Decode the generated token IDs back to text
        # `generated_output` will be a tensor of token IDs: (batch_size, sequence_length)
        # Assuming batch_size is 1 for now for simplicity in decoding
        generated_report = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)

        return generated_report