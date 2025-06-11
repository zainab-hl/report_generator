# At the absolute top of model.py, even before other imports
print("---DEBUG: This specific model.py is now running! (Path:", __file__, ")---")
import sys
print("---DEBUG: sys.path:", sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import warnings
import json
import os
from typing import Optional, Tuple, Dict, Any


from transformers import (
    PretrainedConfig,
    AutoConfig,
    PreTrainedModel,
    AutoTokenizer,
    AutoProcessor, 
    AutoModel, 
    BertConfig,
    AutoProcessor,
    AutoModel,
    BioGptForCausalLM
)
from transformers.models.bert.modeling_bert import BertEncoder
from huggingface_hub import hf_hub_download

import open_clip
from PIL import Image

class ModelOutput:
    """A simple class to mimic Hugging Face model outputs for Q-Former sub-components."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __getitem__(self, key):
        return self.kwargs[key]
    def __setitem__(self, key, value):
        self.kwargs[key] = value
    def __contains__(self, key):
        return key in self.kwargs
    def __repr__(self):
        return str(self.kwargs)

class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Model output class specifically for BERTEncoder with past_key_values and cross_attentions.
    Mimics transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions.
    """
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
        cross_attentions: Optional[Tuple[torch.Tensor]] = None
    ):
        super().__init__(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )

def apply_chunking_to_forward(
    forward_fn, chunk_size, seq_len_dim, *input_tensors
):
    """
    Helper function to apply a forward function to chunks of input tensors.
    Used for gradient checkpointing or memory efficiency.
    """
    if chunk_size > 0:
        return torch.cat(
            [
                forward_fn(*[t.narrow(seq_len_dim, i, chunk_size) for t in input_tensors])
                for i in range(0, input_tensors[0].shape[seq_len_dim], chunk_size)
            ],
            dim=seq_len_dim,
        )
    return forward_fn(*input_tensors)

# --- BertConfig Class for Q-Former ---
class BertConfig(PretrainedConfig):
    """
    Configuration for the Q-Former's internal BERT-like layers.
    This defines the architecture parameters for the Transformer blocks
    that make up the Q-Former.
    """
    model_type = "qformer_bert_config"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
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
        classifier_dropout=None,
        encoder_width=768,
        num_query_tokens=32,
        cross_attention_freq=1,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens
        self.cross_attention_freq = cross_attention_freq
        self.gradient_checkpointing = gradient_checkpointing


# --- Q-Former Sub-components ---

class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        if (self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        past_key_value = (key_layer, value_layer)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs_dropped = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)

        return outputs


class BertCrossAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_width, self.all_head_size)
        self.value = nn.Linear(config.encoder_width, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if encoder_attention_mask is not None:
            attention_scores = attention_scores + encoder_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set):
        if len(heads) == 0:
            return
        warnings.warn("`prune_heads` is not fully implemented for this example.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = self_outputs[0]
        outputs = self_outputs[1:]

        attention_output = self.output(attention_output, hidden_states)

        outputs = (attention_output,) + outputs
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig, layer_num: int):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0)
        self.seq_len_dim = 1
        self.attention = BertAttention(config) # Self-attention for query tokens
        self.layer_num = layer_num

        if self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0:
            self.crossattention = BertCrossAttention(config) # Separate Cross-Attention
            self.crossattention_output = BertSelfOutput(config) # Output layer for cross-attention
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        query_length: int = 0,
    ) -> Tuple[torch.Tensor, ...]:
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        present_key_value = self_attention_outputs[-1] # This is for self-attention past_key_value

        if self.has_cross_attention:
            assert (
                encoder_hidden_states is not None
            ), "encoder_hidden_states must be given for cross-attention layers when has_cross_attention is True"

            cross_attention_outputs = self.crossattention(
                attention_output, # Query from current hidden_states
                encoder_hidden_states, # Keys/Values from encoder_hidden_states
                encoder_attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
            cross_attention_output_from_layer = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:] # Add cross-attention probs if output_attentions

            # Apply the output layer for cross-attention
            layer_output = self.crossattention_output(cross_attention_output_from_layer, attention_output)

        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        
        # Apply the final feed-forward (intermediate and output) after potentially cross-attention
        layer_output = self.feed_forward_chunk(layer_output)


        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,) # Include present_key_value from self-attention

        return outputs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        device: torch.device,
        is_decoder: bool,
        has_query: bool = False,
    ) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=self.layer[0].attention.self.query.weight.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            extended_attention_mask = extended_attention_mask.to(device)
        else:
            raise ValueError(
                f"Wrong shape for input_shape ({input_shape}) or attention_mask ({attention_mask.shape})"
            )
        return extended_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        query_length: int = 0,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, hidden_states.shape[:-1], hidden_states.device, is_decoder=False
            )

        extended_encoder_attention_mask = None
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            extended_encoder_attention_mask = self.get_extended_attention_mask(
                encoder_attention_mask, encoder_hidden_states.shape[:-1], encoder_hidden_states.device, is_decoder=False
            )

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    extended_encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    extended_encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class BiomedCLIPEncoder(nn.Module):
    def __init__(self, model_name: str, weights_path: Optional[str] = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model, _, self.preprocess_fn = open_clip.create_model_and_transforms(model_name)

            self.model.to(self.device)

            if weights_path and os.path.exists(weights_path):
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device, weights_only=True)
                )
            elif weights_path:

            self.model.eval() # Set to evaluation mode

            with torch.no_grad():
                dummy_image = Image.new('RGB', (224, 224), color='red')
                dummy_input = self.preprocess_fn(dummy_image).unsqueeze(0).to(self.device)
                dummy_features = self.model.encode_image(dummy_input)
                self.feature_dim = dummy_features.shape[-1]

        except Exception as e:
            raise ImportError(f"Failed to load BiomedCLIP model using open_clip. Ensure model_name '{model_name}' is correct and accessible. Error: {e}")

    def encode_image(self, image_path: str) -> torch.Tensor:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess_fn(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use self.model.encode_image directly as in your working snippet
            features = self.model.encode_image(processed_image)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features


# --- Qformer Class (Retained from previous working version) ---
class Qformer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        self.bert_encoder = BertEncoder(config)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        query_attention_mask = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=device)

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:-1], dtype=torch.long, device=device
            )

        encoder_outputs = self.bert_encoder(
            hidden_states=query_tokens,
            attention_mask=query_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        output_query_embeddings = encoder_outputs['last_hidden_state']
        return output_query_embeddings


# --- XrayReportGeneratorConfig Class (Retained from previous working version) ---
class XrayReportGeneratorConfig(PretrainedConfig):
    model_type = "xray_report_generator"

    def __init__(
        self,
        biomedclip_model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        biogpt_base_model: str = "microsoft/biogpt",
        qformer_config: Optional[Dict[str, Any]] = None,
        max_seq_length: int = 256,
        biomedclip_finetuned_weights: str = "biomedclip_finetuned.pth",
        biogpt_finetuned_weights: str = "biogpt_finetuned.pth",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.biomedclip_model_name = biomedclip_model_name
        self.biogpt_base_model = biogpt_base_model
        self.max_seq_length = max_seq_length
        self.biomedclip_finetuned_weights = biomedclip_finetuned_weights
        self.biogpt_finetuned_weights = biogpt_finetuned_weights

        self.qformer_config = qformer_config if qformer_config is not None else {
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "add_cross_attention": True,
            "cross_attention_freq": 1,
            "encoder_width": 512,
            "num_query_tokens": 32,
            "gradient_checkpointing": False,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "position_embedding_type": "absolute"
        }

# Register the config for AutoConfig to find it
AutoConfig.register("xray_report_generator", XrayReportGeneratorConfig)


# --- XrayReportGenerator Class (Retained from previous working version) ---
class XrayReportGenerator(PreTrainedModel):
    config_class = XrayReportGeneratorConfig
    base_model_prefix = "xray_report_generator"

    def __init__(self, config: XrayReportGeneratorConfig):
        super().__init__(config)

        self.biomedclip_model_name = config.biomedclip_model_name
        self.biogpt_base_model = config.biogpt_base_model
        self.max_seq_length = config.max_seq_length
        self.repo_id = config._name_or_path

        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name=self.biomedclip_model_name,
            weights_path=None # Assume fine-tuned weights handled by XrayReportGenerator
        )

        # Load fine-tuned BiomedCLIP weights (if path exists in config and file is found)
        if config.biomedclip_finetuned_weights:
            try:
                biomedclip_local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=config.biomedclip_finetuned_weights,
                    cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                )
                self.biomedclip_encoder.model.load_state_dict(
                    torch.load(biomedclip_local_path, map_location=self.device)
                )
            except Exception as e:

        qformer_bert_config = BertConfig(**config.qformer_config)

        assert qformer_bert_config.encoder_width == self.biomedclip_encoder.feature_dim, \
            f"Q-Former encoder_width ({qformer_bert_config.encoder_width}) must match BiomedCLIP feature_dim ({self.biomedclip_encoder.feature_dim})"

        self.qformer = Qformer(qformer_bert_config)
        self.qformer.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.biogpt_base_model)

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            warnings.warn("Tokenizer pad_token is None, setting to eos_token.")

        self.biogpt_decoder = BioGptForCausalLM.from_pretrained(self.biogpt_base_model)
        self.biogpt_decoder.to(self.device)

        if config.biogpt_finetuned_weights:
            try:
                biogpt_local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=config.biogpt_finetuned_weights,
                    cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                )
                self.biogpt_decoder.load_state_dict(torch.load(biogpt_local_path, map_location=self.device))
            except Exception as e:

        biogpt_hidden_size = self.biogpt_decoder.config.hidden_size

        if qformer_bert_config.hidden_size != biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                qformer_bert_config.hidden_size, biogpt_hidden_size
            ).to(self.device)
        else:
            self.qformer_output_to_biogpt_input_projection = None

        self.eos_token_id = self.tokenizer.eos_token_id

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
        **kwargs
        ):
        is_training = image_features is not None and input_ids is not None and attention_mask is not None

        if image_path is not None and not is_training:
            image_features = self.biomedclip_encoder.encode_image(image_path)
        elif image_features is None and not is_training and not prompt_text:
            raise ValueError("For inference, either image_path, image_features, or prompt_text must be provided.")

        query_embeddings = None

        if image_features is not None:
            if image_features.ndim == 1:
                image_features = image_features.unsqueeze(0)

            image_features = image_features.to(self.device)

            image_features_for_qformer = image_features.unsqueeze(1)

            query_embeddings = self.qformer(encoder_hidden_states=image_features_for_qformer)

            if self.qformer_output_to_biogpt_input_projection:
                query_embeddings = self.qformer_output_to_biogpt_input_projection(query_embeddings)

            query_embeddings = query_embeddings.to(self.device)

        if is_training:
            if query_embeddings is None:
                raise ValueError("image_features (and thus query_embeddings) must be provided for training.")

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

        else:
            if query_embeddings is None:
                raise ValueError("For inference, image features (from image_path or image_features) must be provided.")

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
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=kwargs.get('temperature', 1.0),
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

            generated_report = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)

            return generated_report
