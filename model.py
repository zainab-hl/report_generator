import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import warnings
import os # For image path checks in BiomedCLIPEncoder
from typing import Optional, Tuple, Dict, Any

# Standard Hugging Face and external library imports
from transformers import BioGptForCausalLM, BioGptTokenizer, AutoModel, AutoConfig
from transformers.utils import logging
# AutoModelForCausalLM is not strictly needed if only AutoModel is used for the top-level class
# and BioGptForCausalLM is instantiated directly
# from transformers.models.auto.modeling_auto import AutoModelForCausalLM 
from transformers.activations import ACT2FN

# External library for CLIP models (needs to be installed by user: pip install open_clip_torch)
import open_clip 
from PIL import Image # For image loading in BiomedCLIPEncoder

logger = logging.get_logger(__name__)

# --- Helper Classes for Q-Former ---

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

# --- BertConfig Class ---
@AutoConfig.register("qformer_bert_config")
class BertConfig(AutoConfig):
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
        num_hidden_layers=12, # Default to 12 as per original BERT
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
        # Q-Former specific parameters
        encoder_width=768, # Dimensionality of the input image features (from BiomedCLIP in your case)
        num_query_tokens=32, # Number of learnable query embeddings
        cross_attention_freq=1, # How often cross-attention layers appear (e.g., 1 means every layer)
        gradient_checkpointing: bool = False, # Enable gradient checkpointing for memory saving
        **kwargs # Accept additional keyword arguments for flexibility with AutoConfig
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
        
        # Q-Former specific
        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens
        self.cross_attention_freq = cross_attention_freq
        self.gradient_checkpointing = gradient_checkpointing


# --- Q-Former Sub-components (copied directly from your provided code) ---

class BertSelfAttention(nn.Module):
    """
    Self-attention mechanism for BERT-like models.
    Handles both self-attention (queries attend to themselves) and cross-attention
    (queries attend to encoder hidden states, i.e., image features).
    """
    def __init__(self, config: BertConfig, is_cross_attention: bool):
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
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
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
        """
        Transposes a tensor for multi-head attention scores.
        Input: (batch_size, sequence_length, all_head_size)
        Output: (batch_size, num_attention_heads, sequence_length, attention_head_size)
        """
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
        """
        Forward pass for self-attention or cross-attention.
        """
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask # Use encoder's mask for cross-attention
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

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


class BertSelfOutput(nn.Module):
    """
    Output layer for self-attention. Applies a dense layer, dropout, and layer normalization.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the self-attention output.
        Args:
            hidden_states: Output from the self-attention mechanism.
            input_tensor: The original input to the self-attention block (for residual connection).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Combines BertSelfAttention and BertSelfOutput to form a complete attention block.
    """
    def __init__(self, config: BertConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set() 

    def prune_heads(self, heads: set):
        """Placeholder for pruning attention heads."""
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
        """
        Forward pass for the attention block.
        """
        self_outputs = self.self(
            hidden_states, 
            attention_mask, 
            head_mask,
            encoder_hidden_states, 
            encoder_attention_mask, 
            past_key_value, 
            output_attentions,
        )
        attention_output = self_outputs[0] 
        outputs = self_outputs[1:] 

        attention_output = self.output(attention_output, hidden_states) # Apply self output with residual

        # Re-add past_key_value (which is always the last output from BertSelfAttention)
        outputs = (attention_output,) + outputs
        return outputs


class BertIntermediate(nn.Module):
    """
    Intermediate (feed-forward) layer in a BERT Transformer block.
    Applies a dense layer followed by an activation function.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the intermediate layer.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Output layer for the feed-forward network in a BERT Transformer block.
    Applies a dense layer, dropout, and layer normalization.
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward output.
        Args:
            hidden_states: Output from the intermediate layer.
            input_tensor: The input to the feed-forward block (for residual connection).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    A single Transformer layer, consisting of self-attention, cross-attention (optional),
    and a feed-forward network. This is the core building block of the Q-Former's encoder.
    """
    def __init__(self, config: BertConfig, layer_num: int):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = getattr(config, "chunk_size_feed_forward", 0)
        self.seq_len_dim = 1 
        self.attention = BertAttention(config) 
        self.layer_num = layer_num

        if self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0:
            self.crossattention = BertAttention(
                config, is_cross_attention=True 
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config) 
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

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
        """
        Forward pass for a single BERT layer within the Q-Former.
        """
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

        present_key_value = self_attention_outputs[-1] 

        if self.has_cross_attention:
            assert (
                encoder_hidden_states is not None
            ), "encoder_hidden_states must be given for cross-attention layers when has_cross_attention is True"

            cross_attention_outputs = self.crossattention(
                attention_output, 
                attention_mask, # Using attention_output's mask for self-attention
                head_mask,
                encoder_hidden_states, 
                encoder_attention_mask, 
                output_attentions=output_attentions,
            )
            cross_attention_output = cross_attention_outputs[0] 
            outputs = (outputs + cross_attention_outputs[1:]) 

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                cross_attention_output,
            )
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )

        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Standard feed-forward chunk (used if no cross-attention or for non-query tokens)."""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Feed-forward chunk specifically for query tokens after cross-attention."""
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """
    The BERT-like Encoder that forms the backbone of the Q-Former.
    It stacks multiple `BertLayer` modules.
    """
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
        """
        Prepares and extends an attention mask for broadcasting across attention heads.
        Converts a (batch_size, seq_len) mask to (batch_size, 1, 1, seq_len)
        and applies a large negative value to masked positions.
        """
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
        hidden_states: torch.Tensor, # Input (query embeddings for Q-Former) (B, N_q, D)
        attention_mask: Optional[torch.Tensor] = None, # Self-attention mask for queries (B, N_q)
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None, # Image features (B, N_v, D_v)
        encoder_attention_mask: Optional[torch.Tensor] = None, # Mask for image features (B, N_v)
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        query_length: int = 0, # The length of the query sequence (N_q)
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass for the BERT Encoder.
        """
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
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
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

class Qformer(nn.Module):
    """
    The Querying Transformer (Q-Former) module.
    It distills information from high-dimensional image features into a fixed set of
    learnable query embeddings. This is achieved using self-attention among the queries
    and cross-attention between the queries and the image features.
    The output query embeddings are then fed to a language model 
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        self.bert_encoder = BertEncoder(config)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the Q-Former's sub-modules (e.g., linear layers, LayerNorms).
        This follows a standard initialization strategy similar to Hugging Face's PreTrainedModel.
        """
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
        image_features: torch.Tensor, 
        image_attention_mask: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        """
        Forward pass of the Q-Former.
        Args:
            image_features: A tensor representing the visual features from the image encoder.
                            Expected shape: (batch_size, feature_dim) for a global feature,
                            or (batch_size, num_visual_tokens, feature_dim) for patch-level features.
            image_attention_mask: An optional mask for the image features, shape (batch_size, num_visual_tokens).
                                  Defaults to all ones if not provided, meaning all visual tokens are attended to.
        Returns:
            A tensor of shape (batch_size, num_query_tokens, hidden_size) representing
            the distilled query embeddings, which can then be fed to a language model.
        """
        batch_size = image_features.shape[0]
        device = image_features.device

        query_tokens = self.query_tokens.expand(batch_size, -1, -1) # (batch_size, num_query_tokens, hidden_size)

        query_length = query_tokens.shape[1] 

        query_attention_mask = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=device)

        if image_features.dim() == 2:
            encoder_hidden_states = image_features.unsqueeze(1) 
        else: 
            encoder_hidden_states = image_features

        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                encoder_hidden_states.shape[:-1], dtype=torch.long, device=device
            )

        encoder_outputs = self.bert_encoder(
            hidden_states=query_tokens,
            attention_mask=query_attention_mask, 
            encoder_hidden_states=encoder_hidden_states, 
            encoder_attention_mask=image_attention_mask, 
            return_dict=True,
            query_length=query_length 
        )

        output_query_embeddings = encoder_outputs['last_hidden_state']
        return output_query_embeddings

# --- BiomedCLIPEncoder Class ---
# This class needs to be fully defined here, copying its content from your
# original 'models.trained_models.BioMedClip.encoder' file.
class BiomedCLIPEncoder(nn.Module):
    def __init__(self, model_name, weights_path, img_size=224): # Added img_size as it's common for CLIP
        super().__init__()
        # This part assumes open_clip is installed and the weights_path points to a compatible file
        # The 'pretrained' argument in open_clip.create_model_and_transforms can accept a local path or a Hugging Face repo ID.
        # If weights_path is a local file, ensure it's accessible.
        # If it's a Hugging Face repo ID (e.g., "laion/CLIP-ViT-B-32-laion2b-s34b-b79k"), open_clip will download it.
        try:
            # open_clip.create_model_and_transforms will handle loading from HF Hub if model_name is an HF Hub ID.
            # `pretrained` argument is for loading pretrained weights (can be path or HF Hub ID)
            # If model_name is an HF ID, pretrained can be True for default weights, or a specific HF ID/path.
            # Your original code used weights_path, so let's stick to that.
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=weights_path if weights_path else True, img_size=img_size)
        except Exception as e:
            raise ImportError(f"Failed to load open_clip model. Ensure model_name '{model_name}' and weights_path '{weights_path}' are correct, and 'open_clip_torch' is installed. Error: {e}")

        self.model = model
        self.preprocess = preprocess
        # Ensure feature_dim matches your actual CLIP model's output dimension
        # Common attribute for CLIP models; adjust if your specific model uses a different one.
        self.feature_dim = model.visual.output_dim 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # Move model to device

    def encode_image(self, image_path: str) -> torch.Tensor:
        # Load image using PIL and preprocess
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(processed_image)
            # Normalize features if desired (common practice for CLIP embeddings)
            features = features / features.norm(p=2, dim=-1, keepdim=True) 
        return features


# --- XrayReportGenerator Class ---
# This is your main model class, copied from biogpt_model.py.
# The decorator is correctly placed here.
@AutoModel.register(model_type="xray_report_generator")
class XrayReportGenerator(nn.Module):
    def __init__(self, biomedclip_model_name, biomedclip_weights_path, qformer_config,
                 biogpt_weights_path: Optional[str] = None):
        super().__init__()
        
        # BiomedCLIPEncoder is now defined in this same file
        self.biomedclip_encoder = BiomedCLIPEncoder(
            model_name=biomedclip_model_name,
            weights_path=biomedclip_weights_path
        )

        assert qformer_config.encoder_width == self.biomedclip_encoder.feature_dim, \
            "Q-Former encoder_width must match BiomedCLIP feature_dim"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        # Qformer is now defined in this same file
        self.qformer = Qformer(qformer_config) # Pass the Qformer config directly
        self.qformer.to(self.device) # Move Qformer to device

        # Initialize tokenizer first, as it's used for eos_token_id and pad_token_id
        # Use your repo ID for the tokenizer, as you uploaded it there
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt") 
        # You need to ensure the tokenizer with special tokens is loaded if it's from your repo
        # self.tokenizer = AutoTokenizer.from_pretrained(repo_id) # if you uploaded it directly to your model repo
        # If the special tokens were added AFTER loading "microsoft/biogpt" during training,
        # then the logic in `ReportGenerationDataset` where you `add_special_tokens`
        # needs to be reflected here or the tokenizer needs to be saved AFTER those additions.
        # For simplicity, if you uploaded the tokenizer files to your HF repo, use:
        # self.tokenizer = AutoTokenizer.from_pretrained(repo_id) 
        # otherwise, ensure the additions are made:
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            warnings.warn("Tokenizer pad_token is None, setting to eos_token.")


        self.biogpt_decoder = BioGptForCausalLM.from_pretrained("microsoft/biogpt") 
        self.biogpt_decoder.to(self.device) # Move decoder to device

        if biogpt_weights_path:
            print(f"Loading fine-tuned BioGPT weights from: {biogpt_weights_path}")
            # Ensure biogpt_weights_path is accessible (e.g., local path or HF Hub repo_id/filename)
            try:
                state_dict = torch.load(biogpt_weights_path, map_location='cpu') 
                self.biogpt_decoder.load_state_dict(state_dict) 
                print("Fine-tuned BioGPT weights loaded successfully.")
            except FileNotFoundError:
                print(f"Warning: Fine-tuned BioGPT weights file not found at {biogpt_weights_path}. Using default pre-trained BioGPT.")
            except Exception as e:
                print(f"Warning: Error loading fine-tuned BioGPT weights from {biogpt_weights_path}: {e}. Using default pre-trained BioGPT.")
        else:
            print("No fine-tuned BioGPT weights file provided, using default pre-trained BioGPT.")

        self.biogpt_hidden_size = self.biogpt_decoder.config.hidden_size # Store this as an instance attribute

        if qformer_config.hidden_size != self.biogpt_hidden_size:
            self.qformer_output_to_biogpt_input_projection = nn.Linear(
                qformer_config.hidden_size, self.biogpt_hidden_size
            ).to(self.device) # Move projection layer to device
        else:
            self.qformer_output_to_biogpt_input_projection = None

        self.eos_token_id = self.tokenizer.eos_token_id # Set after tokenizer setup

    def forward(self,
        image_path: Optional[str] = None, 
        prompt_text: Optional[str] = None,
        max_new_tokens: int = 50,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        image_features: Optional[torch.Tensor] = None, # For direct feature input (e.g., during training)
        input_ids: Optional[torch.Tensor] = None, # For language model input (e.g., during training)
        attention_mask: Optional[torch.Tensor] = None, # For language model attention mask (e.g., during training)
        **kwargs # Catch-all for other generation parameters
        ):
        
        is_training = image_features is not None and input_ids is not None and attention_mask is not None 
        
        # Handle image encoding or direct feature input
        if image_path is not None and not is_training:
            image_features = self.biomedclip_encoder.encode_image(image_path)
            # image_features is already on self.device from BiomedCLIPEncoder.encode_image
        elif image_features is None and not is_training and not prompt_text: # If no image_features, and not training, and no prompt, this is an error
            raise ValueError("Either image_path, image_features, or prompt_text must be provided for inference.")
        
        query_embeddings = None # Initialize query_embeddings for scope

        # If image_features are provided (either from path or directly)
        if image_features is not None:
            if image_features.ndim == 1:
                image_features = image_features.unsqueeze(0)
            
            # Ensure image_features are on the correct device
            image_features = image_features.to(self.device)

            # Expand image_features for Q-Former (assuming Q-Former expects [batch_size, 1, feature_dim])
            image_features_expanded = image_features.unsqueeze(1) 
            query_embeddings = self.qformer(image_features_expanded)

            # Project Q-Former output if necessary
            if self.qformer_output_to_biogpt_input_projection:
                query_embeddings = self.qformer_output_to_biogpt_input_projection(query_embeddings)
            
            # Ensure query_embeddings are on the correct device
            query_embeddings = query_embeddings.to(self.device)

        # --- Training Mode ---
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
                -100, # -100 is typically used for tokens that should be ignored in loss calculation
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
            
        # --- Inference Mode ---
        else:
            if query_embeddings is None:
                # If no image features, and not training, but there is a prompt,
                # it means we might be trying to run the LLM without image input.
                # Your model is multimodal, so image features are expected for generation.
                raise ValueError("For inference, image features (from image_path or image_features) must be provided.")

            input_embeddings_list = [query_embeddings]
            input_attention_mask_list = [torch.ones(query_embeddings.shape[0], query_embeddings.shape[1], dtype=torch.long, device=self.device)]

            if prompt_text:
                # Tokenize prompt text
                prompt_token_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
                prompt_token_ids = prompt_token_ids.to(self.device)
                
                # Get embeddings for prompt tokens
                text_embeddings = self.biogpt_decoder.get_input_embeddings()(prompt_token_ids)
                
                input_embeddings_list.append(text_embeddings)
                input_attention_mask_list.append(torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], dtype=torch.long, device=self.device))
            
            # Concatenate image-derived embeddings with text embeddings
            input_embeddings = torch.cat(input_embeddings_list, dim=1)
            input_attention_mask = torch.cat(input_attention_mask_list, dim=1)

            # Generate report using BioGPT decoder
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

            # Decode the generated report
            generated_report = self.tokenizer.decode(generated_output[0], skip_special_tokens=True) 

            return generated_report
