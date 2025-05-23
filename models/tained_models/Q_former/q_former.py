import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import warnings
from typing import Optional, Tuple, Dict, Any

# --- Mock/Helper Imports for internal BERT-like components ---
# These are minimal mocks to allow the BERT-like components to run independently
# without requiring the full 'transformers' library for their base classes/utilities.

class MockLogger:
    """A minimal logger mock to prevent errors from missing 'logging' module."""
    def warn(self, message):
        print(f"WARNING: {message}")
logger = MockLogger()

class ModelOutput:
    """A minimal mock for Hugging Face's ModelOutput base class."""
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
    A mock for BaseModelOutputWithPastAndCrossAttentions, used by BertEncoder.
    It simulates the return structure of a BERT encoder with attention outputs and past key values.
    """
    def __init__(self, last_hidden_state: torch.Tensor, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, hidden_states: Optional[Tuple[torch.Tensor]] = None, attentions: Optional[Tuple[torch.Tensor]] = None, cross_attentions: Optional[Tuple[torch.Tensor]] = None):
        super().__init__(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )

ACT2FN = {
    """A dictionary mapping activation function names to PyTorch functions."""
    "gelu": F.gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

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

# --- BertConfig for Q-Former ---
class BertConfig:
    """
    Configuration for the Q-Former's internal BERT-like layers.
    This defines the architecture parameters for the Transformer blocks
    that make up the Q-Former.
    """
    def __init__(self,
                 hidden_size: int = 768, # Dimensionality of the query embeddings and internal representations
                 num_hidden_layers: int = 6, # Number of Transformer layers in the Q-Former
                 num_attention_heads: int = 12, # Number of attention heads in each attention layer
                 intermediate_size: int = 3072, # Size of the "intermediate" (feed-forward) layer
                 hidden_act: str = "gelu", # Activation function for the intermediate layer
                 hidden_dropout_prob: float = 0.1, # Dropout probability for hidden states
                 attention_probs_dropout_prob: float = 0.1, # Dropout probability for attention weights
                 initializer_range: float = 0.02, # Standard deviation for weight initialization
                 layer_norm_eps: float = 1e-12, # Epsilon for layer normalization
                 add_cross_attention: bool = True, # Enable cross-attention to image features
                 cross_attention_freq: int = 1, # How often cross-attention layers appear (e.g., 1 means every layer)
                 encoder_width: int = 512, # Dimensionality of the input image features from BiomedCLIP
                 num_query_tokens: int = 32, # Number of learnable query embeddings
                 gradient_checkpointing: bool = False): # Enable gradient checkpointing for memory saving
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.add_cross_attention = add_cross_attention
        self.cross_attention_freq = cross_attention_freq
        self.encoder_width = encoder_width
        self.num_query_tokens = num_query_tokens
        self.gradient_checkpointing = gradient_checkpointing
        # Dummy values for compatibility with general BERT component expectations,
        # but not directly used by the Q-Former's core logic.
        self.vocab_size = 30522
        self.max_position_embeddings = 512
        self.pad_token_id = 0
        self.position_embedding_type = "absolute"

# --- BERT-like Components (Building Blocks for Q-Former) ---

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
            # For cross-attention, key and value layers operate on the encoder_width (image features)
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            # For self-attention, key and value layers operate on the hidden_size (query embeddings)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # Relative positional embeddings (not typically used in Q-Former but kept for compatibility)
        if (self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False # Flag for saving attention maps (e.g., for visualization)

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
        hidden_states: torch.Tensor, # Input sequence (queries for Q-Former)
        attention_mask: Optional[torch.Tensor] = None, # Mask for self-attention
        head_mask: Optional[torch.Tensor] = None, # Mask for attention heads
        encoder_hidden_states: Optional[torch.Tensor] = None, # Encoder output (image features for cross-attention)
        encoder_attention_mask: Optional[torch.Tensor] = None, # Mask for encoder output
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # Cached past key/value states
        output_attentions: bool = False, # Whether to output attention probabilities
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for self-attention or cross-attention.
        """
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            # If cross-attention, keys and values come from encoder_hidden_states
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask # Use encoder's mask for cross-attention
        elif past_key_value is not None:
            # If using past_key_value, concatenate with current keys/values (for generation)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # For standard self-attention, keys and values come from hidden_states
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer) # Cache current key/value for next steps

        # Compute raw attention scores (query dot key)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Apply relative positional embeddings if configured (not typical for Q-Former's queries)
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
            # Apply the attention mask (e.g., for padding or causal masking)
            attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply dropout to attention probabilities
        attention_probs_dropped = self.dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        # Compute context layer (attention probabilities dot value)
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        # Reshape context layer back to original hidden state dimensions
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,) # Always return past_key_value for compatibility

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
        self.pruned_heads = set() # For head pruning (not implemented in this example)

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
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:] # Add attentions if requested
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
        self.seq_len_dim = 1 # Dimension along which to chunk for feed-forward
        self.attention = BertAttention(config) # Self-attention for queries
        self.layer_num = layer_num

        # Initialize cross-attention if enabled and at the specified frequency
        if self.config.add_cross_attention and layer_num % self.config.cross_attention_freq == 0:
            self.crossattention = BertAttention(
                config, is_cross_attention=True # Explicitly set to True for cross-attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate = BertIntermediate(config) # Standard FFN for general BERT
        self.output = BertOutput(config)

        # Separate FFNs for queries that have interacted with image features
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor, # Query embeddings (B, N_q, D)
        attention_mask: Optional[torch.Tensor] = None, # Self-attention mask for queries
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None, # Image features (B, N_v, D_v)
        encoder_attention_mask: Optional[torch.Tensor] = None, # Cross-attention mask for image features
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        query_length: int = 0, # The length of the query sequence (N_q)
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for a single BERT layer within the Q-Former.
        """
        # 1. Self-attention on the query embeddings
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states, # Input are the queries
            attention_mask, # Mask for self-attention of queries
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0] # Output after self-attention (B, N_q, D)
        outputs = self_attention_outputs[1:-1] # Contains attentions if output_attentions is True

        present_key_value = self_attention_outputs[-1] # Cached key/value for next step (not critical for Q-Former)

        # 2. Cross-attention to image features (if enabled for this layer)
        if self.has_cross_attention:
            assert (
                encoder_hidden_states is not None
            ), "encoder_hidden_states must be given for cross-attention layers when has_cross_attention is True"

            cross_attention_outputs = self.crossattention(
                attention_output, # Queries (output of self-attention)
                attention_mask, # This mask is for the queries (not for encoder_hidden_states)
                head_mask,
                encoder_hidden_states, # Image features (keys/values for cross-attention)
                encoder_attention_mask, # Mask for image features
                output_attentions=output_attentions,
            )
            cross_attention_output = cross_attention_outputs[0] # Output after cross-attention
            outputs = (outputs + cross_attention_outputs[1:-1]) # Add cross-attentions if requested

            # 3. Apply query-specific Feed-Forward Network (FFN)
            # In Q-Former, `hidden_states` are exclusively queries, so we always use the query-specific FFN.
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query, # Use FFN specifically for queries
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                cross_attention_output,
            )
        else:
            # If no cross-attention, just apply the standard FFN (less common for Q-Former layers)
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
        has_query: bool = False, # Not typically used for encoder masks
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
            # Convert to float and apply large negative value for masked positions
            extended_attention_mask = extended_attention_mask.to(dtype=self.layer[0].attention.self.query.weight.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # A common masking value
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

        # Pre-process attention masks once before iterating through layers
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

            # Apply gradient checkpointing if enabled
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

            hidden_states = layer_outputs[0] # Output of the current layer
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
        # Return as a BaseModelOutputWithPastAndCrossAttentions object for structured output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# --- Qformer Implementation ---

class Qformer(nn.Module):
    """
    The Querying Transformer (Q-Former) module.
    It distills information from high-dimensional image features into a fixed set of
    learnable query embeddings. This is achieved using self-attention among the queries
    and cross-attention between the queries and the image features.
    The output query embeddings are then fed to a language model (e.g., BioGPT).
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        # Learnable query tokens: These are the core input to the Q-Former's BERT encoder.
        # They are initialized randomly and will be updated during training to learn
        # to extract relevant visual information from the image features.
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        # Initialize query tokens with a normal distribution
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)

        # The BERT-like encoder that processes the query tokens and cross-attends to image features.
        self.bert_encoder = BertEncoder(config)

        # Initialize weights for the Q-Former module's layers.
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
        image_features: torch.Tensor, # Output from BiomedCLIPEncoder (batch_size, feature_dim) or (batch_size, num_patches, feature_dim)
        image_attention_mask: Optional[torch.Tensor] = None, # Mask for image features if needed
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

        # Expand the learnable query tokens to match the batch size.
        # This creates a batch of identical query token sets.
        query_tokens = self.query_tokens.expand(batch_size, -1, -1) # (batch_size, num_query_tokens, hidden_size)

        # The `hidden_states` argument to BertEncoder will be the query tokens.
        # The `query_length` parameter in BertEncoder/BertLayer is used to tell the layers
        # that the input sequence consists entirely of "query" tokens, which might
        # receive special FFN treatment (like `feed_forward_chunk_query`).
        query_length = query_tokens.shape[1] # This is simply config.num_query_tokens

        # Create a "ones" attention mask for the query tokens.
        # This mask indicates that all query tokens can fully attend to each other
        # in the self-attention mechanism within the Q-Former.
        query_attention_mask = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=device)

        # Ensure image_features is formatted as a sequence of visual tokens for cross-attention.
        # If BiomedCLIP outputs a single global feature vector (batch_size, feature_dim),
        # we unsqueeze it to make it a sequence of one token (batch_size, 1, feature_dim).
        if image_features.dim() == 2:
            encoder_hidden_states = image_features.unsqueeze(1) # (batch_size, 1, feature_dim)
        else: # Assumed (batch_size, num_visual_tokens, feature_dim)
            encoder_hidden_states = image_features

        # Create an attention mask for the image features if not provided.
        # If no mask is given, assume all visual tokens are relevant and should be attended to.
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                encoder_hidden_states.shape[:-1], dtype=torch.long, device=device
            ) # (batch_size, num_visual_tokens)

        # Pass the query tokens and image features through the BERT encoder.
        # The query tokens are the `hidden_states` that undergo self-attention.
        # The image features are the `encoder_hidden_states` that the queries cross-attend to.
        encoder_outputs = self.bert_encoder(
            hidden_states=query_tokens,
            attention_mask=query_attention_mask, # Self-attention mask for queries
            encoder_hidden_states=encoder_hidden_states, # Image features from BiomedCLIP
            encoder_attention_mask=image_attention_mask, # Cross-attention mask for image features
            return_dict=True,
            query_length=query_length # Pass query_length to guide FFN selection in BertLayer
        )

        # The `last_hidden_state` from the encoder's output will be the refined query embeddings
        # after they have processed the image information.
        output_query_embeddings = encoder_outputs.last_hidden_state
        return output_query_embeddings # Final shape: (batch_size, num_query_tokens, hidden_size)
