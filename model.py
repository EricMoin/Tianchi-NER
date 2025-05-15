import torch.nn as nn
from torch.utils.data import Dataset
# CRF and related losses are removed
# from TorchCRF import CRF
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

from config import Config


class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        """
        Spatial dropout: drops entire feature maps/channels rather than individual elements
        """
        if not self.training or self.drop_prob == 0:
            return inputs

        # inputs shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = inputs.shape

        # Create mask that drops the same channels for all sequence positions
        # Shape: [batch_size, 1, hidden_dim]
        mask = torch.rand(batch_size, 1, hidden_dim,
                          device=inputs.device) > self.drop_prob
        mask = mask.float() / (1 - self.drop_prob if self.drop_prob <
                               1 else 1.0)  # Scale, avoid div by zero

        # Broadcast mask along sequence dimension without adding a new dimension
        # Final shape remains [batch_size, seq_len, hidden_dim]
        return inputs * mask


class Biaffine(nn.Module):
    def __init__(self, in_features, out_features, bias_start=True, bias_end=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # num_labels for spans

        self.U = nn.Parameter(torch.Tensor(
            out_features, in_features, in_features))
        self.W_start = nn.Linear(in_features, out_features, bias=bias_start)
        self.W_end = nn.Linear(in_features, out_features, bias=bias_end)
        # If both W_start and W_end have bias, ensure they are not redundant or use a single separate bias
        # For simplicity, let W_start handle the main bias, W_end can be bias=False or its bias adds if bias_end=True.
        # If bias_start and bias_end are true, then effectively two bias terms contribute. Often one is set to False.
        # Let's adjust: W_start with bias, W_end without to avoid redundancy if a single overall bias per label is desired.
        if bias_start and bias_end:  # A common setup
            # Allow both for flexibility
            self.W_end = nn.Linear(in_features, out_features, bias=True)
        elif bias_start:  # W_start has bias, W_end no bias
            self.W_end = nn.Linear(in_features, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        # nn.Linear layers (W_start, W_end) are initialized by default

    def forward(self, start_reps, end_reps):
        # start_reps: [batch, seq_len, in_features]
        # end_reps: [batch, seq_len, in_features]
        # U: [out_features, in_features, in_features]

        # Bilinear term: (h_i^start)^T U_l h_j^end
        # [B, Out, S_start, In]
        bilin_left = torch.einsum('bsi,oij->bosj', start_reps, self.U)
        span_scores_bilinear = torch.einsum(
            'bosj,bej->bose', bilin_left, end_reps)  # [B, Out, S_start, S_end]
        span_scores_bilinear = span_scores_bilinear.permute(
            0, 2, 3, 1)  # [B, S_start, S_end, Out]

        # Linear terms for start and end tokens
        scores_start = self.W_start(start_reps)  # [B, S_start, Out]
        scores_end = self.W_end(end_reps)    # [B, S_end, Out]

        # Combine: span_scores_bilinear + scores_start (broadcasted over S_end) + scores_end (broadcasted over S_start)
        span_scores = span_scores_bilinear + \
            scores_start.unsqueeze(2) + scores_end.unsqueeze(1)

        return span_scores  # [batch, seq_len_start, seq_len_end, out_features]


class AddressNER(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super(AddressNER, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config = config  # Save config
        self.num_labels = num_labels  # Number of span types including 'O'

        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        bert_hidden_size = self.bert.config.hidden_size  # Typically 768

        # MLPs for start and end representations
        self.mlp_start = nn.Linear(
            bert_hidden_size, config.biaffine_hidden_dim)
        self.mlp_end = nn.Linear(bert_hidden_size, config.biaffine_hidden_dim)

        # Biaffine layer for span scoring
        self.biaffine = Biaffine(config.biaffine_hidden_dim, num_labels)

        # Loss function (CrossEntropyLoss for span classification)
        # Assumes config.ignore_index is defined, e.g., -100
        # Assumes config.O_label_id is defined for non-entity spans
        self.loss_fct = nn.CrossEntropyLoss(
            ignore_index=getattr(config, 'ignore_index', -100))

        # Freeze BERT layers
        self._freeze_bert_layers(config.freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze the first num_layers_to_freeze layers of BERT"""
        if num_layers_to_freeze <= 0:
            return

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    # labels renamed to span_labels_gold
    def forward(self, input_ids, attention_mask, span_labels_gold=None):
        if self.training:
            embeddings = self.bert.embeddings.word_embeddings(input_ids)
            embeddings = self.embedding_dropout(embeddings)
            outputs = self.bert(inputs_embeds=embeddings,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)

        # [batch, seq_len, bert_hidden_size]
        bert_output = outputs.last_hidden_state
        bert_output = self.spatial_dropout(bert_output)

        # Get start and end representations using MLPs
        # [B, S, biaffine_hidden_dim]
        start_reps = F.relu(self.mlp_start(bert_output))
        # [B, S, biaffine_hidden_dim]
        end_reps = F.relu(self.mlp_end(bert_output))

        # Get span logits from Biaffine layer
        # span_logits: [batch, seq_len_start, seq_len_end, num_labels]
        span_logits = self.biaffine(start_reps, end_reps)

        if span_labels_gold is not None:
            # Training mode: Calculate loss
            # span_labels_gold: [batch, seq_len, seq_len], with target class indices or ignore_index

            # Mask for active spans to consider in loss
            seq_len = span_logits.shape[1]
            upper_triangular_mask = torch.triu(
                torch.ones(seq_len, seq_len,
                           device=span_logits.device, dtype=torch.bool)
            )
            start_token_mask = attention_mask.unsqueeze(2)  # [B, S_start, 1]
            end_token_mask = attention_mask.unsqueeze(1)    # [B, 1, S_end]
            # [B, S_start, S_end]
            valid_span_padding_mask = start_token_mask & end_token_mask

            active_span_mask = upper_triangular_mask.unsqueeze(
                0) & valid_span_padding_mask  # [B, S, S]

            if hasattr(self.config, 'max_span_length') and self.config.max_span_length > 0 and self.config.max_span_length < seq_len:
                indices = torch.arange(seq_len, device=span_logits.device)
                span_lengths = indices.unsqueeze(
                    0) - indices.unsqueeze(1) + 1  # [S,S] lengths
                max_len_mask = (span_lengths <= self.config.max_span_length) & (
                    span_lengths > 0)  # Ensure positive length
                active_span_mask = active_span_mask & max_len_mask.unsqueeze(0)

            # Reshape for potentially more memory-efficient selection
            # span_logits: [batch, seq_len_start, seq_len_end, num_labels]
            # active_span_mask: [batch, seq_len_start, seq_len_end]
            # span_labels_gold: [batch, seq_len_start, seq_len_end]

            num_active_spans = active_span_mask.sum()
            if num_active_spans == 0:
                return torch.tensor(0.0, device=span_logits.device, requires_grad=True)

            # Flatten logits, targets, and mask
            # Batch dimension is kept, other dimensions are flattened, then select based on mask
            batch_size = span_logits.shape[0]
            flat_logits = span_logits.view(
                batch_size, -1, self.num_labels)  # [B, S*S, NumLabels]
            flat_targets = span_labels_gold.view(
                batch_size, -1)            # [B, S*S]
            flat_active_mask = active_span_mask.view(
                batch_size, -1)       # [B, S*S]

            # Select active elements for each batch item, then concatenate
            # This avoids creating a giant intermediate tensor if global sum of active_span_mask is huge
            active_logits_list = []
            active_targets_list = []
            for i in range(batch_size):
                active_logits_list.append(flat_logits[i][flat_active_mask[i]])
                active_targets_list.append(
                    flat_targets[i][flat_active_mask[i]])

            # [TotalActiveSpans, NumLabels]
            active_logits = torch.cat(active_logits_list, dim=0)
            active_targets = torch.cat(
                active_targets_list, dim=0)  # [TotalActiveSpans]

            if active_targets.numel() == 0:
                return torch.tensor(0.0, device=span_logits.device, requires_grad=True)

            loss = self.loss_fct(active_logits, active_targets)
            return loss
        else:
            # Inference mode: return raw span_logits
            # Decoding of spans (finding best spans, converting to token tags) should be handled by the Trainer/evaluation script
            return span_logits

    # __len__ method removed as it was incorrect

# PGD class remains the same as it operates on model parameters and gradients


class PGD:
    def __init__(self, model, epsilon=0.68, alpha=0.3, steps=2):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.backup = {}
        self.emb_backup = {}

    def attack(self, emb_name='bert.embeddings.word_embeddings.weight', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                    self.backup[name] = param.data.clone()

                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):  # Add isnan check
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def project(self, param_name, param_data, epsilon):
        delta = param_data - self.emb_backup[param_name]
        norm = torch.norm(delta)
        if norm > epsilon:
            delta = delta * epsilon / norm
        return self.emb_backup[param_name] + delta

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:  # Ensure key exists before restoring
                    param.data = self.backup[name]
        self.backup = {}

    def restore_emb(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.emb_backup:  # Ensure key exists
                    param.data = self.emb_backup[name]
        # self.emb_backup = {} # emb_backup should be persistent across attack steps within an iteration

# Removed CRFAwareLoss, FocalLoss, HybridLoss classes
# class CRFAwareLoss(nn.Module): ...
# class FocalLoss(nn.Module): ...
# class HybridLoss(nn.Module): ...
