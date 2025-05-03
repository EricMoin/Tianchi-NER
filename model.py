import torch.nn as nn
from torch.utils.data import Dataset
from TorchCRF import CRF
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


class AddressNER(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, freeze_bert_layers=8):
        super(AddressNER, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels=num_labels)

        # Freeze BERT layers
        self._freeze_bert_layers(freeze_bert_layers)

    def _freeze_bert_layers(self, num_layers_to_freeze):
        """Freeze the first num_layers_to_freeze layers of BERT"""
        if num_layers_to_freeze <= 0:
            return

        # Always freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first n encoder layers
        for layer_idx in range(min(num_layers_to_freeze, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Pass BERT output directly to LSTM
        lstm_output, _ = self.lstm(bert_output)

        # Apply dropout and classification
        attended_output = self.dropout(lstm_output)
        logits = self.classifier(attended_output)

        if labels is not None:
            # During training, calculate the loss
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss.mean()
        else:
            # During inference, decode the best path
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

    def __len__(self):
        return len(self.data)


class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model, epsilon=0.68, alpha=0.3, steps=3):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha      # Step size
        self.steps = steps      # Number of attack iterations
        self.backup = {}
        self.emb_backup = {}

    def attack(self, emb_name='bert.embeddings.word_embeddings.weight', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                    # 保存初始参数
                    self.backup[name] = param.data.clone()

                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)

                    # 投影回 epsilon 球
                    param.data = self.project(name, param.data, self.epsilon)

    def project(self, param_name, param_data, epsilon):
        # 将扰动投影到epsilon球上
        delta = param_data - self.emb_backup[param_name]
        norm = torch.norm(delta)
        if norm > epsilon:
            delta = delta * epsilon / norm
        return self.emb_backup[param_name] + delta

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def restore_emb(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
