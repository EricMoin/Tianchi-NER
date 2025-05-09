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
            # loss = -self.crf(logits, labels, mask=attention_mask.bool())
            crf_loss = CRFAwareLoss(self.crf)
            loss = crf_loss(logits, labels, attention_mask.bool())
            return loss.mean()
        else:
            # During inference, decode the best path
            return self.crf.viterbi_decode(logits, mask=attention_mask.bool())

    def __len__(self):
        return len(self.data)


class PGD:
    def __init__(self, model, epsilon=0.68, alpha=0.3, steps=2):
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


class CRFAwareLoss(nn.Module):
    def __init__(self, crf: CRF, transition_penalty=0.175):
        super().__init__()
        self.crf = crf
        self.transition_probs = F.softmax(
            self.crf.trans_matrix, dim=1).detach()
        self.transition_penalty = transition_penalty
        # 获取 CRF 的转移矩阵（形状: [num_tags, num_tags]）
        self.transition_matrix = crf.trans_matrix.detach()

    def forward(self, emissions, tags, mask):
        # 常规 CRF 负对数似然损失
        crf_loss = -self.crf(emissions, tags, mask=mask)

        # 计算标签转移的不合理性惩罚
        batch_size, seq_len = tags.shape
        penalty = 0.0
        for i in range(seq_len - 1):
            current_tags = tags[:, i]
            next_tags = tags[:, i + 1]
            # 对每对连续标签计算转移概率的负值（越小越合理）
            invalid_transitions = - \
                torch.log(
                    self.transition_probs[current_tags, next_tags] + 1e-8)
            penalty += invalid_transitions.mean()

        # 总损失 = CRF 损失 + 惩罚项
        total_loss = crf_loss + self.transition_penalty * penalty
        return total_loss
