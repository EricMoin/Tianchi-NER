import torch.nn as nn
from torch.utils.data import Dataset
from TorchCRF import CRF
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


class AddressNER(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, freeze_bert_layers=8, focal_alpha=0.25, focal_gamma=2.0, weight_crf=0.5, weight_focal=0.5, crf_transition_penalty=0.175):
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

        # Initialize HybridLoss
        self.loss_fn = HybridLoss(
            crf_model=self.crf,
            num_classes=num_labels,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            weight_crf=weight_crf,
            weight_focal=weight_focal,
            crf_transition_penalty=crf_transition_penalty
        )

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
            # crf_loss = CRFAwareLoss(self.crf)
            # loss = crf_loss(logits, labels, attention_mask.bool())
            loss = self.loss_fn(logits, labels, attention_mask.bool())
            # Ensure we still return a scalar mean loss if HybridLoss doesn't average internally
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        # inputs: [batch_size, seq_len, num_classes]
        # targets: [batch_size, seq_len]
        # mask: [batch_size, seq_len] (boolean)

        BCE_loss = F.cross_entropy(
            inputs.view(-1, inputs.size(-1)), targets.view(-1), reduction='none')

        if mask is not None:
            # Flatten mask and apply to loss
            mask_flat = mask.view(-1)
            BCE_loss = BCE_loss * mask_flat

        pt = torch.exp(-BCE_loss)  # Prevents nans when BCE_loss is large
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            if mask is not None:
                # Compute mean only over masked elements
                return F_loss.sum() / mask_flat.sum()
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class HybridLoss(nn.Module):
    def __init__(self, crf_model: CRF, num_classes: int, focal_alpha=0.25, focal_gamma=2.0, weight_crf=0.5, weight_focal=0.5, crf_transition_penalty=0.175):
        super().__init__()
        self.crf_aware_loss = CRFAwareLoss(
            crf_model, transition_penalty=crf_transition_penalty)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.weight_crf = weight_crf
        self.weight_focal = weight_focal
        self.num_classes = num_classes

    def forward(self, emissions, tags, mask):
        # emissions: [batch_size, seq_len, num_classes]
        # tags: [batch_size, seq_len]
        # mask: [batch_size, seq_len] (boolean)

        loss_crf = self.crf_aware_loss(emissions, tags, mask)

        # FocalLoss expects class probabilities, CRF emissions are logits.
        # No explicit softmax needed here as FocalLoss uses F.cross_entropy which handles logits.
        loss_focal = self.focal_loss(emissions, tags, mask)

        combined_loss = self.weight_crf * loss_crf + self.weight_focal * loss_focal
        return combined_loss


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
            current_tags = tags[:, i].to(self.transition_matrix.device)
            next_tags = tags[:, i + 1].to(self.transition_matrix.device)
            # 对每对连续标签计算转移概率的负值（越小越合理）
            invalid_transitions = - \
                torch.log(
                    self.transition_probs[current_tags, next_tags] + 1e-8)
            penalty += invalid_transitions.mean()

        # 总损失 = CRF 损失 + 惩罚项
        total_loss = crf_loss + self.transition_penalty * penalty
        return total_loss
