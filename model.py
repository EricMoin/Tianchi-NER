import torch.nn as nn
from torch.utils.data import Dataset
from TorchCRF import CRF
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
        mask = mask.float() / (1 - self.drop_prob)  # Scale to maintain expected value

        # Broadcast mask along sequence dimension without adding a new dimension
        # Final shape remains [batch_size, seq_len, hidden_dim]
        return inputs * mask


class AddressNER(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super(AddressNER, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add embedding dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        # Add spatial dropout
        self.spatial_dropout = SpatialDropout(config.spatial_dropout)

        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels=num_labels)

        # Initialize losses
        self.crf_loss = CRFAwareLoss(
            self.crf, transition_penalty=config.crf_transition_penalty)
        self.focal_loss = FocalLoss(
            num_labels, alpha=config.focal_loss_alpha, gamma=config.focal_loss_gamma)
        self.hybrid_loss = HybridLoss(
            self.crf_loss,
            self.focal_loss,
            crf_weight=config.hybrid_loss_weight_crf,
            focal_weight=config.hybrid_loss_weight_focal
        )

        # Freeze BERT layers
        self._freeze_bert_layers(config.freeze_bert_layers)

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
        # Apply embedding dropout directly to the input embeddings
        if self.training:
            # Get the embeddings
            embeddings = self.bert.embeddings.word_embeddings(input_ids)
            # Apply dropout to embeddings
            embeddings = self.embedding_dropout(embeddings)
            # Pass modified embeddings through BERT
            outputs = self.bert(inputs_embeds=embeddings,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)

        bert_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        # Apply spatial dropout to BERT output
        bert_output = self.spatial_dropout(bert_output)

        # Pass BERT output directly to LSTM
        lstm_output, _ = self.lstm(bert_output)

        logits = self.classifier(lstm_output)

        if labels is not None:
            # During training, calculate the hybrid loss
            loss = self.hybrid_loss(logits, labels, attention_mask.bool())
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
            current_tags = tags[:, i].to(self.transition_probs.device)
            next_tags = tags[:, i + 1].to(self.transition_probs.device)
            # 对每对连续标签计算转移概率的负值（越小越合理）
            invalid_transitions = - \
                torch.log(
                    self.transition_probs[current_tags, next_tags] + 1e-8)
            penalty += invalid_transitions.mean()

        # 总损失 = CRF 损失 + 惩罚项
        total_loss = crf_loss + self.transition_penalty * penalty
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        """
        计算Focal Loss

        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 目标标签 [batch_size, seq_len]
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        # 将logits转换为概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类别的概率
        batch_size, seq_len, _ = logits.shape

        # 创建one-hot编码
        target_one_hot = F.one_hot(targets, self.num_classes).float()

        # 计算每个位置的概率
        pt = (probs * target_one_hot).sum(dim=-1)  # [batch_size, seq_len]

        # 计算focal loss
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.ones_like(pt) * self.alpha
        alpha_weight = torch.where(
            targets > 0, alpha_weight, 1 - alpha_weight)  # 对背景类使用1-alpha

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, self.num_classes),
            targets.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        # 应用权重
        loss = alpha_weight * focal_weight * ce_loss

        # 应用掩码并执行reduction
        loss = loss * mask.float()

        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-10)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class HybridLoss(nn.Module):
    def __init__(self, crf_loss, focal_loss, crf_weight=0.5, focal_weight=0.5):
        super(HybridLoss, self).__init__()
        self.crf_loss = crf_loss
        self.focal_loss = focal_loss
        self.crf_weight = crf_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets, mask):
        """
        计算混合损失

        Args:
            logits: 模型输出 [batch_size, seq_len, num_classes]
            targets: 目标标签 [batch_size, seq_len]
            mask: 有效位置掩码 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        crf_loss_val = self.crf_loss(logits, targets, mask)
        focal_loss_val = self.focal_loss(logits, targets, mask)

        return self.crf_weight * crf_loss_val + self.focal_weight * focal_loss_val
