import os
import random
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, whole_word_mask=False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.whole_word_mask = whole_word_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


def read_domain_corpus(file_path):
    """读取领域语料文件"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                texts.append(line)
    return texts


def create_masked_input(input_ids, tokenizer, mask_probability=0.15, whole_word_mask=False):
    """创建掩码输入:
       - 当whole_word_mask为True时，掩码整个词
       - 当whole_word_mask为False时，随机掩码单个token
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_probability)

    # 创建特殊token的掩码（不对特殊token进行掩码）
    special_tokens_mask = torch.tensor([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ])
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

    # 获取注意力掩码
    padding_mask = labels.eq(tokenizer.pad_token_id)
    padding_mask = padding_mask.to(probability_matrix.device)
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    if whole_word_mask:
        # 批处理方式实现全词掩码
        for i in range(input_ids.size(0)):  # 遍历batch中的每个样本
            # 为当前样本创建词掩码
            word_begins = []  # 记录词开始的位置
            word_ends = []    # 记录词结束的位置

            # 标记哪些位置有效（非特殊token，非填充）
            valid_positions = (
                ~special_tokens_mask[i].bool() & ~padding_mask[i]).tolist()
            tokens = [tokenizer.convert_ids_to_tokens(
                input_ids[i, j].item()) for j in range(len(input_ids[i]))]

            # 识别词边界
            j = 0
            while j < len(valid_positions):
                if not valid_positions[j]:
                    j += 1
                    continue

                # 词的开始
                if not tokens[j].startswith('##'):
                    word_start = j
                    j += 1
                    # 查找词的结束
                    while j < len(valid_positions) and valid_positions[j] and tokens[j].startswith('##'):
                        j += 1
                    word_end = j - 1

                    word_begins.append(word_start)
                    word_ends.append(word_end)
                else:
                    j += 1

            # 决定哪些词要被掩码
            for start, end in zip(word_begins, word_ends):
                # 以mask_probability的概率决定掩码整个词
                if random.random() < mask_probability:
                    probability_matrix[i, start:end+1] = 1.0

    # 选择需要掩码的token
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 将未掩码的位置设为-100，这样它们在计算损失时会被忽略

    # 80%的掩码token替换为[MASK]
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10%的掩码token替换为随机token
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random].to(
        input_ids.device)

    return input_ids, labels


def train(args):
    """训练函数"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载领域语料
    texts = read_domain_corpus(args.train_file)
    logger.info(f"加载了 {len(texts)} 条语料")

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=args.max_length, whole_word_mask=args.whole_word_mask)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 准备训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=args.learning_rate, eps=args.adam_epsilon)

    # 学习率调度器
    total_steps = len(dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # 训练循环
    global_step = 0
    model.zero_grad()

    logger.info("开始训练")
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 创建掩码输入
            masked_input_ids, labels = create_masked_input(
                input_ids.clone(),
                tokenizer,
                mask_probability=args.mask_probability,
                whole_word_mask=args.whole_word_mask
            )

            # 前向传播
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({"loss": loss.item()})

            # 保存中间模型检查点
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"保存模型检查点到 {checkpoint_dir}")

        # 保存每个epoch的模型
        epoch_avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} 平均损失: {epoch_avg_loss:.4f}")

        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"保存epoch {epoch+1}的模型到 {epoch_dir}")

    # 保存最终模型
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"训练完成，最终模型保存到 {args.output_dir}")

    return args.output_dir


if __name__ == "__main__":
    class Args:
        model_name = "bert-base-chinese"
        train_file = "data/addresses.txt"  # 地址语料文件
        output_dir = "pretrained/address_adapted_model"
        max_length = 128
        batch_size = 16
        learning_rate = 5e-5
        weight_decay = 0.01
        adam_epsilon = 1e-8
        max_grad_norm = 1.0
        num_train_epochs = 3
        warmup_steps = 0
        save_steps = 500
        mask_probability = 0.15
        seed = 42
        whole_word_mask = True  # 开启全词掩码

    args = Args()

    # 提供选择全词掩码或随机掩码的功能
    import argparse
    parser = argparse.ArgumentParser(description="领域适配预训练")
    parser.add_argument("--whole_word_mask",
                        action="store_true", help="使用全词掩码")
    parser.add_argument("--random_mask", action="store_false",
                        dest="whole_word_mask", help="使用随机掩码")
    parser.add_argument("--train_file", type=str,
                        default=args.train_file, help="领域语料文件")
    parser.add_argument("--output_dir", type=str,
                        default=args.output_dir, help="保存模型的目录")
    parser.add_argument("--model_name", type=str,
                        default=args.model_name, help="基础预训练模型名称")
    parser.add_argument("--num_epochs", type=int,
                        default=args.num_train_epochs, help="训练轮数")
    parser.set_defaults(whole_word_mask=True)

    parsed_args = parser.parse_args()
    args.whole_word_mask = parsed_args.whole_word_mask
    args.train_file = parsed_args.train_file
    args.output_dir = parsed_args.output_dir
    args.model_name = parsed_args.model_name
    args.num_train_epochs = parsed_args.num_epochs

    mask_type = "全词掩码" if args.whole_word_mask else "随机掩码"
    logger.info(f"使用{mask_type}进行领域适配预训练")

    output_dir = train(args)
    logger.info(f"领域适配预训练完成，模型保存在 {output_dir}")
