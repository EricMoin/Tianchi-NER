import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorForWholeWordMask
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
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

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


def train(args):
    """训练函数"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载领域语料
    texts = read_domain_corpus(args.train_file)
    logger.info(f"加载了 {len(texts)} 条语料")

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=args.max_length)

    # 创建数据整理器用于全词掩码
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mask_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

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

            # 将数据移到设备上
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播 (data_collator已经处理了掩码)
            outputs = model(**batch)
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
        model_name = "hfl/chinese-roberta-wwm-ext"
        train_file = "data/address.txt"  # 地址语料文件
        output_dir = "pretrained/address_adapted_model"
        max_length = 128
        batch_size = 16
        learning_rate = 5e-5
        weight_decay = 0.01
        adam_epsilon = 1e-8
        max_grad_norm = 1.0
        num_train_epochs = 2
        warmup_steps = 0
        save_steps = 500
        mask_probability = 0.15
        seed = 2025

    args = Args()

    import argparse
    parser = argparse.ArgumentParser(description="领域适配预训练")
    parser.add_argument("--train_file", type=str,
                        default=args.train_file, help="领域语料文件")
    parser.add_argument("--output_dir", type=str,
                        default=args.output_dir, help="保存模型的目录")
    parser.add_argument("--model_name", type=str,
                        default=args.model_name, help="基础预训练模型名称")
    parser.add_argument("--num_epochs", type=int,
                        default=args.num_train_epochs, help="训练轮数")

    parsed_args = parser.parse_args()
    args.train_file = parsed_args.train_file
    args.output_dir = parsed_args.output_dir
    args.model_name = parsed_args.model_name
    args.num_train_epochs = parsed_args.num_epochs

    logger.info("使用全词掩码进行领域适配预训练")

    output_dir = train(args)
    logger.info(f"领域适配预训练完成，模型保存在 {output_dir}")
