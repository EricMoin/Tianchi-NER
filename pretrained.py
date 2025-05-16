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
    if not os.path.exists(file_path):
        logger.error(f"Corpus file not found: {file_path}")
        return texts
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                texts.append(line)
    return texts


def run_domain_adaptation(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    num_train_epochs: int = 3,
    warmup_steps: int = 0,
    save_intermediate_steps: int = 0,
    mask_probability: float = 0.15,
    seed: int = 42
):
    """领域适配预训练函数"""
    logger.info(f"Starting domain adaptation with parameters: {locals()}")

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA is available. Using GPU for adaptation.")
    else:
        logger.info("CUDA not available. Using CPU for adaptation.")

    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.error(
            f"Error loading model or tokenizer from {model_name_or_path}: {e}")
        return None

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # 加载领域语料
    texts = read_domain_corpus(train_file)
    logger.info(f"加载了 {len(texts)} 条语料 from {train_file}")
    if not texts:
        logger.error("No texts loaded from corpus. Aborting adaptation.")
        return None

    # 创建dataset
    dataset = DomainDataset(texts, tokenizer, max_length=max_length)

    # 创建数据整理器用于全词掩码
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mask_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
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
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if not any(pg["params"] for pg in optimizer_grouped_parameters):
        logger.error(
            "No parameters to optimize. Check model and no_decay list.")
        return None

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=learning_rate, eps=adam_epsilon)

    # 学习率调度器
    total_steps = len(dataloader) * num_train_epochs
    scheduler = None
    if total_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        logger.warning(
            "Total steps is 0. No training will occur. Check data and batch size.")
        # Save the original model if no training steps
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"No training steps. Original model saved to {output_dir}")
        return output_dir

    # 训练循环
    global_step = 0
    model.zero_grad()  # Clear gradients before starting training

    logger.info("开始训练 (domain adaptation)")
    for epoch in range(num_train_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs} (Adaptation)")
        epoch_loss = 0
        batch_count = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()

            # 将数据移到设备上
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
            except AttributeError:
                logger.error(
                    f"Error moving batch to device. Batch type: {type(batch)}, keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
                continue  # Skip this batch

            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

            if loss is None:
                logger.warning(
                    f"Loss is None at epoch {epoch+1}, step {step}. Skipping batch.")
                continue

            epoch_loss += loss.item()
            batch_count += 1

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({"loss": loss.item()})

            # 保存中间模型检查点
            if save_intermediate_steps > 0 and global_step % save_intermediate_steps == 0:
                checkpoint_dir = os.path.join(
                    output_dir, f"checkpoint-{global_step}")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"保存模型检查点到 {checkpoint_dir}")

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(
                f"Epoch {epoch+1} (Adaptation) 平均损失: {avg_epoch_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} (Adaptation) had no batches.")

    # 保存最终模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"领域适配预训练完成，最终模型保存到 {output_dir}")

    return output_dir


if __name__ == "__main__":
    # This part is for standalone execution of this script
    import argparse
    parser = argparse.ArgumentParser(description="领域适配预训练 (Standalone)")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-roberta-wwm-ext", help="基础预训练模型名称或路径")
    parser.add_argument("--train_file", type=str,
                        default="data/address.txt", help="领域语料文件")
    parser.add_argument("--output_dir", type=str,
                        default="pretrained/address_adapted_standalone", help="保存模型的目录")
    parser.add_argument("--num_train_epochs", type=int,
                        default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int,
                        default=128, help="Max sequence length")
    # Add more arguments as needed from run_domain_adaptation signature
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")

    args = parser.parse_args()
    logger.info(f"Running standalone domain adaptation with args: {args}")

    run_domain_adaptation(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed
        # Pass other arguments from parser if added, or rely on defaults
    )
