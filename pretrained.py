import os
import random
from config import AdaptationConfig, Config
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from logger import logger
from sentence_reader import SentenceReader


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


def run_domain_adaptation(
    cfg: AdaptationConfig
):
    """领域适配预训练函数"""
    logger.info(f"Starting domain adaptation with parameters: {locals()}")

    # 设置随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        logger.info("CUDA is available. Using GPU for adaptation.")
    else:
        logger.info("CUDA not available. Using CPU for adaptation.")

    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
        model = AutoModelForMaskedLM.from_pretrained(cfg.base_model_name)
    except Exception as e:
        logger.error(
            f"Error loading model or tokenizer from {cfg.base_model_name}: {e}")
        return None

    # 确保输出目录存在
    if not os.path.exists(cfg.adapted_model_dir):
        os.makedirs(cfg.adapted_model_dir)
        logger.info(f"Created output directory: {cfg.adapted_model_dir}")

    # 加载领域语料
    sentence_reader = SentenceReader()
    texts = sentence_reader.read_corpus(cfg.corpus_file)
    logger.info(f"加载了 {len(texts)} 条语料 from {cfg.corpus_file}")
    if not texts:
        logger.error("No texts loaded from corpus. Aborting adaptation.")
        return None

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=cfg.max_length)

    # 创建Electra数据整理器
    data_collator = ElectraDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mask_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # 准备训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # 优化器
    # Apply weight decay to all parameters other than bias and LayerNorm weights
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": cfg.weight_decay,
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
                                  lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    # 学习率调度器
    total_steps = len(dataloader) * cfg.num_epochs
    scheduler = None
    if total_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
        )
    else:
        logger.warning(
            "Total steps is 0. No training will occur. Check data and batch size.")
        # Save the original model if no training steps
        model.save_pretrained(cfg.adapted_model_dir)
        tokenizer.save_pretrained(cfg.adapted_model_dir)
        logger.info(
            f"No training steps. Original model saved to {cfg.adapted_model_dir}")
        return cfg.adapted_model_dir

    # 训练循环
    global_step = 0
    model.zero_grad()  # Clear gradients before starting training

    logger.info("开始训练 (domain adaptation)")
    for epoch in range(cfg.num_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} (Adaptation)")
        epoch_loss = 0
        batch_count = 0
        for step, batch in enumerate(epoch_iterator):
            generator.train()
            discriminator.train()

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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({
                "total_loss": total_loss.item(),
                "gen_loss": loss_gen.item(),
                "disc_loss": loss_disc.item()
            })

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            logger.info(
                f"Epoch {epoch+1} (Adaptation) 平均损失: {avg_epoch_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} (Adaptation) had no batches.")

    # 保存最终模型
    model.save_pretrained(cfg.adapted_model_dir)
    tokenizer.save_pretrained(cfg.adapted_model_dir)
    logger.info(f"领域适配预训练完成，最终模型保存到 {cfg.adapted_model_dir}")

    return cfg.adapted_model_dir


if __name__ == "__main__":
    # This part is for standalone execution of this script
    config = AdaptationConfig('pretrained.yaml')

    run_domain_adaptation(config)
