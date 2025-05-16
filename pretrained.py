import os
import random
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


class ElectraDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_input_ids = torch.stack([ex['input_ids'] for ex in examples])
        batch_attention_mask = torch.stack(
            [ex['attention_mask'] for ex in examples])

        original_input_ids = batch_input_ids.clone()

        generator_labels = original_input_ids.clone()

        probability_matrix = torch.full(
            original_input_ids.shape, self.mlm_probability, device=original_input_ids.device)

        special_tokens_mask_list = []
        for val in original_input_ids.tolist():
            special_tokens_mask_list.append(
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
            )
        special_tokens_mask_t = torch.tensor(
            special_tokens_mask_list, dtype=torch.bool, device=original_input_ids.device)
        probability_matrix.masked_fill_(special_tokens_mask_t, value=0.0)

        mlm_masked_indices = torch.bernoulli(probability_matrix).bool()

        # -100 is the ignore index for CrossEntropyLoss
        generator_labels[~mlm_masked_indices] = -100

        generator_input_ids_for_model = original_input_ids.clone()

        # 80% of the time, replace with [MASK]
        indices_replaced_mask = torch.bernoulli(torch.full(
            original_input_ids.shape, 0.8, device=original_input_ids.device)).bool() & mlm_masked_indices
        generator_input_ids_for_model[indices_replaced_mask] = self.tokenizer.mask_token_id

        # 10% of the time, replace with a random token (the remaining 10% is original token, already handled by clone)
        indices_random_token = torch.bernoulli(torch.full(
            # 0.5 of the remaining 20% -> 10% of total
            original_input_ids.shape, 0.5, device=original_input_ids.device)).bool() & mlm_masked_indices & ~indices_replaced_mask
        random_words = torch.randint(len(
            self.tokenizer), original_input_ids.shape, dtype=torch.long, device=original_input_ids.device)
        generator_input_ids_for_model[indices_random_token] = random_words[indices_random_token]

        # The remaining 10% of mlm_masked_indices will keep their original tokens

        return {
            # For discriminator, and to create labels
            "original_input_ids": original_input_ids,
            "generator_input_ids": generator_input_ids_for_model,  # Input to generator
            "generator_labels": generator_labels,  # Labels for generator (MLM)
            # Boolean mask indicating which tokens were originally selected for MLM
            "mlm_masked_indices": mlm_masked_indices,
            "attention_mask": batch_attention_mask,
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

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 加载Generator和Discriminator模型
    # You might want to use smaller models for the generator for efficiency
    # For example, if discriminator is bert-base, generator could be bert-small
    generator = AutoModelForMaskedLM.from_pretrained(
        args.generator_model_name if args.generator_model_name else args.model_name)
    discriminator = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=2)  # 0 for original, 1 for replaced

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载领域语料
    texts = read_domain_corpus(args.train_file)
    logger.info(f"加载了 {len(texts)} 条语料")

    # 创建dataset
    dataset = DomainDataset(
        texts, tokenizer, max_length=args.max_length)

    # 创建Electra数据整理器
    data_collator = ElectraDataCollator(
        tokenizer=tokenizer,
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
    generator.to(device)
    discriminator.to(device)

    # 优化器
    # Apply weight decay to all parameters other than bias and LayerNorm weights
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in list(generator.named_parameters()) + list(discriminator.named_parameters()) if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in list(generator.named_parameters()) + list(discriminator.named_parameters()) if any(nd in n for nd in no_decay) and p.requires_grad],
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
    logger.info("开始ELECTRA训练 (Generator + Discriminator)")
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        epoch_total_loss = 0
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        for step, batch in enumerate(epoch_iterator):
            generator.train()
            discriminator.train()

            # 将数据移到设备上
            original_input_ids = batch['original_input_ids'].to(device)
            generator_input_ids = batch['generator_input_ids'].to(device)
            generator_labels = batch['generator_labels'].to(device)
            mlm_masked_indices = batch['mlm_masked_indices'].to(
                device)  # boolean mask
            attention_mask = batch['attention_mask'].to(device)

            # 1. Generator Forward Pass (MLM task)
            # The generator tries to predict the original tokens at the masked positions
            gen_outputs = generator(
                input_ids=generator_input_ids,
                attention_mask=attention_mask,
                labels=generator_labels  # Contains -100 for non-masked tokens
            )
            loss_gen = gen_outputs.loss
            # Shape: (batch_size, seq_length, vocab_size)
            gen_logits = gen_outputs.logits

            # 2. Construct Discriminator Input using Generator's samples
            # We take the original_input_ids and replace the MLM-selected tokens
            # with the tokens sampled from the generator's output.
            discriminator_input_ids = original_input_ids.clone()
            with torch.no_grad():  # Don't track gradients for this part
                # Sample tokens from the generator's output distribution
                # (greedy sampling via argmax)
                # Shape: (batch_size, seq_length)
                sampled_tokens = torch.argmax(gen_logits, dim=-1)

                # Replace original tokens with generator's samples ONLY at mlm_masked_indices
                # This creates the "fake" input for the discriminator
                discriminator_input_ids[mlm_masked_indices] = sampled_tokens[mlm_masked_indices]

                # Create labels for the discriminator: 1 if token was replaced, 0 if original
                # This is done by comparing the discriminator_input_ids with the original_input_ids.
                actual_discriminator_labels = (
                    discriminator_input_ids != original_input_ids).long()

            # 3. Discriminator Forward Pass & Loss
            # The discriminator predicts whether each token in discriminator_input_ids is "original" or "replaced"
            disc_outputs = discriminator(
                input_ids=discriminator_input_ids,
                attention_mask=attention_mask,
                # Shape: (batch_size, seq_length) with 0s and 1s
                labels=actual_discriminator_labels
            )
            loss_disc = disc_outputs.loss  # This should be a cross-entropy loss over all tokens

            # 4. Total Loss: sum of generator MLM loss and discriminator's replaced token detection loss
            # The discriminator loss is usually weighted by a factor (e.g., 50)
            total_loss = loss_gen + args.electra_lambda * loss_disc

            epoch_total_loss += total_loss.item()
            epoch_gen_loss += loss_gen.item()
            epoch_disc_loss += loss_disc.item()

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_iterator.set_postfix({
                "total_loss": total_loss.item(),
                "gen_loss": loss_gen.item(),
                "disc_loss": loss_disc.item()
            })

            # 保存中间模型检查点
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                gen_save_dir = os.path.join(checkpoint_dir, "generator")
                disc_save_dir = os.path.join(checkpoint_dir, "discriminator")
                if not os.path.exists(gen_save_dir):
                    os.makedirs(gen_save_dir)
                if not os.path.exists(disc_save_dir):
                    os.makedirs(disc_save_dir)

                generator.save_pretrained(gen_save_dir)
                discriminator.save_pretrained(disc_save_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"保存模型检查点到 {checkpoint_dir}")

        # 保存每个epoch的模型
        avg_epoch_total_loss = epoch_total_loss / len(dataloader)
        avg_epoch_gen_loss = epoch_gen_loss / len(dataloader)
        avg_epoch_disc_loss = epoch_disc_loss / len(dataloader)
        logger.info(
            f"Epoch {epoch+1} 平均损失: Total={avg_epoch_total_loss:.4f}, Gen={avg_epoch_gen_loss:.4f}, Disc={avg_epoch_disc_loss:.4f}")

        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        gen_save_dir = os.path.join(epoch_dir, "generator")
        disc_save_dir = os.path.join(epoch_dir, "discriminator")
        if not os.path.exists(gen_save_dir):
            os.makedirs(gen_save_dir)
        if not os.path.exists(disc_save_dir):
            os.makedirs(disc_save_dir)

        generator.save_pretrained(gen_save_dir)
        discriminator.save_pretrained(disc_save_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"保存epoch {epoch+1}的模型到 {epoch_dir}")

    # 保存最终模型
    final_gen_save_dir = os.path.join(args.output_dir, "generator")
    final_disc_save_dir = os.path.join(args.output_dir, "discriminator")
    if not os.path.exists(final_gen_save_dir):
        os.makedirs(final_gen_save_dir)
    if not os.path.exists(final_disc_save_dir):
        os.makedirs(final_disc_save_dir)

    generator.save_pretrained(final_gen_save_dir)
    discriminator.save_pretrained(final_disc_save_dir)
    # Save tokenizer in the main output dir
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"训练完成，最终模型保存到 {args.output_dir}")

    return args.output_dir


if __name__ == "__main__":
    class Args:
        # Used for discriminator and as default for generator
        model_name = "hfl/chinese-electra-180g-base-discriminator"
        # Can specify a different (e.g., smaller) model for generator
        generator_model_name = 'hfl/chinese-electra-180g-base-generator'
        train_file = "data/address.txt"
        output_dir = "pretrained/address_adapted_electra_model"
        max_length = 128
        batch_size = 16  # Adjust based on GPU memory
        learning_rate = 5e-5
        weight_decay = 0.01
        adam_epsilon = 1e-8
        max_grad_norm = 1.0
        num_train_epochs = 3  # ELECTRA might need more epochs
        warmup_steps = 0
        save_steps = 500  # Save checkpoint every N steps
        mask_probability = 0.15  # Standard for MLM
        electra_lambda = 50.0  # Weight for discriminator loss, often 50.0
        seed = 2025

    args = Args()

    import argparse
    parser = argparse.ArgumentParser(description="领域适配预训练 (ELECTRA)")
    parser.add_argument("--train_file", type=str,
                        default=args.train_file, help="领域语料文件")
    parser.add_argument("--output_dir", type=str,
                        default=args.output_dir, help="保存模型的目录")
    parser.add_argument("--model_name", type=str,
                        default=args.model_name, help="基础预训练模型名称 (用于Discriminator and default for Generator)")
    parser.add_argument("--generator_model_name", type=str,
                        default=args.generator_model_name, help="预训练模型名称 (用于Generator, if different from model_name)")
    parser.add_argument("--num_train_epochs", type=int,
                        default=args.num_train_epochs, help="训练轮数")
    parser.add_argument("--max_length", type=int,
                        default=args.max_length, help="Max sequence length")
    parser.add_argument("--batch_size", type=int,
                        default=args.batch_size, help="Batch size")
    parser.add_argument("--learning_rate", type=float,
                        default=args.learning_rate, help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=args.weight_decay, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float,
                        default=args.adam_epsilon, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float,
                        default=args.max_grad_norm, help="Max gradient norm")
    parser.add_argument("--warmup_steps", type=int,
                        default=args.warmup_steps, help="Warmup steps")
    parser.add_argument("--save_steps", type=int,
                        default=args.save_steps, help="Save checkpoint every N steps")
    parser.add_argument("--mask_probability", type=float,
                        default=args.mask_probability, help="Probability for masking tokens (MLM)")
    parser.add_argument("--electra_lambda", type=float,
                        default=args.electra_lambda, help="Weight for discriminator loss")
    parser.add_argument("--seed", type=int,
                        default=args.seed, help="Random seed")

    parsed_args = parser.parse_args()

    # Update Args class with parsed arguments
    args.train_file = parsed_args.train_file
    args.output_dir = parsed_args.output_dir
    args.model_name = parsed_args.model_name
    args.generator_model_name = parsed_args.generator_model_name if parsed_args.generator_model_name else args.model_name
    args.num_train_epochs = parsed_args.num_train_epochs
    args.max_length = parsed_args.max_length
    args.batch_size = parsed_args.batch_size
    args.learning_rate = parsed_args.learning_rate
    args.weight_decay = parsed_args.weight_decay
    args.adam_epsilon = parsed_args.adam_epsilon
    args.max_grad_norm = parsed_args.max_grad_norm
    args.warmup_steps = parsed_args.warmup_steps
    args.save_steps = parsed_args.save_steps
    args.mask_probability = parsed_args.mask_probability
    args.electra_lambda = parsed_args.electra_lambda
    args.seed = parsed_args.seed

    logger.info(f"使用ELECTRA方法进行领域适配预训练. Args: {vars(args)}")

    output_dir = train(args)
    logger.info(f"领域适配预训练完成，模型保存在 {output_dir}")
