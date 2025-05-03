import os
import torch
from config import Config
from torch import nn
import torch.distributed as dist  # 正确导入分布式模块
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # 添加自动混合精度和梯度缩放

from conll_reader import ConllReader
from dataset import NERDataset, NERTestDataset
from model import FGM, PGD  # Import FGM class


class Trainer:
    config: Config
    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: torch.device
    id2label: dict
    local_rank: int
    is_main_process: bool

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str, local_rank=-1):
        self.config = config
        self.local_rank = local_rank

        if local_rank != -1:
            # Initialize process group
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            self.model = model.to(self.device)
            self.model = DDP(self.model,
                             device_ids=[local_rank],
                             output_device=local_rank,
                             find_unused_parameters=True)
            self.is_main_process = (local_rank == 0)
        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.is_main_process = True

        # Re-create dataloaders with distributed samplers
        if hasattr(train_dataloader.dataset, 'dataset'):
            train_sampler = DistributedSampler(
                train_dataloader.dataset) if local_rank != -1 else None
            val_sampler = DistributedSampler(
                val_dataloader.dataset, shuffle=False) if local_rank != -1 else None

            self.train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=train_dataloader.num_workers
            )

            self.val_dataloader = DataLoader(
                val_dataloader.dataset,
                batch_size=config.batch_size,
                sampler=val_sampler,
                num_workers=val_dataloader.num_workers
            )
        else:
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

    def train(self):
        # Get model module if it's wrapped with DDP
        model_module = self.model.module if hasattr(
            self.model, 'module') else self.model

        optimizer = torch.optim.AdamW([
            {'params': model_module.bert.embeddings.parameters(), 'lr': 1e-4},
            {'params': model_module.lstm.parameters(), 'lr': 5e-4},
            {'params': model_module.classifier.parameters(), 'lr': 5e-4},
            {'params': model_module.crf.parameters(), 'lr': 1e-3}
        ])

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0, total_iters=total_steps)
        best_f1 = 0

        # 初始化梯度缩放器
        scaler = GradScaler()

        # Initialize adversarial training
        pgd = PGD(model_module)

        for epoch in range(self.config.num_epochs):
            # Set epoch for sampler
            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            # Training
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]",
                disable=not self.is_main_process
            )

            for batch in train_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 1. 正常前向和反向传播
                optimizer.zero_grad()
                with autocast(device_type=self.device.type):
                    loss = self.model(input_ids, attention_mask, labels)
                scaler.scale(loss).backward()

                # 2. PGD对抗训练
                pgd.attack(is_first_attack=True)  # 初始扰动

                for _ in range(pgd.steps - 1):
                    optimizer.zero_grad()
                    with autocast(device_type=self.device.type):
                        loss_adv = self.model(
                            input_ids, attention_mask, labels)
                    scaler.scale(loss_adv).backward()
                    pgd.attack()

                with autocast(device_type=self.device.type):
                    loss_adv = self.model(input_ids, attention_mask, labels)
                scaler.scale(loss_adv).backward()

                pgd.restore()  # 恢复embedding参数

                # 3. 一次性更新所有参数
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            self.scheduler.step()  # 确保在optimizer.step()之后调用
            # Average loss across processes
            if self.local_rank != -1:
                torch.distributed.all_reduce(
                    torch.tensor(train_loss).to(self.device))
                train_loss = train_loss / dist.get_world_size()

            avg_train_loss = train_loss / len(self.train_dataloader)
            if self.is_main_process:
                print(f"Average training loss: {avg_train_loss:.4f}")

            # Evaluate on training set
            self.model.eval()
            train_preds = []
            train_labels_list = []

            with torch.no_grad():
                for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train Eval]"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Get predictions
                    predictions = self.model(input_ids, attention_mask)

                    # Convert predictions and labels to lists
                    for pred, mask, label in zip(predictions, attention_mask, labels):
                        length = mask.sum().item()
                        train_preds.extend(pred[:length])
                        train_labels_list.extend(label[:length].cpu().numpy())

            # Calculate training metrics
            train_correct = sum(p == l for p, l in zip(
                train_preds, train_labels_list))
            train_total = len(train_preds)
            train_accuracy = train_correct / train_total
            print(f"Training Accuracy: {train_accuracy:.4f}")

            # Evaluation on validation set
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Eval]"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Get predictions
                    predictions = self.model(input_ids, attention_mask)

                    # Convert predictions and labels to lists
                    for pred, mask, label in zip(predictions, attention_mask, labels):
                        length = mask.sum().item()
                        all_preds.extend(pred[:length])
                        all_labels.extend(label[:length].cpu().numpy())

            # Calculate validation metrics
            correct = sum(p == l for p, l in zip(all_preds, all_labels))
            total = len(all_preds)
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")

            # Save model only from main process
            if self.is_main_process:
                if accuracy > best_f1:
                    best_f1 = accuracy
                    saved_path = os.path.join(
                        self.config.work_dir, f"best_model.pt")
                    os.makedirs(self.config.work_dir, exist_ok=True)
                    torch.save(model_module.state_dict(), saved_path)
                    print(
                        f"Saved new best model with accuracy: {accuracy:.4f}")

                # Save the last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': avg_train_loss,
                }, f"{self.config.work_dir}/checkpoint_epoch_{epoch+1}.pt")

        # Cleanup distributed processes
        if self.local_rank != -1:
            dist.destroy_process_group()

    def test(self):
        # Load the best model for inference
        self.model.load_state_dict(torch.load(
            f"{self.config.work_dir}/best_model.pt"))

        # Convert test data to CoNLL format with O labels
        temp_conll_file = f"{self.config.work_dir}/temp_test.conll"

        # Create temporary CoNLL file from test data
        with open(self.config.test_file, 'r', encoding='utf-8') as f_in, \
                open(temp_conll_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if line:
                    # Remove the line number prefix (e.g., "1朝阳区..." -> "朝阳区...")
                    text = line.split('\u0001')[1]
                    # Write each character with O label in CoNLL format (no space)
                    for char in text:
                        if char == '\u0001':
                            print("HERE")
                        f_out.write(f"{char} O\n")
                    f_out.write("\n")  # Empty line between examples

        # Use standard ConllReader to read the temporary file
        test_reader = ConllReader(temp_conll_file)
        test_conll = list(test_reader.read())

        # Create dataset using the ConllReader output
        test_dataset = NERDataset(
            test_conll, self.model.tokenizer, self.config.label2id)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.batch_size)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Predicting test data"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get predictions
                predictions = self.model(input_ids, attention_mask)

                # Process predictions for each example
                for pred, mask in zip(predictions, attention_mask):
                    length = mask.sum().item()
                    pred_labels = [self.config.id2label[p]
                                   for p in pred[:length]]
                    all_preds.append(pred_labels)

        # Write predictions to file
        with open(f"{self.config.work_dir}/pred.txt", "w", encoding="utf8") as f:
            for i, example in enumerate(test_conll):
                for j, token in enumerate(example.tokens):
                    if j < len(all_preds[i]):
                        f.write(f"{token}\t{all_preds[i][j]}\n")
                    else:
                        f.write(f"{token}\tO\n")
                f.write("\n")

        print(f"Predictions saved to {self.config.work_dir}/pred.txt")
