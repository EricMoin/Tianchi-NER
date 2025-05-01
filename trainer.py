import os
import torch
from config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)

        self.model.to(self.device)
        print(f"Training on {self.device}")

    def train(self):
        self.model.train()
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 5e-4},
            {'params': self.model.classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.crf.parameters(), 'lr': 1e-3}
        ])

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0, total_iters=total_steps)
        best_f1 = 0

        # Initialize FGM
        # fgm = FGM(self.model)
        pgd = PGD(self.model)

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]")

            for batch in train_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                loss = self.model(input_ids, attention_mask, labels)

                # Backward pass
                loss.backward()

                # PGD adversarial training
                pgd.attack(is_first_attack=True)  # 初始扰动

                for _ in range(pgd.steps - 1):
                    optimizer.zero_grad()
                    loss_adv = self.model(input_ids, attention_mask, labels)
                    loss_adv.backward()  # 反向传播，计算梯度
                    pgd.attack()  # 多步更新扰动

                loss_adv = self.model(input_ids, attention_mask, labels)
                loss_adv.backward()  # 计算最终的对抗梯度

                pgd.restore()  # 恢复embedding参数

                optimizer.step()
                self.scheduler.step()

                train_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(self.train_dataloader)
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

            # Save model if it's the best so far
            if accuracy > best_f1:
                best_f1 = accuracy
                saved_path = os.path.join(
                    self.config.work_dir, f"best_model.pt")
                os.makedirs(self.config.work_dir, exist_ok=True)
                torch.save(self.model.state_dict(), saved_path)
                print(f"Saved new best model with accuracy: {accuracy:.4f}")

            # Save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': avg_train_loss,
            }, f"{self.config.work_dir}/checkpoint_epoch_{epoch+1}.pt")

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
