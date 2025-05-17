import os
from sklearn.model_selection import KFold
import torch
from config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import logging

from conll_reader import ConllReader, MultiConllReader
from dataset import NERDataset, NERTestDataset
from model import PGD, AddressNER
from label import LabelMap

logger = logging.getLogger(__name__)


class StochasticWeightAveraging:
    def __init__(self, model, swa_start_epoch, swa_lr=None, swa_freq=5):
        """
        Implements Stochastic Weight Averaging (SWA)
        Args:
            model: The model to apply SWA to
            swa_start_epoch: The epoch to start averaging weights (0-indexed)
            swa_lr: The learning rate to use during SWA (currently not used for optimizer re-init)
            swa_freq: How frequently (in epochs) to update the SWA model
        """
        self.model_base = model  # Keep a reference to the original model structure for deepcopy
        self.swa_start_epoch = swa_start_epoch
        # Note: SWA LR is often handled by scheduler or fixed small LR during SWA phase
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        self.swa_model = None
        self.n_averaged = 0
        logger.info(
            f"SWA initialized: start_epoch={swa_start_epoch}, lr={swa_lr}, freq={swa_freq}")

    def update(self, epoch, model_current_state):
        """Update the SWA model by averaging with current model weights"""
        if epoch < self.swa_start_epoch:
            return

        if (epoch - self.swa_start_epoch) % self.swa_freq != 0:
            return

        logger.info(f"Updating SWA model at epoch {epoch}")
        if self.swa_model is None:
            # Create SWA model from base structure
            self.swa_model = copy.deepcopy(self.model_base)
            self.swa_model.load_state_dict(copy.deepcopy(model_current_state))
            logger.info("SWA model initialized with current model weights.")
        else:
            # Update running average of parameters
            current_params = dict(model_current_state)
            for name, swa_param in self.swa_model.named_parameters():
                if swa_param.requires_grad:
                    model_param = current_params[name]
                    swa_param.data.mul_(
                        self.n_averaged / (self.n_averaged + 1))
                    swa_param.data.add_(
                        model_param.data / (self.n_averaged + 1))
            logger.info(
                f"SWA model updated. n_averaged became {self.n_averaged + 1}")
        self.n_averaged += 1

    def get_final_model_state_dict(self):
        """Return the SWA model's state_dict with averaged weights"""
        if self.swa_model is None:
            logger.warning(
                "SWA was enabled, but no averaging was done. Returning None for SWA model state_dict.")
            return None
        logger.info("Final SWA model state_dict retrieved.")
        return self.swa_model.state_dict()


class Trainer:
    config: Config
    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    device: torch.device

    def __init__(self, config: Config, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, device: str):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.scheduler = None  # Initialize scheduler attribute

        # Initialize SWA if enabled
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(
                model=self.model,  # Pass the model instance for deepcopy base
                swa_start_epoch=self.config.swa_start_epoch,
                swa_lr=self.config.swa_lr,
                swa_freq=self.config.swa_freq
            )
        else:
            self.swa = None

        self.model.to(self.device)
        logger.info(
            f"Trainer initialized for training. Training on {self.device}. Work directory: {self.config.work_dir}")

    def train(self):
        self.model.train()  # Ensure model is in training mode

        # Optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.classifier.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.crf.parameters(
            ), 'lr': self.config.learning_rate * 50}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )

        best_val_metric = 0
        # Fold specific work_dir
        os.makedirs(self.config.work_dir, exist_ok=True)

        pgd = None
        if hasattr(self.config, 'adversarial_training_start_epoch') and self.config.adversarial_training_start_epoch >= 0:
            pgd = PGD(self.model)
            logger.info("PGD adversarial training is configured.")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train] ({os.path.basename(self.config.work_dir)})")

            for batch in train_pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                loss = self.model(input_ids, attention_mask, labels)
                loss.backward()

                if pgd and epoch >= self.config.adversarial_training_start_epoch:
                    pgd.attack(is_first_attack=True)
                    # loss_adv_accumulation = 0 # Not used
                    for _ in range(pgd.steps - 1):
                        optimizer.zero_grad()
                        loss_adv_step = self.model(
                            input_ids, attention_mask, labels)
                        loss_adv_step.backward()
                        pgd.attack()
                        # loss_adv_accumulation += loss_adv_step.item() # Not used

                    optimizer.zero_grad()
                    loss_adv_final = self.model(
                        input_ids, attention_mask, labels)
                    loss_adv_final.backward()
                    pgd.restore()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                train_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / \
                len(self.train_dataloader) if len(
                    self.train_dataloader) > 0 else 0
            logger.info(
                f"Epoch {epoch+1} ({os.path.basename(self.config.work_dir)}) avg training loss: {avg_train_loss:.4f}")

            # Evaluation
            self.model.eval()
            all_preds_eval = []
            all_labels_eval = []
            eval_loss = 0

            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Eval] ({os.path.basename(self.config.work_dir)})"):
                    input_ids_eval = batch["input_ids"].to(self.device)
                    attention_mask_eval = batch["attention_mask"].to(
                        self.device)
                    labels_eval = batch["labels"].to(self.device)

                    loss_eval_batch = self.model(
                        input_ids_eval, attention_mask_eval, labels_eval)
                    eval_loss += loss_eval_batch.item()

                    predictions_eval = self.model(
                        input_ids_eval, attention_mask_eval)

                    for pred_seq, mask_seq, label_seq in zip(predictions_eval, attention_mask_eval, labels_eval):
                        true_length = mask_seq.sum().item()
                        all_preds_eval.extend(pred_seq[:true_length])
                        all_labels_eval.extend(
                            label_seq[:true_length].cpu().numpy())

            avg_eval_loss = eval_loss / \
                len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
            correct_eval = sum(p == l for p, l in zip(
                all_preds_eval, all_labels_eval))
            total_eval = len(all_preds_eval) if len(
                all_preds_eval) > 0 else 1
            current_val_metric = correct_eval / total_eval
            logger.info(
                f"Epoch {epoch+1} ({os.path.basename(self.config.work_dir)}) Val Loss: {avg_eval_loss:.4f}, Val Acc: {current_val_metric:.4f}")

            if self.swa is not None:
                self.swa.update(epoch, self.model.state_dict())

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                saved_path = os.path.join(
                    self.config.work_dir, "best_model.pt")  # Saved in fold-specific dir
                torch.save(self.model.state_dict(), saved_path)
                logger.info(
                    f"Saved new best model ({os.path.basename(self.config.work_dir)}) with val metric: {current_val_metric:.4f} to {saved_path}")

        if self.swa is not None:
            swa_model_state_dict = self.swa.get_final_model_state_dict()
            if swa_model_state_dict is not None:
                swa_save_path = os.path.join(
                    self.config.work_dir, "swa_model.pt")  # Saved in fold-specific dir
                torch.save(swa_model_state_dict, swa_save_path)
                logger.info(
                    f"Saved final SWA model ({os.path.basename(self.config.work_dir)}) to {swa_save_path}")
        logger.info(
            f"Training finished for work directory: {self.config.work_dir}")


class KFoldTrainer:
    def __init__(self, config: Config):
        self.config = config

    def kfold_train(self):
        logger.info("--- Starting K-Fold Training Pipeline ---")
        folds_base_dir = os.path.join(
            self.config.work_dir, self.config.model_name)
        os.makedirs(folds_base_dir, exist_ok=True)
        logger.info(
            f"Base directory for K-Folds of model '{self.config.model_name}': {folds_base_dir}")

        multi_reader = MultiConllReader(
            [self.config.train_file, self.config.dev_file])
        full_data_conll = list(multi_reader.read())  # List of Sentence objects

        kf = KFold(n_splits=self.config.k_folds, shuffle=True,
                   random_state=self.config.seed)

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_data_conll)):
            fold_num = fold_idx + 1
            logger.info(
                f"--- Processing Fold {fold_num}/{self.config.k_folds} for model '{self.config.model_name}' ---")

            fold_train_data = [full_data_conll[i] for i in train_indices]
            fold_val_data = [full_data_conll[i] for i in val_indices]

            # Create a specific config for this fold
            # Start with a copy of the main config
            fold_cfg = copy.deepcopy(self.config)
            fold_cfg.work_dir = os.path.join(
                folds_base_dir, f"fold_{fold_num}")
            # model_name for AddressNER should be the adapted model path being k-folded
            fold_cfg.model_name = self.config.model_name
            os.makedirs(fold_cfg.work_dir, exist_ok=True)
            logger.info(
                f"Fold {fold_num} config: work_dir='{fold_cfg.work_dir}', model_name_for_tokenizer='{fold_cfg.model_name}'")

            # Instantiate model for this fold (uses fold_cfg.model_name for tokenizer)
            model = AddressNER(num_labels=len(
                fold_cfg.label_map.labels), config=fold_cfg)

            # Create datasets and dataloaders for this fold
            train_dataset = NERDataset(
                fold_train_data, model.tokenizer, fold_cfg.label_map.label2id)
            val_dataset = NERDataset(
                fold_val_data, model.tokenizer, fold_cfg.label_map.label2id)
            train_loader = DataLoader(
                train_dataset, batch_size=fold_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=fold_cfg.batch_size,
                                    shuffle=False, num_workers=4, pin_memory=True)

            # Instantiate and run trainer for this fold
            trainer_fold = Trainer(config=fold_cfg,
                                   model=model,
                                   train_dataloader=train_loader,
                                   val_dataloader=val_loader,
                                   device=fold_cfg.device)
            trainer_fold.train()  # This will save best_model.pt and swa_model.pt in fold_cfg.work_dir
        logger.info("--- K-Fold Training Pipeline Finished ---")
