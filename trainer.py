import os
import torch
from config import Config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import copy
import logging

from conll_reader import ConllReader
from dataset import NERDataset, NERTestDataset
from model import PGD

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
    id2label: dict

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
            f"Trainer initialized. Training on {self.device}. Work directory: {self.config.work_dir}")

    def train(self):
        self.model.train()  # Ensure model is in training mode

        # Optimizer: Consider making parameters more configurable via self.config
        optimizer = torch.optim.AdamW([
            {'params': self.model.bert.embeddings.parameters(
            ), 'lr': self.config.learning_rate * 5},  # Example: higher LR for embeddings
            {'params': self.model.lstm.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.classifier.parameters(
            ), 'lr': self.config.learning_rate * 25},
            {'params': self.model.crf.parameters(
            ), 'lr': self.config.learning_rate * 50}
        ], lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        if total_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                # End factor 0.1, not 0
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        else:
            logger.warning(
                "Total training steps is 0. No scheduler will be used. Check data loader.")
            self.scheduler = None

        best_val_metric = 0  # Assuming higher is better (e.g., F1 or accuracy)
        os.makedirs(self.config.work_dir, exist_ok=True)

        pgd = None
        if self.config.adversarial_training_start_epoch >= 0:  # Enable PGD if start epoch is non-negative
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
                    loss_adv_accumulation = 0
                    for _ in range(pgd.steps - 1):  # K-1 steps for PGD
                        optimizer.zero_grad()
                        loss_adv_step = self.model(
                            input_ids, attention_mask, labels)
                        loss_adv_step.backward()  # Accumulate gradients on perturbed input
                        pgd.attack()  # Update perturbation based on new gradients
                        loss_adv_accumulation += loss_adv_step.item()

                    # Final adversarial step loss calculation and gradient update
                    optimizer.zero_grad()
                    loss_adv_final = self.model(
                        input_ids, attention_mask, labels)
                    loss_adv_final.backward()
                    pgd.restore()  # Restore original embeddings before optimizer step

                # Gradient clipping (optional but good practice)
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
                f"Epoch {epoch+1} avg training loss: {avg_train_loss:.4f}")

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
                        input_ids_eval, attention_mask_eval)  # Get decoded sequences

                    for pred_seq, mask_seq, label_seq in zip(predictions_eval, attention_mask_eval, labels_eval):
                        true_length = mask_seq.sum().item()
                        # pred_seq is already list of ints
                        all_preds_eval.extend(pred_seq[:true_length])
                        all_labels_eval.extend(
                            label_seq[:true_length].cpu().numpy())

            avg_eval_loss = eval_loss / \
                len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
            # Calculate metrics (e.g., accuracy, F1)
            # For simplicity, using accuracy here. Replace with proper NER metrics (seqeval F1)
            correct_eval = sum(p == l for p, l in zip(
                all_preds_eval, all_labels_eval))
            total_eval = len(all_preds_eval) if len(
                all_preds_eval) > 0 else 1  # Avoid division by zero
            current_val_metric = correct_eval / total_eval
            logger.info(
                f"Epoch {epoch+1} Validation Loss: {avg_eval_loss:.4f}, Validation Accuracy: {current_val_metric:.4f}")

            if self.swa is not None:
                # Pass state_dict
                self.swa.update(epoch, self.model.state_dict())

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                saved_path = os.path.join(
                    self.config.work_dir, "best_model.pt")
                torch.save(self.model.state_dict(), saved_path)
                logger.info(
                    f"Saved new best model with validation metric: {current_val_metric:.4f} to {saved_path}")

            # Save checkpoint at end of epoch (optional)
            checkpoint_path = os.path.join(
                self.config.work_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'loss': avg_train_loss,
                'val_metric': current_val_metric
            }, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

        # Save the final SWA model if enabled and averaging was done
        if self.swa is not None:
            swa_model_state_dict = self.swa.get_final_model_state_dict()
            if swa_model_state_dict is not None:
                swa_save_path = os.path.join(
                    self.config.work_dir, "swa_model.pt")
                torch.save(swa_model_state_dict, swa_save_path)
                logger.info(f"Saved final SWA model to {swa_save_path}")
        logger.info(
            f"Training finished for work directory: {self.config.work_dir}")

    def get_predictions_on_test_set(self):
        """
        Runs predictions on the test set specified in config and returns them.
        Loads the best or SWA model based on availability from self.config.work_dir.
        Returns: list of lists of predicted label strings for each test example.
        """
        model_to_load = None
        swa_model_path = os.path.join(self.config.work_dir, "swa_model.pt")
        best_model_path = os.path.join(self.config.work_dir, "best_model.pt")

        if self.config.use_swa and self.swa is not None and os.path.exists(swa_model_path):
            logger.info(
                f"Using SWA model for inference from: {swa_model_path}")
            model_to_load = swa_model_path
        elif os.path.exists(best_model_path):
            logger.info(
                f"Using best_model.pt for inference from: {best_model_path}")
            model_to_load = best_model_path
        else:
            logger.error(
                f"No SWA or best model found in {self.config.work_dir}. Cannot generate predictions.")
            # Fallback: try loading last checkpoint if available?
            # For now, return empty if no primary model found.
            return []

        try:
            self.model.load_state_dict(torch.load(
                model_to_load, map_location=self.device))
        except Exception as e:
            logger.error(
                f"Error loading model state_dict from {model_to_load}: {e}")
            return []

        self.model.to(self.device)
        self.model.eval()

        # Prepare test data (similar to original test method but without file writing here)
        # This part needs the original character sequences from the test file.
        # config.test_file should be the raw test file (e.g. final_test.txt)

        # Create CoNLL-like structure for test data (tokens only)
        test_sentences_tokens = []  # List of lists of character tokens
        if not os.path.exists(self.config.test_file):
            logger.error(f"Test file not found at: {self.config.test_file}")
            return []

        with open(self.config.test_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    try:
                        text_part = line.split('\u0001')[1]
                        test_sentences_tokens.append(list(text_part))
                    except IndexError:
                        logger.warning(
                            f"Skipping malformed line in test file during prediction: {line}")
                        # Add empty list to maintain example count if needed
                        test_sentences_tokens.append([])

        if not test_sentences_tokens:
            logger.warning("No tokens extracted from test file.")
            return []

        # Create NERDataset for test (it handles tokenization and label mapping to 'O' if no labels)
        # We need a temporary ConllReader-like input for NERDataset
        # NERDataset expects list of Sentence objects. We have list of list of tokens.
        class TempSentence:  # Mimic ConllReader's Sentence object for NERDataset
            def __init__(self, tokens):
                self.tokens = tokens
                # Dummy labels for dataset creation
                self.labels = ['O'] * len(tokens)

        test_conll_examples = [TempSentence(tokens)
                               for tokens in test_sentences_tokens]

        test_dataset = NERDataset(
            test_conll_examples, self.model.tokenizer, self.config.label2id)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False)

        all_preds_sequences = []  # List of lists of label strings
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Predicting test data ({os.path.basename(self.config.work_dir)})"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                batch_pred_indices = self.model(
                    input_ids, attention_mask)  # List[List[int]] from CRF

                for i in range(len(batch_pred_indices)):
                    pred_indices_for_example = batch_pred_indices[i]
                    pred_labels = [self.config.id2label.get(
                        p_idx, 'O') for p_idx in pred_indices_for_example]
                    all_preds_sequences.append(pred_labels)

        logger.info(
            f"Generated {len(all_preds_sequences)} prediction sequences for test set.")
        return all_preds_sequences

    def test(self):  # Original test method, now uses get_predictions_on_test_set
        logger.info(
            f"Executing standard test method for {self.config.work_dir}...")

        all_preds_sequences = self.get_predictions_on_test_set()

        if not all_preds_sequences:
            logger.error("No predictions generated. Cannot write output file.")
            return

        # To write output in CoNLL like format, we need original tokens per example.
        # Re-read test file to get original token structure aligned with predictions.
        test_sentences_orig_tokens = []
        with open(self.config.test_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    try:
                        text_part = line.split('\u0001')[1]
                        test_sentences_orig_tokens.append(list(text_part))
                    except IndexError:
                        logger.warning(
                            f"Skipping malformed line in test file for final output: {line}")
                        test_sentences_orig_tokens.append([])

        output_pred_path = os.path.join(self.config.work_dir, "pred.txt")
        with open(output_pred_path, "w", encoding="utf8") as f:
            for i, tokens_for_example in enumerate(test_sentences_orig_tokens):
                if i < len(all_preds_sequences):
                    pred_labels_for_example = all_preds_sequences[i]
                    for j, token_text in enumerate(tokens_for_example):
                        label_to_write = 'O'
                        if j < len(pred_labels_for_example):
                            label_to_write = pred_labels_for_example[j]
                        f.write(f"{token_text}\t{label_to_write}\n")
                    f.write("\n")
                else:
                    logger.warning(
                        f"Missing prediction for example index {i} when writing to {output_pred_path}. Writing 'O' labels.")
                    for token_text in tokens_for_example:
                        f.write(f"{token_text}\tO\n")
                    f.write("\n")

        logger.info(
            f"Standard test method predictions saved to {output_pred_path}")
