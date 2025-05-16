import os
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter
import torch
import random
import logging
import shutil  # For cleaning up fold directories if needed

from conll_reader import ConllReader
from dataset import NERDataset
from model import AddressNER
from config import Config
from torch.utils.data import DataLoader
from trainer import Trainer
from pretrained import run_domain_adaptation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Seed set to {seed_value}")


def get_label_maps():
    label_list = [
        'B-prov', 'I-prov', 'E-prov', 'S-prov',
        'B-city', 'I-city', 'E-city', 'S-city',
        'B-district', 'I-district', 'E-district', 'S-district',
        'B-devzone', 'I-devzone', 'E-devzone', 'S-devzone',
        'B-town', 'I-town', 'E-town', 'S-town',
        'B-community', 'I-community', 'E-community', 'S-community',
        'B-village_group', 'I-village_group', 'E-village_group', 'S-village_group',
        'B-road', 'I-road', 'E-road', 'S-road',
        'B-roadno', 'I-roadno', 'E-roadno', 'S-roadno',
        'B-poi', 'I-poi', 'E-poi', 'S-poi',
        'B-subpoi', 'I-subpoi', 'E-subpoi', 'S-subpoi',
        'B-houseno', 'I-houseno', 'E-houseno', 'S-houseno',
        'B-cellno', 'I-cellno', 'E-cellno', 'S-cellno',
        'B-floorno', 'I-floorno', 'E-floorno', 'S-floorno',
        'B-roomno', 'I-roomno', 'E-roomno', 'S-roomno',
        'B-detail', 'I-detail', 'E-detail', 'S-detail',
        'B-assist', 'I-assist', 'E-assist', 'S-assist',
        'B-distance', 'I-distance', 'E-distance', 'S-distance',
        'B-intersection', 'I-intersection', 'E-intersection', 'S-intersection',
        'B-redundant', 'I-redundant', 'E-redundant', 'S-redundant',
        'B-others', 'I-others', 'E-others', 'S-others',
        'O'
    ]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    logger.info(f"Generated label maps with {len(label_list)} labels.")
    return label_list, id2label, label2id


def load_and_combine_data(config: Config) -> list:  # list of Sentence objects
    logger.info(f"Loading training data from: {config.train_file}")
    full_train_data_conll = []
    if os.path.exists(config.train_file):
        train_reader = ConllReader(config.train_file)
        full_train_data_conll.extend(list(train_reader.read()))
    else:
        logger.warning(f"Training file {config.train_file} not found.")

    if config.dev_file and os.path.exists(config.dev_file):
        logger.info(f"Loading dev data from: {config.dev_file} and appending.")
        dev_reader = ConllReader(config.dev_file)
        full_train_data_conll.extend(list(dev_reader.read()))
    elif config.dev_file:
        logger.warning(f"Dev file {config.dev_file} specified but not found.")

    logger.info(f"Total examples for K-fold CV: {len(full_train_data_conll)}")
    return full_train_data_conll


def get_raw_test_token_sequences(test_file_path: str) -> list[list[str]]:
    """Reads the raw test file and returns a list of token lists for each example."""
    test_sentences_tokens = []
    if not os.path.exists(test_file_path):
        logger.error(
            f"Test file for token extraction not found at: {test_file_path}")
        return []
    with open(test_file_path, 'r', encoding='utf-8') as f_in:
        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if line:
                parts = line.split('\u0001', 1)
                if len(parts) == 2:
                    test_sentences_tokens.append(list(parts[1]))
                else:
                    logger.warning(
                        f"Malformed line {line_idx+1} in test file '{test_file_path}': {line}")
                    test_sentences_tokens.append([])
    logger.info(
        f"Extracted {len(test_sentences_tokens)} token sequences from {test_file_path}")
    return test_sentences_tokens


def write_ensembled_predictions_conll_format(
    output_path: str,
    original_char_sequences: list[list[str]],
    ensembled_labels: list[list[str]]
):
    logger.info(
        f"Writing ensembled predictions to CoNLL format: {output_path}")
    if len(original_char_sequences) != len(ensembled_labels):
        logger.error(
            f"Mismatch between number of original char sequences ({len(original_char_sequences)}) "
            f"and ensembled label sequences ({len(ensembled_labels)}). Cannot write CoNLL output."
        )
        # Optionally, could raise an error or return a status
        return False  # Indicate failure

    try:
        with open(output_path, "w", encoding="utf8") as f:
            for i, chars in enumerate(original_char_sequences):
                labels_for_example = ensembled_labels[i]
                if len(chars) != len(labels_for_example) and chars:  # Check if chars is not empty
                    logger.warning(
                        f"Example {i}: Token/label length mismatch. Chars: {len(chars)}, Labels: {len(labels_for_example)}. "
                        f"Aligning by truncating the longer sequence for this example."
                    )

                # Iterate up to the minimum of the two lengths to prevent IndexError
                min_len = min(len(chars), len(labels_for_example))
                for j in range(min_len):
                    char_token = chars[j]
                    label_to_write = labels_for_example[j]
                    f.write(f"{char_token}\t{label_to_write}\n")

                # If original chars are more than labels, fill remaining with 'O'
                if len(chars) > min_len:
                    for j in range(min_len, len(chars)):
                        f.write(f"{chars[j]}\tO\n")
                # If labels are more than original chars (less likely but possible if logic error upstream)
                # This part is usually not needed if original_char_sequences drives the length.

                f.write("\n")  # Sentence separator
        logger.info(
            f"Ensembled predictions in CoNLL format written to {output_path}")
        return True  # Indicate success
    except IOError as e:
        logger.error(f"Failed to write CoNLL output to {output_path}: {e}")
        return False  # Indicate failure


def process_final_submission_output(ensembled_conll_pred_file: str, final_submission_file: str, original_test_file_path: str):
    logger.info(
        f"Processing {ensembled_conll_pred_file} into final submission format {final_submission_file}")

    original_test_data = []  # List of (guid, original_text_string)
    if not os.path.exists(original_test_file_path):
        logger.error(
            f"Original test file {original_test_file_path} not found. Cannot create submission.")
        return
    with open(original_test_file_path, 'r', encoding='utf-8') as f_orig:
        for line_idx, line in enumerate(f_orig):
            line = line.strip()
            if line:
                parts = line.split('\u0001', 1)  # guid<SEP>text
                if len(parts) == 2:
                    original_test_data.append((parts[0], parts[1]))
                else:
                    logger.warning(
                        f"Malformed line {line_idx+1} in original test file {original_test_file_path}: '{line}'. Skipping.")

    # List of [list of labels for example1, list of labels for example2, ...]
    predicted_labels_per_example = []
    current_example_labels = []
    if not os.path.exists(ensembled_conll_pred_file):
        logger.error(
            f"Ensembled CoNLL prediction file {ensembled_conll_pred_file} not found. Cannot create submission.")
        return

    with open(ensembled_conll_pred_file, 'r', encoding='utf-8') as f_conll:
        for line_conll in f_conll:
            line_conll = line_conll.strip()
            if not line_conll:  # Sentence boundary
                if current_example_labels:
                    predicted_labels_per_example.append(current_example_labels)
                    current_example_labels = []
            else:
                parts = line_conll.split('\t')
                if len(parts) == 2:
                    current_example_labels.append(parts[1])  # Store the label
                else:
                    logger.warning(
                        f"Malformed line in CoNLL pred file '{ensembled_conll_pred_file}': '{line_conll}'. Ignoring token.")
        if current_example_labels:  # Add last example if file doesn't end with newline
            predicted_labels_per_example.append(current_example_labels)

    if len(original_test_data) != len(predicted_labels_per_example):
        logger.error(
            f"Number of examples mismatch! Original test data examples: {len(original_test_data)}, "
            f"Predicted examples from CoNLL: {len(predicted_labels_per_example)}. Submission file may be incorrect or incomplete."
        )
        # This is a critical error; the output will likely be misaligned.
        # Consider the shorter length to prevent crashing, though the output will be partial.
        num_examples_to_process = min(
            len(original_test_data), len(predicted_labels_per_example))
    else:
        num_examples_to_process = len(original_test_data)

    try:
        with open(final_submission_file, 'w', encoding='utf8') as fout:
            written_count = 0
            for i in range(num_examples_to_process):
                guid, original_text = original_test_data[i]
                labels_for_this_example = predicted_labels_per_example[i]

                # Critical Sanity Check: Compare length of original text characters with number of predicted labels
                original_text_char_list = list(original_text)
                if len(original_text_char_list) != len(labels_for_this_example):
                    logger.warning(
                        f"GUID {guid} (Example {i}): Length mismatch! Original text has {len(original_text_char_list)} chars, "
                        f"but {len(labels_for_this_example)} labels were predicted. "
                        f"Labels: [{' '.join(labels_for_this_example[:20])}...]. Text: '{original_text[:20]}...'."
                        f"The output for this line will use the number of predicted labels, which might be incorrect if they don't align with original characters."
                    )
                    # Option 1: Pad/truncate labels to match original text length (safer for format, riskier for meaning)
                    # final_labels_list = (labels_for_this_example + ['O'] * len(original_text_char_list))[:len(original_text_char_list)]
                    # Option 2: Use labels as is (what's currently done, might lead to format error if checker is strict on space separation)
                    final_labels_list = labels_for_this_example
                else:
                    final_labels_list = labels_for_this_example

                labels_str = ' '.join(final_labels_list)
                fout.write(f"{guid}\u0001{original_text}\u0001{labels_str}\n")
                written_count += 1
            logger.info(
                f"Wrote {written_count} lines to final submission file: {final_submission_file}")
    except IOError as e:
        logger.error(
            f"Failed to write final submission file to {final_submission_file}: {e}")


def main():
    label_list, id2label, label2id = get_label_maps()

    cfg = Config(
        train_file='data/train.conll',
        dev_file='data/dev.conll',
        test_file='data/final_test.txt',  # Raw test file (e.g., idtext)
        output_file='result/ensembled_submission.txt',  # Final submission file
        model_name='_placeholder_',
        batch_size=16,
        num_epochs=2,  # Reduced for testing
        learning_rate=2e-5,
        weight_decay=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        work_dir='result',  # Base work directory
        freeze_bert_layers=0,
        num_prefix_tokens=0,  # Set to 0 if not using prefix tuning.
        label2id=label2id,
        id2label=id2label,
        adversarial_training_start_epoch=0,
        crf_transition_penalty=0.175,
        focal_loss_alpha=0.25,
        focal_loss_gamma=1.5,
        hybrid_loss_weight_crf=0.5,
        hybrid_loss_weight_focal=0.5,
        spatial_dropout=0.15,
        embedding_dropout=0.15,
        use_swa=True,
        swa_start_epoch=0,
        swa_lr=1e-5,
        swa_freq=1,
        seed=2024,
        k_folds=5,  # Reduced for testing
        base_model_names=["hfl/chinese-roberta-wwm-ext",
                          "hfl/chinese-bert-wwm-ext",],
        adaptation_corpus_file="data/address.txt",
        adaptation_num_epochs=3,  # Reduced for testing
        adaptation_batch_size=16
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.work_dir, exist_ok=True)
    logger.info(f"Initial base configuration: {vars(cfg)}")

    # --- 1. Domain Adaptation ---
    cfg.adapted_model_paths = []
    if not cfg.base_model_names:
        logger.error("No base_model_names for adaptation. Exiting.")
        return

    for base_model_hf_name in cfg.base_model_names:
        logger.info(f"Processing for adaptation: {base_model_hf_name}")
        sanitized_name = base_model_hf_name.replace('/', '_')
        adapted_model_dir = os.path.join(
            "pretrained", f"{sanitized_name}_adapted_ep{cfg.adaptation_num_epochs}_seed{cfg.seed}")

        if os.path.exists(os.path.join(adapted_model_dir, "model.safetensors")):
            logger.info(f"Found existing adapted model: {adapted_model_dir}")
            cfg.adapted_model_paths.append(adapted_model_dir)
        else:
            logger.info(
                f"Starting domain adaptation: {base_model_hf_name} -> {adapted_model_dir}")
            if not os.path.exists(cfg.adaptation_corpus_file):
                logger.error(
                    f"Adaptation corpus {cfg.adaptation_corpus_file} not found! Skipping {base_model_hf_name}")
                continue
            os.makedirs(adapted_model_dir, exist_ok=True)
            adapted_path = run_domain_adaptation(
                model_name_or_path=base_model_hf_name,
                train_file=cfg.adaptation_corpus_file,
                output_dir=adapted_model_dir,
                max_length=cfg.adaptation_max_length,
                batch_size=cfg.adaptation_batch_size,
                learning_rate=cfg.adaptation_lr,
                weight_decay=cfg.adaptation_weight_decay,
                adam_epsilon=cfg.adaptation_adam_epsilon,
                max_grad_norm=cfg.adaptation_max_grad_norm,
                num_train_epochs=cfg.adaptation_num_epochs,
                warmup_steps=cfg.adaptation_warmup_steps,
                mask_probability=cfg.adaptation_mask_probability,
                seed=cfg.seed
            )
            if adapted_path and os.path.exists(os.path.join(adapted_path, "model.safetensors")):
                cfg.adapted_model_paths.append(adapted_path)
            else:
                logger.error(f"Adaptation failed for {base_model_hf_name}.")

    if not cfg.adapted_model_paths:
        logger.error("No models available after adaptation phase. Exiting.")
        return
    logger.info(f"Adapted models for K-fold: {cfg.adapted_model_paths}")

    # --- 2. K-Fold Cross-Validation ---
    full_train_val_data = load_and_combine_data(cfg)
    if not full_train_val_data:
        logger.error("No training/validation data loaded. Exiting.")
        return

    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    # Stores predictions from each fold of each adapted model
    all_test_predictions_sources = []

    for adapted_model_path in cfg.adapted_model_paths:
        sanitized_adapted_name = os.path.basename(adapted_model_path)
        logger.info(
            f"--- Starting K-fold CV for adapted model: {adapted_model_path} ---")
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_train_val_data)):
            logger.info(
                f"--- Fold {fold_idx + 1}/{cfg.k_folds} (Model: {sanitized_adapted_name}) ---")

            fold_train_data = [full_train_val_data[i] for i in train_indices]
            fold_val_data = [full_train_val_data[i] for i in val_indices]

            fold_config = Config(**vars(cfg))  # Create a fresh config copy
            fold_config.work_dir = os.path.join(
                cfg.work_dir, sanitized_adapted_name, f"fold_{fold_idx+1}")
            # Use current adapted model for this fold
            fold_config.model_name = adapted_model_path
            os.makedirs(fold_config.work_dir, exist_ok=True)

            logger.info(
                f"Fold config using model: {fold_config.model_name}, work_dir: {fold_config.work_dir}")

            model = AddressNER(num_labels=len(label_list), config=fold_config)
            train_dataset = NERDataset(
                fold_train_data, model.tokenizer, fold_config.label2id)
            val_dataset = NERDataset(
                fold_val_data, model.tokenizer, fold_config.label2id)
            train_loader = DataLoader(
                train_dataset, batch_size=fold_config.batch_size, shuffle=True)
            val_loader = DataLoader(
                val_dataset, batch_size=fold_config.batch_size, shuffle=False)

            if not train_loader or not val_loader or len(train_dataset) == 0 or len(val_dataset) == 0:
                logger.error(
                    f"Skipping fold {fold_idx+1} due to empty data. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
                continue

            trainer = Trainer(config=fold_config, model=model, train_dataloader=train_loader,
                              val_dataloader=val_loader, device=fold_config.device)
            trainer.train()

            # List[List[str_labels]]
            fold_test_preds = trainer.get_predictions_on_test_set()
            if fold_test_preds:
                all_test_predictions_sources.append(fold_test_preds)
                logger.info(
                    f"Collected {len(fold_test_preds)} test predictions from fold {fold_idx+1} of {sanitized_adapted_name}.")
            else:
                logger.warning(
                    f"No test predictions from fold {fold_idx+1} of {sanitized_adapted_name}.")

    # --- 3. Ensemble Voting --- #
    if not all_test_predictions_sources:
        logger.error(
            "No test predictions collected from any fold/model. Cannot perform ensemble. Exiting.")
        return

    logger.info(
        f"Collected predictions from {len(all_test_predictions_sources)} sources (model/fold combinations).")

    # Get original character sequences from the raw test file to determine sequence lengths for voting
    original_test_char_sequences = get_raw_test_token_sequences(cfg.test_file)
    if not original_test_char_sequences:
        logger.error(
            "Cannot perform ensemble: failed to load original test character sequences for length reference. Exiting.")
        return

    num_test_examples = len(original_test_char_sequences)
    if num_test_examples == 0:
        logger.error(
            "No test examples found from parsing the test file. Cannot ensemble. Exiting")
        return

    # Verify that all prediction sources have the same number of examples as the original test file
    # This is a basic sanity check. More rigorous checks might be needed if source predictions can be ragged.
    for i, preds_from_one_source in enumerate(all_test_predictions_sources):
        if len(preds_from_one_source) != num_test_examples:
            logger.warning(
                f"Prediction source {i} has {len(preds_from_one_source)} examples, "
                f"but expected {num_test_examples} based on test file. "
                f"Ensemble will proceed but might be based on incomplete data for some examples if lengths differ later."
            )

    # List of lists: [example_idx][token_idx] -> final_label_str
    ensembled_final_labels = []

    for example_idx in range(num_test_examples):
        # Use the length of the original character sequence for this example
        char_seq_len = len(original_test_char_sequences[example_idx])

        if char_seq_len == 0:  # Handle genuinely empty test examples if they exist
            ensembled_final_labels.append([])
            logger.debug(
                f"Example {example_idx} is empty. Appending empty list for ensembled labels.")
            continue

        example_ensembled_labels_for_tokens = []
        # Iterate up to the length of original characters
        for token_idx in range(char_seq_len):
            votes_for_this_token = []
            for predictor_idx, source_predictions in enumerate(all_test_predictions_sources):
                # Ensure the current source_predictions has this example_idx
                if example_idx < len(source_predictions):
                    # Ensure the current example's prediction list from this source has this token_idx
                    if token_idx < len(source_predictions[example_idx]):
                        votes_for_this_token.append(
                            source_predictions[example_idx][token_idx])
                    else:
                        # Source's prediction for this example is shorter than original char sequence
                        logger.debug(
                            f"Vote from source {predictor_idx} missing for example {example_idx}, token {token_idx} (pred_len={len(source_predictions[example_idx])} < char_len={char_seq_len}). Using 'O'.")
                        votes_for_this_token.append('O')
                else:
                    # Source does not have this example_idx (e.g. if a source failed to produce predictions for all examples)
                    logger.debug(
                        f"Vote from source {predictor_idx} missing for example {example_idx} (source has only {len(source_predictions)} examples). Using 'O'.")
                    votes_for_this_token.append('O')

            if not votes_for_this_token:
                # This case should ideally not be reached if all_test_predictions_from_sources is populated
                # and the outer loops are correct. It means no votes were cast at all.
                logger.warning(
                    f"No votes collected for example {example_idx}, token {token_idx}. Defaulting to 'O'.")
                example_ensembled_labels_for_tokens.append('O')
            else:
                vote_counts = Counter(votes_for_this_token)
                majority_label = vote_counts.most_common(1)[0][0]
                example_ensembled_labels_for_tokens.append(majority_label)

        ensembled_final_labels.append(example_ensembled_labels_for_tokens)

    logger.info(
        f"Ensemble voting complete. Produced {len(ensembled_final_labels)} ensembled label sequences.")

    # --- 4. Output Generation --- #
    # Save the ensembled predictions in CoNLL-like format (char_token<TAB>label)
    # This uses the original character sequences obtained earlier for the token part.
    ensembled_conll_output_path = os.path.join(
        cfg.work_dir, "ensembled_predictions_raw.conll")

    write_ensembled_predictions_conll_format(
        output_path=ensembled_conll_output_path,
        original_char_sequences=original_test_char_sequences,
        ensembled_labels=ensembled_final_labels
    )

    # Process this CoNLL file into the final submission format (guid<SEP>text<SEP>labels)
    # cfg.output_file is 'result/ensembled_submission.txt' (or similar, from Config)
    # cfg.test_file is the original raw test data (e.g., 'data/final_test.txt') for GUIDs and full text strings.
    # Proceed only if CoNLL file was created
    if os.path.exists(ensembled_conll_output_path):
        process_final_submission_output(
            ensembled_conll_pred_file=ensembled_conll_output_path,
            final_submission_file=cfg.output_file,
            original_test_file_path=cfg.test_file
        )
    else:
        logger.error(
            f"Ensembled CoNLL output file {ensembled_conll_output_path} was not found. Cannot generate final submission file.")

    logger.info(
        f"Ensemble pipeline finished. Check logs and outputs in {cfg.work_dir} and {cfg.output_file}")


if __name__ == "__main__":
    main()
