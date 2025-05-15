import os
from conll_reader import ConllReader
from dataset import NERDataset
from model import AddressNER
from config import Config
from torch.utils.data import DataLoader

from trainer import Trainer


def get_result():
    path = os.getcwd()
    exp_dir = os.path.join(path, 'result')
    os.makedirs(exp_dir, exist_ok=True)
    with open(f'{exp_dir}/pred.txt', 'r', encoding='utf8') as fin, \
            open(f'{exp_dir}/result.pred.txt', 'w', encoding='utf8') as fout:
        guid = 1
        tokens = []
        labels = []
        is_orig = True
        for line in fin:
            if line == '' or line == '\n':
                if tokens:
                    print(guid, ''.join(tokens), ' '.join(
                        labels), sep='\u0001', file=fout)
                    guid += 1
                    tokens = []
                    labels = []
                    is_orig = True
            else:
                splits = line.split('\t')
                if splits[0] == '<EOS>':
                    is_orig = False
                if is_orig:
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
        if tokens:
            print(guid, ''.join(tokens), ' '.join(
                labels), sep='\u0001', file=fout)


def main():
    # Define SPAN-BASED labels for Biaffine model
    entity_types = [
        'prov', 'city', 'district', 'devzone', 'town',
        'community', 'village_group', 'road', 'roadno',
        'poi', 'subpoi', 'houseno', 'cellno', 'floorno',
        'roomno', 'detail', 'assist', 'distance',
        'intersection', 'redundant', 'others'
    ]
    label_list = []
    for entity_type in entity_types:
        label_list.append(f'B-{entity_type}')
        label_list.append(f'I-{entity_type}')
        label_list.append(f'E-{entity_type}')
        label_list.append(f'S-{entity_type}')
    label_list.append('O')

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    config = Config(
        train_file='data/train.conll',
        dev_file='data/dev.conll',
        test_file='data/final_test.txt',
        output_file='result/prediction.conll',
        model_name='pretrained/address_adapted_model',
        batch_size=8,  # User confirmed trying batch_size=8
        num_epochs=15,
        learning_rate=2e-5,
        weight_decay=0.01,
        device='cuda',
        work_dir='result',
        freeze_bert_layers=0,
        num_prefix_tokens=20,
        label2id=label2id,
        id2label=id2label,
        adversarial_training_start_epoch=3,
        spatial_dropout=0.2,
        embedding_dropout=0.1,
        use_swa=True,
        swa_start_epoch=0,
        swa_lr=1e-5,
        swa_freq=2,
        biaffine_hidden_dim=150,
        ignore_index=-100,
        # Try reducing if OOM persists AFTER NERDataset is fixed (e.g., to 10)
        max_span_length=15
    )
    train_reader = ConllReader(config.train_file)
    dev_reader = ConllReader(config.dev_file)

    train_conll = list(train_reader.read())
    dev_conll = list(dev_reader.read())

    print(f"Number of SPAN labels for Biaffine model: {len(label_list)}")
    print(f"SPAN labels: {label_list}")
    print(f"Label ID for 'O' (non-entity span): {label2id['O']}")

    # CRITICAL: NERDataset (in dataset.py) MUST be updated.
    # It currently processes CoNLL BIOES tags for token-level classification.
    # For the Biaffine model, it needs to:
    # 1. Take `train_conll` (list of sentences with BIOES token tags).
    # 2. For each sentence, generate a `span_labels_gold` matrix of shape [seq_len, seq_len].
    #    - Each cell (i, j) in this matrix should contain the `label_id` (from the SPAN `label2id` map)
    #      for the text span from token `i` to token `j`.
    #    - If the span (i,j) is not one of the defined entities, its label should be `label2id['O']`.
    #    - Invalid spans (e.g., j < i, or spans involving padding, or spans exceeding config.max_span_length)
    #      should be assigned `config.ignore_index`.
    # 3. The `__getitem__` in NERDataset should return this `span_labels_gold` tensor instead of token-level `labels`.

    model = AddressNER(
        num_labels=len(label_list),  # Number of SPAN types
        config=config
    )

    # Pass config to NERDataset, it needs ignore_index, max_span_length, O_label_id etc.
    train_dataset = NERDataset(
        train_conll, model.tokenizer, label2id)
    dev_dataset = NERDataset(
        dev_conll, model.tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False)

    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=dev_loader,
        device=config.device,
    )

    trainer.train()

    trainer.test()

    get_result()


if __name__ == "__main__":
    main()
