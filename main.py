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
    labels = [
        'prov', 'city', 'district', 'devzone',
        'town', 'community', 'village_group',
        'road', 'roadno', 'poi', 'subpoi',
        'houseno', 'cellno', 'floorno', 'roomno',
        'detail', 'assist', 'distance',
        'intersection', 'redundant',
        'person', 'others',
    ]
    label_list = []
    for label in labels:
        label_list.append(f'B-{label}')
        label_list.append(f'I-{label}')
        label_list.append(f'E-{label}')
        label_list.append(f'S-{label}')
    label_list.append('O')
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    config = Config(
        train_file='data/train.conll',
        dev_file='data/dev.conll',
        test_file='data/final_test.txt',
        output_file='result/prediction.conll',
        model_name='bert-base-chinese',
        batch_size=16,
        num_epochs=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        device='cuda',
        work_dir='result',
        freeze_bert_layers=0,
        num_prefix_tokens=20,
        label2id=label2id,
        id2label=id2label,
        adversarial_training_start_epoch=3,
        focal_loss_alpha=0.25,
        focal_loss_gamma=1.5,
        hybrid_loss_weight_crf=0.5,
        hybrid_loss_weight_focal=0.5,
        crf_transition_penalty=0.175,
    )
    train_reader = ConllReader(config.train_file)
    dev_reader = ConllReader(config.dev_file)

    train_conll = list(train_reader.read())
    dev_conll = list(dev_reader.read())

    print("len(label_list):", len(label_list))
    print("label_list:", label_list)

    model = AddressNER(pretrained_model_name=config.model_name,
                       num_labels=len(label_list),
                       freeze_bert_layers=config.freeze_bert_layers,
                       focal_alpha=config.focal_loss_alpha,
                       focal_gamma=config.focal_loss_gamma,
                       weight_crf=config.hybrid_loss_weight_crf,
                       weight_focal=config.hybrid_loss_weight_focal,
                       crf_transition_penalty=config.crf_transition_penalty
                       )

    train_dataset = NERDataset(train_conll, model.tokenizer, label2id)
    dev_dataset = NERDataset(dev_conll, model.tokenizer, label2id)

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

    # trainer.train()

    trainer.test()

    get_result()


if __name__ == "__main__":
    main()
