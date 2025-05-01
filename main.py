import os
from conll_reader import ConllReader
from dataset import NERDataset
from model import AddressNER
from config import Config
from torch.utils.data import DataLoader

from prompt_writer import PromptWriter
from tagger import RuleBasedTagger, Tagger
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
    config = Config(
        train_file='data/train.conll',
        dev_file='data/dev.conll',
        test_file='data/final_test.conll',
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
        id2label=id2label
    )
    train_reader = ConllReader(config.train_file)
    dev_reader = ConllReader(config.dev_file)

    train_conll = list(train_reader.read())
    dev_conll = list(dev_reader.read())

    print("len(label_list):", len(label_list))
    print("label_list:", label_list)

    model = AddressNER(pretrained_model_name=config.model_name,
                       num_labels=len(label_list), freeze_bert_layers=config.freeze_bert_layers)

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
    # tagger = RuleBasedTagger()
    # prompt_writer = PromptWriter(
    #     conll_file='result/test_with_prompt.conll',
    #     tagger=tagger,
    # )
    # prompt_writer.write_test('data/final_test.txt')
