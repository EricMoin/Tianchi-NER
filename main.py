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
    config = Config(
        train_file='data/train.with_prompt.conll',
        dev_file='data/dev.with_prompt.conll',
        test_file='data/test.with_prompt.conll',
        output_file='result/prediction.conll',
        model_name='bert-base-chinese',
        batch_size=16,
        num_epochs=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        device='cuda',
        work_dir='result'
    )
    train_reader = ConllReader(config.train_file)
    dev_reader = ConllReader(config.dev_file)

    train_conll = list(train_reader.read())
    dev_conll = list(dev_reader.read())
    label_list = []
    for example in train_conll+dev_conll:
        label_list.extend(example.labels)
    label_list = sorted(list(set(label_list)))
    id2label = {i: label for i, label in enumerate(label_list)}
    print("labels:", label_list)

    model = AddressNER(pretrained_model_name=config.model_name,
                       num_labels=len(label_list))

    train_dataset = NERDataset(train_conll, model.tokenizer)
    dev_dataset = NERDataset(dev_conll, model.tokenizer)

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
        id2label=id2label
    )

    trainer.train()

    # trainer.test()

    # get_result()


if __name__ == "__main__":
    main()
