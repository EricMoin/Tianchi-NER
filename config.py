class Config:
    num_epochs: int
    train_file: str
    dev_file: str
    test_file: str
    output_file: str
    model_name: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    device: str
    work_dir: str
    freeze_bert_layers: int
    num_prefix_tokens: int
    label2id: dict[str, int]
    id2label: dict[int, str]
    adversarial_training_start_epoch: int

    def __init__(self,
                 train_file: str,
                 dev_file: str,
                 test_file: str,
                 output_file: str,
                 model_name: str,
                 batch_size: int,
                 learning_rate: float,
                 weight_decay: float,
                 label2id: dict[str, int],
                 id2label: dict[int, str],
                 num_epochs: int,
                 device: str,
                 work_dir: str,
                 freeze_bert_layers: int,
                 num_prefix_tokens: int,
                 adversarial_training_start_epoch: int = 0,
                 ):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.output_file = output_file
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.device = device
        self.work_dir = work_dir
        self.freeze_bert_layers = freeze_bert_layers
        self.num_prefix_tokens = num_prefix_tokens
        self.label2id = label2id
        self.id2label = id2label
        self.adversarial_training_start_epoch = adversarial_training_start_epoch
