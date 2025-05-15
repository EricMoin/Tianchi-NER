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
    device: str
    work_dir: str
    freeze_bert_layers: int
    num_prefix_tokens: int
    label2id: dict[str, int]
    id2label: dict[int, str]
    adversarial_training_start_epoch: int
    spatial_dropout: float
    embedding_dropout: float
    use_swa: bool
    swa_start_epoch: int
    swa_lr: float
    swa_freq: int

    # New parameters for Biaffine model
    biaffine_hidden_dim: int
    ignore_index: int  # For CrossEntropyLoss
    max_span_length: int  # To constrain span search during training/inference

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
                 adversarial_training_start_epoch: int,
                 spatial_dropout: float,
                 embedding_dropout: float,
                 use_swa: bool,
                 swa_start_epoch: int,
                 swa_lr: float,
                 swa_freq: int,
                 # New parameters
                 biaffine_hidden_dim: int,
                 ignore_index: int,
                 max_span_length: int
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
        self.spatial_dropout = spatial_dropout
        self.embedding_dropout = embedding_dropout
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        # New parameters
        self.biaffine_hidden_dim = biaffine_hidden_dim
        self.ignore_index = ignore_index
        self.max_span_length = max_span_length
