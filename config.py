class Config:
    num_epochs: int
    train_file: str
    dev_file: str
    test_file: str
    output_file: str
    model_name: str  # Path to the primary pre-trained or adapted model for single runs
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    work_dir: str  # Base working directory
    freeze_bert_layers: int
    num_prefix_tokens: int
    label2id: dict[str, int]
    id2label: dict[int, str]
    adversarial_training_start_epoch: int
    focal_loss_alpha: float
    focal_loss_gamma: float
    hybrid_loss_weight_crf: float
    hybrid_loss_weight_focal: float
    crf_transition_penalty: float
    spatial_dropout: float
    embedding_dropout: float
    use_swa: bool
    swa_start_epoch: int
    swa_lr: float
    swa_freq: int
    seed: int  # Added for reproducibility

    # New fields for K-Fold and Multiple Model Adaptation
    k_folds: int
    # List of HuggingFace model names for adaptation
    base_model_names: list[str]
    adapted_model_paths: list[str]  # To be populated after adaptation

    # Domain Adaptation Specific Hyperparameters (optional, with defaults)
    adaptation_corpus_file: str
    adaptation_max_length: int
    adaptation_batch_size: int
    adaptation_lr: float
    adaptation_weight_decay: float
    adaptation_adam_epsilon: float
    adaptation_max_grad_norm: float
    adaptation_num_epochs: int
    adaptation_warmup_steps: int
    adaptation_mask_probability: float

    def __init__(self,
                 train_file: str,
                 dev_file: str,
                 test_file: str,
                 output_file: str,
                 model_name: str,  # Default model if not using multi-model adaptation
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
                 focal_loss_alpha: float,
                 focal_loss_gamma: float,
                 hybrid_loss_weight_crf: float,
                 hybrid_loss_weight_focal: float,
                 crf_transition_penalty: float,
                 spatial_dropout: float,
                 embedding_dropout: float,
                 use_swa: bool,
                 swa_start_epoch: int,
                 swa_lr: float,
                 swa_freq: int,
                 seed: int = 2025,  # Default seed
                 # K-Fold and Multi-Model params
                 k_folds: int = 5,
                 base_model_names: list[str] | None = None,
                 # Adaptation params with defaults
                 adapted_model_paths: list[str] | None = None,
                 adaptation_corpus_file: str = "data/address.txt",
                 adaptation_max_length: int = 128,
                 adaptation_batch_size: int = 16,
                 adaptation_lr: float = 5e-5,
                 adaptation_weight_decay: float = 0.01,
                 adaptation_adam_epsilon: float = 1e-8,
                 adaptation_max_grad_norm: float = 1.0,
                 adaptation_num_epochs: int = 3,
                 adaptation_warmup_steps: int = 0,
                 adaptation_mask_probability: float = 0.15
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
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.hybrid_loss_weight_crf = hybrid_loss_weight_crf
        self.hybrid_loss_weight_focal = hybrid_loss_weight_focal
        self.crf_transition_penalty = crf_transition_penalty
        self.spatial_dropout = spatial_dropout
        self.embedding_dropout = embedding_dropout
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        self.seed = seed

        self.k_folds = k_folds
        self.base_model_names = base_model_names if base_model_names is not None else []
        self.adapted_model_paths = adapted_model_paths if adapted_model_paths is not None else []

        self.adaptation_corpus_file = adaptation_corpus_file
        self.adaptation_max_length = adaptation_max_length
        self.adaptation_batch_size = adaptation_batch_size
        self.adaptation_lr = adaptation_lr
        self.adaptation_weight_decay = adaptation_weight_decay
        self.adaptation_adam_epsilon = adaptation_adam_epsilon
        self.adaptation_max_grad_norm = adaptation_max_grad_norm
        self.adaptation_num_epochs = adaptation_num_epochs
        self.adaptation_warmup_steps = adaptation_warmup_steps
        self.adaptation_mask_probability = adaptation_mask_probability
