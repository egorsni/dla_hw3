import argparse
import collections
import warnings

import numpy as np
import torch

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def _get_ft_model(self, finetuing_path):
        finetuing_path = str(finetuing_path)
        self.logger.info("Loading checkpoint: {} ...".format(finetuing_path))
        checkpoint = torch.load(finetuing_path, self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logger.info(
            "Checkpoint loaded."
        )
    

class FastSpeechConfig:
    def __init__(
            self,
            vocab_size = 300,
            max_seq_len = 3000,
        
            encoder_dim = 256,
            encoder_n_layer = 4,
            encoder_head = 2,
            encoder_conv1d_filter_size = 1024,
        
            decoder_dim = 256,
            decoder_n_layer = 4,
            decoder_head = 2,
            decoder_conv1d_filter_size = 1024,
        
            fft_conv1d_kernel = (9, 1),
            fft_conv1d_padding = (4, 0),
        
            duration_predictor_filter_size = 256,
            duration_predictor_kernel_size = 3,
            dropout = 0.1,
            
            PAD = 0,
            UNK = 1,
            BOS = 2,
            EOS = 3,
        
            PAD_WORD = '<blank>',
            UNK_WORD = '<unk>',
            BOS_WORD = '<s>'
            ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.encoder_n_layer = encoder_n_layer
        self.encoder_head = encoder_head
        self.encoder_conv1d_filter_size = encoder_conv1d_filter_size
        self.decoder_dim = decoder_dim
        self.decoder_n_layer = decoder_n_layer
        self.decoder_head = decoder_head
        self.decoder_conv1d_filter_size = decoder_conv1d_filter_size
        self.fft_conv1d_kernel = fft_conv1d_kernel
        self.fft_conv1d_padding = fft_conv1d_padding
        self.duration_predictor_filter_size = duration_predictor_filter_size
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.dropout = dropout
        self.PAD = PAD
        self.UNK = UNK
        self.BOS = BOS
        self.EOS = EOS
        self.PAD_WORD = PAD_WORD
        self.UNK_WORD = UNK_WORD
        self.BOS_WORD = BOS_WORD
        
        
@dataclass
class MelSpectrogramConfig:
    num_mels = 80
    

def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model_config = FastSpeechConfig(**dict(config["arch"]['args']))
    mel_config = MelSpectrogramConfig()
#     model = config.init_obj({}, module_arch, model_config=model_config)
    module_name = config["arch"]["type"]
    model = getattr(module_arch, module_name)(model_config, mel_config)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-f",
        "--finetune",
        default=None,
        type=str,
        help="path to pretrained model checkpoint (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)





# @dataclass
# class TrainConfig:
#     def __init__(checkpoint_path = "./model_new",
#     logger_path = "./logger",
#     mel_ground_truth = "./mels",
#     alignment_path = "./alignments",
#     data_path = './data/train.txt',
    
#     wandb_project = 'fastspeech_example',
    
#     text_cleaners = ['english_cleaners'],

# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#     device = 'cuda:0',

#     batch_size = 16,
#     epochs = 2000,
#     n_warm_up_step = 4000,

#     learning_rate = 1e-3,
#     weight_decay = 1e-6,
#     grad_clip_thresh = 1.0,
#     decay_step = [500000, 1000000, 2000000],

#     save_step = 3000,
#     log_step = 5,
#     clear_Time = 20,

#     batch_expand_size = 32):
#     self.checkpoint_path = checkpoint_path
#     self.logger_path = logger_path
#     self.mel_ground_truth = mel_ground_truth
#     self.alignment_path = alignment_path
#     self.data_path = data_path
#     self.wandb_project = wandb_project
#     self.text_cleaners = text_cleaners
#     self.device = device
#     self.batch_size = batch_size
#     self.epochs = epochs
#     self.n_warm_up_step = n_warm_up_step
#     self.learning_rate = learning_rate
#     self.weight_decay = weight_decay
#     self.grad_clip_thresh = grad_clip_thresh
#     self.decay_step = decay_step
#     self.save_step = save_step
#     self.log_step = log_step
#     self.clear_Time = clear_Time
#     self.batch_expand_size = batch_expand_size