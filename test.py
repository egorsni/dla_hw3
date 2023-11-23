import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
import hw_asr.model as module_arch
import waveglow
import text
import audio
import utils
import numpy as np
import os

from train import FastSpeechConfig, MelSpectrogramConfig

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

def synthesis(model, text, alpha=1.0, beta=1.0, gamma=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to('cuda')
    src_pos = torch.from_numpy(src_pos).long().to('cuda')
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, train=False, beta=beta, gamma=gamma)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def main(config, texts):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
#     text_encoder = config.get_text_encoder()

    # setup data_loader instances
#     dataloaders = get_dataloaders(config)

    # build model architecture
    model_config = FastSpeechConfig(**dict(config["arch"]['args']))
    mel_config = MelSpectrogramConfig()
#     model = config.init_obj({}, module_arch, model_config=model_config)
    module_name = config["arch"]["type"]
    model = getattr(module_arch, module_name)(model_config, mel_config)
    
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    results = []
    
    tests = open(texts, 'r').read().split('\n')
    data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)

    for i, phn in tqdm(enumerate(data_list)):
        for alpha in [0.8, 1, 1.2]:
            for beta in [0.8, 1, 1.2]:
                for gamma in [0.8, 1, 1.2]:
            
                    mel, mel_cuda = synthesis(model, phn,alpha=alpha,beta=beta,gamma=gamma)
                    
                    os.makedirs(f"./test_results/{alpha}/{beta}/{gamma}", exist_ok=True)
                    
                    audio.tools.inv_mel_spec(
                        mel, f"./test_results/{alpha}/{beta}/{gamma}/s={i}_alpha_{alpha}_beta_{beta}_gamma_{gamma}.wav"
                    )
                    
                    waveglow.inference.inference(
                        mel_cuda, WaveGlow,
                        f"./test_results/{alpha}/{beta}/{gamma}/s={i}_waveglow_alpha_{alpha}_beta_{beta}_gamma_{gamma}.wav")

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
        "-t",
        "--text",
        default='./test_texts.txt',
        type=str,
        help="path to texts u wan to speech",
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

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path('hw_asr/configs/') / "one_batch_test.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

#     # if `--test-data-folder` was provided, set it as a default test set
#     if args.test_data_folder is not None:
#         test_data_folder = Path(args.test_data_folder).absolute().resolve()
#         assert test_data_folder.exists()
#         config.config["data"] = {
#             "test": {
#                 "batch_size": args.batch_size,
#                 "num_workers": args.jobs,
#                 "datasets": [
#                     {
#                         "type": "CustomDirAudioDataset",
#                         "args": {
#                             "audio_dir": str(test_data_folder / "audio"),
#                             "transcription_dir": str(
#                                 test_data_folder / "transcriptions"
#                             ),
#                         },
#                     }
#                 ],
#             }
#         }

#     assert config.config.get("data", {}).get("test", None) is not None
#     config["data"]["test"]["batch_size"] = args.batch_size
#     config["data"]["test"]["n_jobs"] = args.jobs

    print(args.text)
    main(config, args.text)