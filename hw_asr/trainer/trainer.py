import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker
import torch

import waveglow
import text
import audio
import utils
import numpy as np
import os


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
#         self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        print('LEN_EPOCH', self.len_epoch)
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        print(self.lr_scheduler)
        self.log_step = 50
        
        self.logging_metrics = ["loss",'mel_loss','duration_loss','energy_predictor_loss','pitch_predictor_loss', 'mean_energy','mean_pitch','mean_energy_target', 'mean_pitch_target']
        self.train_metrics = MetricTracker(
            *[m for m in self.logging_metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            *[m for m in self.logging_metrics], writer=self.writer
        )
        self.WaveGlow = utils.get_WaveGlow()
        self.WaveGlow = self.WaveGlow.cuda()
        
        if isinstance(self.optimizer, torch.optim.Adam):
            d = self.optimizer.state_dict()
            d['param_groups'][0]['betas'] = (0.9, 0.98)
            self.optimizer.load_state_dict(d)
        print('optimizer', self.optimizer.state_dict())

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ['ref_audios','mix_audios','target_audios', 'ref_lengths']:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        batch['target_ids'] = torch.tensor(batch['target_ids']).to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
#             self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
#                 self.writer.add_scalar(
#                     "learning rate", self.lr_scheduler.get_last_lr()[0]
#                 )

                self.writer.add_scalar(
                    "learning rate", self.optimizer.param_groups[0]['lr']
                )
    
#                 self._log_predictions(**batch)
#                 self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch - 1:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        character = batch["text"].long().to(self.device)
        mel_target = batch["mel_target"].float().to(self.device)
        duration = batch["duration"].int().to(self.device)
        mel_pos = batch["mel_pos"].long().to(self.device)
        src_pos = batch["src_pos"].long().to(self.device)
        max_mel_len = batch["mel_max_len"]
        
        energy = batch["energy"].to(self.device)
        pitch = batch["pitch"].to(self.device)
        
#         batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
            
        mel_output, duration_predictor_output, energy_output, pitch_output = self.model(character,
                                                          src_pos,
                                                          mel_pos=mel_pos,
                                                          mel_max_length=max_mel_len,
                                                          length_target=duration,
                                                          pitch_target=pitch, energy_target=energy)
        mel_loss, duration_loss, energy_predictor_loss, pitch_predictor_loss = self.criterion(mel_output,
                                                    duration_predictor_output,
                                                     energy_output, pitch_output,
                                                    mel_target,
                                                    duration,
                                                    energy, pitch)
        total_loss = mel_loss + duration_loss + energy_predictor_loss + pitch_predictor_loss
        batch["loss"] = total_loss
        batch['mel_loss'] = mel_loss
        batch['duration_loss'] = duration_loss
        batch['energy_predictor_loss'] = energy_predictor_loss
        batch['pitch_predictor_loss'] = pitch_predictor_loss
        
        batch['mean_energy'] = energy_output.mean()
        batch['mean_pitch'] = pitch_output.mean()
        
        batch['mean_energy_target'] = energy.mean()
        batch['mean_pitch_target'] = pitch.mean()
        
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        for metric in self.logging_metrics:
            metrics.update(metric, batch[metric].item(), n=batch['text'].shape[0])
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
#             self._log_predictions(**batch)
#             self._log_spectrogram(batch["spectrogram"])
        data_list = get_data()
    
        aboba = [
                    (1, 1, 1),
                    (1, 1, 1.2),
                    (1, 1, 0.8),
                    (1, 1.2, 1),
                    (1, 0.8, 1),
                    (1.2, 1, 1),
                    (0.8, 1, 1),
                    (1.2, 1.2, 1.2),
                    (0.8, 0.8, 0.8),
                ]
        
        for i, phn in tqdm(enumerate(data_list)):
            for alpha in [0.8, 1, 1.2]:
                for beta in [0.8, 1, 1.2]:
                    for gamma in [0.8, 1, 1.2]:
                
                        mel, mel_cuda = synthesis(self.model, phn,alpha=alpha,beta=beta,gamma=gamma)
                        
                        os.makedirs(f"results/{alpha}/{beta}/{gamma}", exist_ok=True)
                        
                        audio.tools.inv_mel_spec(
                            mel, f"results/{alpha}/{beta}/{gamma}/s={i}_alpha_{alpha}_beta_{beta}_gamma_{gamma}.wav"
                        )
                        
                        waveglow.inference.inference(
                            mel_cuda, self.WaveGlow,
                            f"results/{alpha}/{beta}/{gamma}/s={i}_waveglow_alpha_{alpha}_beta_{beta}_gamma_{gamma}.wav"
                        )
        # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins="auto")

#         if self.lr_scheduler is not None:
#                 self.lr_scheduler.step(self.evaluation_metrics.avg('loss'))
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    
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


def get_data():
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
#     data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)


    return data_list
#     def _log_predictions(
#             self,
#             text,
#             log_probs,
#             log_probs_length,
#             audio_path,
#             examples_to_log=10,
#             *args,
#             **kwargs,
#     ):
#         # TODO: implement logging of beam search results
#         if self.writer is None:
#             return
#         argmax_inds = log_probs.cpu().argmax(-1).numpy()
#         argmax_inds = [
#             inds[: int(ind_len)]
#             for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
#         ]
#         argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
#         argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
#         tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
#         shuffle(tuples)
#         rows = {}
#         for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
#             target = BaseTextEncoder.normalize_text(target)
#             wer = calc_wer(target, pred) * 100
#             cer = calc_cer(target, pred) * 100

#             rows[Path(audio_path).name] = {
#                 "target": target,
#                 "raw prediction": raw_pred,
#                 "predictions": pred,
#                 "wer": wer,
#                 "cer": cer,
#             }
#         self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

#     def _log_spectrogram(self, spectrogram_batch):
#         spectrogram = random.choice(spectrogram_batch.cpu())
#         image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
#         self.writer.add_image("spectrogram", ToTensor()(image))

#     @torch.no_grad()
#     def get_grad_norm(self, norm_type=2):
#         parameters = self.model.parameters()
#         if isinstance(parameters, torch.Tensor):
#             parameters = [parameters]
#         parameters = [p for p in parameters if p.grad is not None]
#         total_norm = torch.norm(
#             torch.stack(
#                 [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
#             ),
#             norm_type,
#         )
#         return total_norm.item()