import os
import signal
import sys

sys.dont_write_bytecode = True


# Ensures that orphaned threads and libp2p daemons are killed when this script dies
def sigint_handler(signum, frame):
    print("\nCtrl+C detected. Killing all spawned processes.")
    # Kill the entire process group
    os.killpg(os.getpgid(0), signal.SIGTERM)
    sys.exit(1)


# Create a new process group
os.setpgrp()

# Set up the SIGINT handler
signal.signal(signal.SIGINT, sigint_handler)


import argparse
import logging
import math
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import disable_possible_user_warnings
from pytorch_optimizer import create_optimizer
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from api import APIServer
from interface import TerminalDashboard
from praxis import (
    PraxisConfig,
    PraxisForCausalLM,
    PraxisModel,
    TokenMonster,
    TokenMonsterConfig,
)

disable_possible_user_warnings()
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logger = CSVLogger("logs", name="praxis")

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)

AutoTokenizer.register(TokenMonsterConfig, TokenMonster)

# User args, accepted via CLI
parser = argparse.ArgumentParser(description="User-supplied arguments to this script.")
parser.add_argument(
    "--no_dashboard",
    action="store_true",
    default=False,
    help="Use dashboard (default: True)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use (default: cpu)",
)
parser.add_argument(
    "--data_path",
    type=str,
    nargs="+",
    default=None,
    help="Paths to directories of files to use as training data (default: None)",
)

args = parser.parse_args()

use_dashboard = False if args.no_dashboard else True
device = args.device if args.device else "cpu"
data_path = args.data_path

tokenizer_model = "englishcode-8000-consistent-v1"
tokenizer_config = TokenMonsterConfig(vocab_file=tokenizer_model, add_bos_token=True)
tokenizer = TokenMonster(vocab_file=tokenizer_config.vocab_file)

# System args
config = PraxisConfig(
    n_positions=512,
    n_embd=256,
    n_layer=6,
    n_head=8,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map=device,
    torch_dtype="float32",
)

dataset_config = dict(repo="HuggingFaceFW/fineweb", key="text")
# dataset_config = dict(repo="open-phi/textbooks", key="markdown")

optimizer_config = dict(
    optimizer_name="Lion",
    lr=1e-4,
    weight_decay=1e-5,
    wd_ban_list=[
        "bias",
        "RMSNorm.weight",
        "RMSNorm.bias",
    ],
)

# Dashboard config
max_data_points = 10000
max_prompt_charts = 2048
predict_interval = 3

hparams = dict(
    batch_size=16,
    accumulate_grad_batches=4,
    block_size=256,
)

train_params = dict(
    accelerator=f"cpu" if args.device == "cpu" else "gpu",
    strategy="auto",
    devices=[int(device.split(":")[1])] if args.device.startswith("cuda") else "auto",
    max_steps=-1,
    max_epochs=-1,
    reload_dataloaders_every_n_epochs=0,
    precision="32-true",
    accumulate_grad_batches=hparams[
        "accumulate_grad_batches"
    ],  # must be 1 for Hivemind training
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    benchmark=True,
    enable_progress_bar=False if use_dashboard else True,
    enable_model_summary=False,
    logger=logger,
    callbacks=[],
)


class PraxisTrainer(LightningModule):
    """
    A training module for Praxis.
    """

    def __init__(self, model, optimizer, hparams):
        super(PraxisTrainer, self).__init__()

        self.model, self.optimizer = (model, optimizer)

        self.automatic_optimization = True
        self.batch_size = hparams["batch_size"]
        self.save_hyperparameters(ignore=["model", "optimizer"])

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch, labels=batch)
        loss = outputs[0]

        self.log_dict(
            {
                "loss": loss,
                "step": batch_idx // train_params["accumulate_grad_batches"],
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss

    def configure_optimizers(self):
        "Create optimizer and scheduler"
        return [self.optimizer]


class TerminalInterface(Callback):
    """
    A single pane of glass containing charts and information.
    """

    def __init__(self):
        super().__init__()
        self.ema = 0
        self.last_time = datetime.now()
        self.text = "Once"
        self.max_length = max_prompt_charts
        self.interval = predict_interval
        self.dashboard = TerminalDashboard(max_data_points)
        try:
            self.dashboard.start()
            self.dashboard.update_url(api_url)
        except KeyboardInterrupt:
            self.dashboard.stop()

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        loss = trainer.callback_metrics.get("loss", 0)
        self.ema = self._compute_ema_loss(float(loss), self.ema)
        self.dashboard.update_losses(self.ema, random.gauss())

        step = trainer.callback_metrics.get("step", 0)
        self.dashboard.update_step(step.item())

        self._generate_sample_text(lm, batch_idx, interval=self.interval)
        self.dashboard.update_utilization(lm.model.get_expert_utilization())

    def _generate_sample_text(self, lm, batch_idx, interval=10):

        if not self._is_time_passed(self.last_time, self.interval):
            return

        lm.model.eval()
        self.text = generator.stream(self.text)

        while len(self.text) > self.max_length:
            self.text = self.text[1:]

        self.dashboard.update_status(self.text)
        self.last_time = datetime.now()
        lm.model.train()

    def _is_time_passed(self, original_time, x_seconds):
        current_time = datetime.now()
        time_difference = current_time - original_time
        return time_difference > timedelta(seconds=x_seconds)

    def _compute_ema_loss(self, current_loss, prev_avg_loss, alpha=0.01):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (alpha * current_loss) + (1 - alpha) * prev_avg_loss


class HuggingfaceDataset(IterableDataset):
    """
    A wrapper that streams, tokenizes and batches data for training.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dataset = load_dataset(
            config.get("repo", "HuggingFaceFW/fineweb"),
            split="train",
            streaming=True,
            cache_dir="./tmp/pile",
            trust_remote_code=True,
        )
        self.key = config.get("key", "text")

        self.cached_text = ""

    def __iter__(self):

        buffer_size = 10_000
        text_cache_size = 10 * buffer_size

        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=buffer_size,
        )

        for document in shuffled:
            self.cached_text += document.get(self.key) + self.tokenizer.eos_token
            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=self.block_size,
                stride=16,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != self.block_size:
                    break
                yield batch


class MultiDirectoryDataset(IterableDataset):
    """
    A file-based iterable dataset that recursively reads files from multiple directories,
    tokenizes them, and returns batches for PyTorch Lightning.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, directories: List[str], block_size: int
    ):
        self.tokenizer = tokenizer
        self.directories = directories
        self.block_size = block_size
        self.cached_text = ""
        self.file_list = self._get_file_list()

    def _get_file_list(self) -> List[str]:
        """Recursively get all files in all directories."""
        file_list = []
        for directory in self.directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_list.append(os.path.join(root, file))
        return file_list

    def _read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def __iter__(self):
        buffer_size = 10_000
        text_cache_size = 10 * buffer_size
        block_size = self.block_size

        random.shuffle(self.file_list)

        for file_path in self.file_list:
            self.cached_text += self._read_file(file_path) + self.tokenizer.eos_token
            if len(self.cached_text) < text_cache_size:
                continue

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=block_size,
                stride=16,
                padding=True,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != block_size:
                    break
                yield batch


class Generator:
    """
    Wraps a model in a simplified generation API, using TokenMonster's decoder for streaming.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, kwargs={}):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if args.device.startswith("cuda"):
            input_ids = input_ids.to(int(device.split(":")[1]))
        defaults = dict(
            do_sample=True,
            max_new_tokens=1,
            temperature=0.95,
            eta_cutoff=0.002,
            penalty_alpha=0.6,
            top_k=4,
            repetition_penalty=1.5,
        )
        combined = {**defaults, **kwargs}
        if "prompt" in combined:
            del combined["prompt"]
        outputs = self.model.generate(input_ids, **combined)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def stream(self, prompt, kwargs={}):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]

        if args.device.startswith("cuda"):
            input_ids = input_ids.to(int(device.split(":")[1]))

        defaults = dict(
            do_sample=True,
            max_new_tokens=1,
            temperature=0.95,
            eta_cutoff=0.002,
            penalty_alpha=0.6,
            top_k=4,
            repetition_penalty=1.5,
        )
        combined = {**defaults, **kwargs}
        if "prompt" in combined:
            del combined["prompt"]

        generated_text = ""
        total_tokens = 0
        max_tokens = combined.get("max_new_tokens", 1)

        decoder = tokenizer.get_decoder()
        while total_tokens < max_tokens:
            outputs = self.model.generate(input_ids, **combined)
            new_token = outputs[0, -1].unsqueeze(0)  # Get only the last generated token
            decoded_token = decoder.decode(new_token.tolist())  # Decode single token

            if decoded_token:
                generated_text += decoded_token
                total_tokens += 1

            input_ids = outputs  # Use the full output as the next input

            if total_tokens >= max_tokens:
                break

        return prompt + generated_text


model = AutoModelForCausalLM.from_config(config)

print(model)

generator = Generator(model, tokenizer)

api_server = APIServer(generator)
api_server.start()
api_url = api_server.get_url() + "/generate"

if use_dashboard:
    train_params["callbacks"].append(TerminalInterface())

# create the optimizer
optimizer = create_optimizer(model, **optimizer_config)

# Load a dataset
if data_path:
    dataset = MultiDirectoryDataset(tokenizer, data_path, hparams["block_size"])
else:
    dataset = HuggingfaceDataset(tokenizer, dataset_config, hparams["block_size"])

# Wrap the model in a pytorch-lightning module
train_model = PraxisTrainer(model, optimizer, hparams)

# fit the trainer and run
trainer = Trainer(**train_params)
trainer.fit(train_model, dataset)

# import ipaddress
# from functools import partial
# from hivemind.utils.networking import log_visible_maddrs
# from lightning.fabric.utilities.seed import reset_seed, seed_everything
# from lightning_hivemind.strategy import HivemindStrategy


# # set some basic configuration values
# initial_peers = flatten_list(args.initial_peers)
# target_batch_size = 8192

# # define the hivemind strategy
# strategy = HivemindStrategy(
#     run_id=f"hiveminer",
#     batch_size=batch_size,
#     target_batch_size=target_batch_size,
#     initial_peers=initial_peers,
#     use_ipfs=False,
#     use_relay=True,
#     use_auto_relay=True,
#     verbose=False,
#     wait_timeout=60,
#     bootstrap_timeout=45,
#     matchmaking_time=90.0,
#     averaging_timeout=300.0,
#     delay_state_averaging=True,
#     delay_grad_averaging=True,
#     delay_optimizer_step=True,
#     offload_optimizer=True,
#     reuse_grad_buffers=False,
#     # grad_compression=Float16Compression(),
#     # state_averaging_compression=Float16Compression(),
#     # load_state_compression=NoCompression(),
#     # scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),
# )

# # print my peer id to console
# visible_addresses = [
#     str(a)
#     for a in strategy.dht.get_visible_maddrs()
#     if not ipaddress.ip_address(a.values()[0]).is_loopback
# ]

# log_visible_maddrs(strategy.dht.get_visible_maddrs(), only_p2p=False)
# # my_ids = []
# # pattern = r"(/p2p/.*)"
# # for peer in list(visible_addresses):
# #     match = re.search(pattern, peer)
# #     if match:
# #         my_ids.append(match.group(1))

# # for peer in list(set(my_ids)):
# #     print(f"PEER-ID: {peer}")


# class MinerModelSaver(Callback):
#     """Periodically save the model during training."""

#     def __init__(
#         self,
#         save_every,
#         output_dir,
#     ):
#         super().__init__()
#         self.step = 0
#         self.last_step = 0
#         self.save_every = save_every
#         self.output_dir = output_dir

#     @property
#     def save_every_check(self):
#         return (
#             self.step > 0
#             and self.save_every > 0
#             and self.last_step != self.step
#             and self.step % self.save_every == 0
#         )

#     def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
#         super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

#         self.step = int(trainer.callback_metrics.get("global_step", 0))

#         if self.save_every_check:
#             self.save_pytorch_model(trainer, lm)

#         self.last_step = self.step

#     def save_pytorch_model(self, trainer, lm):
#         lm.model.save_pretrained(self.output_dir, safe_serialization=True)
