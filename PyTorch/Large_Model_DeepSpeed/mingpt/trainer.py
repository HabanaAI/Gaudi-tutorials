# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import gc
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import CfgNode as CN

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
except ImportError:
    print("INFO: no habana framework package installed")


def see_memory_usage(message, force=True, use_hpu=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(message)
    if use_hpu:
        print(
            f"MA {round(hthpu.memory_allocated() / (1024 * 1024),2 )} MB \
            Max_MA {round(hthpu.max_memory_allocated() / (1024 * 1024),2)} MB "
        )

        # get the peak memory to report correct data, so reset the counter for the next call
        hthpu.reset_peak_memory_stats()
    else:
        print(
            f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024),2 )} MB \
            Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024),2)} MB"
        )

        # get the peak memory to report correct data, so reset the counter for the next call
        if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
            torch.cuda.reset_peak_memory_stats()


class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, ds_enabled=False, use_hpu=False, dump_mem=False):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.ds_enabled = ds_enabled
        self.use_hpu = use_hpu
        self.dump_mem = dump_mem

        # determine the device we'll train on
        if config.device == "auto":
            if use_hpu:
                self.device = "hpu" if hthpu.is_available() else "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        if not ds_enabled:
            self.model = self.model.to(self.device)
            print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        if self.ds_enabled:
            self.optimizer = model.optimizer
        else:
            self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        step = 0
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]

            x, y = batch
            if self.ds_enabled:
                fp16 = model.fp16_enabled()
                if fp16 and x.is_floating_point():
                    x = x.half()

            # forward the model
            see_memory_usage(f"Step {step}: Before FWD:", force=self.dump_mem, use_hpu=self.use_hpu)
            logits, self.loss = model(x, y)
            see_memory_usage(f"Step {step}: After FWD:", force=self.dump_mem, use_hpu=self.use_hpu)

            # backprop and update the parameters
            if not self.ds_enabled:
                model.zero_grad(set_to_none=True)

            see_memory_usage(f"Step {step}: Before BWD:", force=self.dump_mem, use_hpu=self.use_hpu)
            if self.ds_enabled:
                model.backward(self.loss)
            else:
                self.loss.backward()
            if self.use_hpu:
                htcore.mark_step()
            see_memory_usage(f"Step {step}: After BWD:", force=self.dump_mem, use_hpu=self.use_hpu)

            # see_memory_usage(f'Step {step}: Before CLIP_NORM:', force=self.dump_mem, use_hpu=self.use_hpu)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # see_memory_usage(f'Step {step}: After CLIP_NORM:', force=self.dump_mem, use_hpu=self.use_hpu)

            see_memory_usage(f"Step {step}: Before STEP:", force=self.dump_mem, use_hpu=self.use_hpu)
            if self.ds_enabled:
                model.step()
            else:
                self.optimizer.step()
            if self.use_hpu:
                htcore.mark_step()
            see_memory_usage(f"Step {step}: After STEP:", force=self.dump_mem, use_hpu=self.use_hpu)

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
            step += 1
