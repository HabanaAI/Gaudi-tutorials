from argparse import ArgumentParser
from urllib.request import urlopen

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.runtime.lr_schedules import WarmupLR

# import lightning as L
from lightning_utilities import module_available

if module_available("lightning"):
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin
    from lightning.pytorch.profilers.pytorch import PyTorchProfiler
    from lightning.pytorch.strategies import StrategyRegistry
    from lightning.pytorch.utilities.types import STEP_OUTPUT
elif module_available("pytorch_lightning"):
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.plugins import DeepSpeedPrecisionPlugin
    from pytorch_lightning.profilers.pytorch import PyTorchProfiler
    from pytorch_lightning.strategies import StrategyRegistry
    from pytorch_lightning.utilities.types import STEP_OUTPUT

from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.strategies import HPUDeepSpeedStrategy, HPUParallelStrategy

import torch
from torch.utils.data import DataLoader

from lightning_gpt import callbacks, data, models


try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
except:
    print("INFO: no habana framework package installed")

import gc
import torch.distributed as dist


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


def main(args):
    with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as f:
        text = f.read()

    train_dataset = data.CharDataset(text, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    GPT_class = None
    extra_kwargs = {}

    if args.implementation == "mingpt":
        GPT_class = models.MinGPT
        extra_kwargs.update(
            dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
            )
        )

    elif args.implementation == "nanogpt":
        GPT_class = models.NanoGPT
        extra_kwargs["dropout"] = 0.1

    else:
        raise ValueError(f"Unsupported implementation {args.implementation}")

    if args.strategy == "deepspeed":
        if GPT_class == models.MinGPT:
            GPT_class = models.DeepSpeedMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.DeepSpeedNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with DeepSpeed")
        extra_kwargs["offload"] = False

    elif args.strategy == "fsdp_native":
        if GPT_class == models.MinGPT:
            GPT_class = models.FSDPMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.FSDPNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with FSDP")

    model = GPT_class(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        **extra_kwargs,
    )

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."
                "Please install torch >= 1.14 or disable compile."
            )
        model = torch.compile(model)

    callback_list = []

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.CUDAMetricsCallback())

    trainer = L.Trainer(
        accelerator=HPUAccelerator() if args.device_type == "hpu" else "auto",
        devices="auto",
        strategy=HPUDeepSpeedStrategy(
            stage=args.deepspeed_stage,
            # offload_optimizer=cfg.offload_optimizer,
            # offload_parameters=cfg.offload_parameters,
            # remote_device=cfg.offload_device,
            # offload_params_device=cfg.offload_device,
            # offload_optimizer_device=cfg.offload_device,
            # nvme_path=cfg.nvme_path,
            logging_batch_size_per_gpu=1,  # cfg.batch_size,
            # partition_activations=cfg.partition_activations,
            cpu_checkpointing=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            pin_memory=True,
            contiguous_memory_optimization=False,
            process_group_backend="hccl"
            # add the option to load a config from json file with more deepspeed options
            # note that if supplied all defaults are ignored - model settings defaults this arg to None
            # config=cfg.deepspeed_cfg_file
        )
        if args.strategy == "deepspeed"
        else HPUParallelStrategy(bucket_cap_mb=125, gradient_as_bucket_view=True, static_graph=True),
        callbacks=callback_list,
        accumulate_grad_batches=1,
        precision="bf16-mixed" if args.strategy == "deepspeed" else "16-mixed",  # 16,
        max_epochs=args.max_epochs,
        num_nodes=1,
        check_val_every_n_epoch=5000,
        val_check_interval=50,
        log_every_n_steps=10,
        limit_val_batches=10,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        plugins=[DeepSpeedPrecisionPlugin(precision="bf16-mixed")] if args.strategy == "deepspeed" else None,
    )

    trainer.fit(model, train_loader)

    context = "Friends of my soul"  # Prime with something
    import os

    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    x = train_dataset.to_tokens(context, "hpu")
    y = model.generate(x, max_new_tokens=20, temperature=1.0, do_sample=True, top_k=3)


if __name__ == "__main__":
    L.seed_everything(42)

    parser = ArgumentParser()
    # parser = L.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--device_type", default="hpu", type=str)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--compile", default=None, choices=[None, "dynamo"])
    parser.add_argument("--implementation", default="mingpt", choices=["mingpt", "nanogpt"])
    parser.add_argument("--strategy", default="deepspeed", choices=["deepspeed", "fsdp_native"])
    parser.add_argument("--deepspeed_stage", default=2, choices=[1, 2, 3])
    parser.add_argument("--max_steps", default=140, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    args = parser.parse_args()

    main(args)
