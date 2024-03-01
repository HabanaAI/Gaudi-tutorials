import functools
import warnings
from typing import Any, Optional, Tuple

import torch.optim
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningModule
    from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
elif module_available("pytorch_lightning"):
    from lightning_habana.strategies.deepspeed import _DEEPSPEED_AVAILABLE
    from pytorch_lightning import LightningModule

from lightning_utilities.core.overrides import is_overridden

import mingpt.model
import mingpt.trainer
import nanogpt.model
from mingpt.utils import CfgNode

MINGPT_PRESETS = {
    # names follow the huggingface naming conventions
    # GPT-1
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    # GPT-2 configs
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    "gpt2-xxl": dict(n_layer=96, n_head=25, n_embd=1600),  # 2951M params
    "gpt2-xxxl": dict(n_layer=100, n_head=30, n_embd=1920),  # 4426M params
    "gpt2-4xl": dict(n_layer=190, n_head=30, n_embd=1920),  # 8409M params
    # Gophers
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    # (there are a number more...)
    # I made these tiny models up
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}


class MinGPT(LightningModule):
    mingpt: mingpt.model.GPT

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        model_type: Optional[str] = "gpt2",
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.build_mingpt_configs()
        if not is_overridden("configure_sharded_model", self, LightningModule):
            self.mingpt = mingpt.model.GPT(self.mingpt_config)

    def build_mingpt_configs(self) -> None:
        params = [
            self.hparams.n_layer,
            self.hparams.n_head,
            self.hparams.n_embd,
        ]

        params_given = all([el is not None for el in params])
        some_params_given = any([el is not None for el in params])

        if some_params_given and not params_given:
            raise ValueError(
                "Please provide all values for n_layer, n_head, and n_embd, or just model_type."
                f"Got n_layer={self.hparams.n_layer}, n_head={self.hparams.n_head}, "
                f"and n_embd={self.hparams.n_embd}."
            )

        if not params_given:
            # We take ownership of presets over minGPT here
            preset = MINGPT_PRESETS[self.hparams.model_type]
            self.hparams.update(preset)
            self.hparams.model_type = None

        self.mingpt_config = mingpt.model.GPT.get_default_config()
        self.merge_with_hparams(self.mingpt_config)

        self.mingpt_trainer_config = mingpt.trainer.Trainer.get_default_config()
        self.merge_with_hparams(self.mingpt_trainer_config)

    def merge_with_hparams(self, config: CfgNode) -> None:
        keys = set(config.to_dict().keys())
        hparams = {k: v for k, v in self.hparams.items() if k in keys}
        config.merge_from_dict(hparams)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.mingpt(idx, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.mingpt.configure_optimizers(self.mingpt_trainer_config)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        return self.mingpt.generate(idx, max_new_tokens, temperature, do_sample, top_k)


class NanoGPT(LightningModule):
    nanogpt: nanogpt.model.GPT

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        model_type: Optional[str] = "gpt2",
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        dropout: float = 0.1,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.build_nanogpt_configs()
        if not is_overridden("configure_sharded_model", self, LightningModule):
            self.nanogpt = nanogpt.model.GPT(self.nanogpt_config)

    def build_nanogpt_configs(self) -> None:
        params = [
            self.hparams.n_layer,
            self.hparams.n_head,
            self.hparams.n_embd,
        ]

        params_given = all([el is not None for el in params])
        some_params_given = any([el is not None for el in params])

        if some_params_given and not params_given:
            raise ValueError(
                "Please provide all values for n_layer, n_head, and n_embd, or just model_type."
                f"Got n_layer={self.hparams.n_layer}, n_head={self.hparams.n_head}, "
                f"and n_embd={self.hparams.n_embd}."
            )

        if not params_given:
            # We take ownership of presets over minGPT here
            preset = MINGPT_PRESETS[self.hparams.model_type]
            self.hparams.update(preset)
            self.hparams.model_type = None

        self.nanogpt_config = nanogpt.model.GPTConfig()
        self.merge_with_hparams(self.nanogpt_config)

        self.nanogpt_trainer_config = mingpt.trainer.Trainer.get_default_config()
        self.merge_with_hparams(self.nanogpt_trainer_config)

    def merge_with_hparams(self, config: CfgNode) -> None:
        for k, v in self.hparams.items():
            if hasattr(config, k):
                setattr(config, k, v)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.nanogpt(idx, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.nanogpt.configure_optimizers(
            weight_decay=self.nanogpt_trainer_config.weight_decay,
            learning_rate=self.nanogpt_trainer_config.learning_rate,
            betas=self.nanogpt_trainer_config.betas,
            device_type=self.hparams.device_type,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None
    ) -> torch.Tensor:
        return self.nanogpt.generate(idx, max_new_tokens, temperature, top_k)


class DeepSpeedMinGPT(MinGPT):
    # TODO: activation checkpointing (requires overriding forward)
    def __init__(self, fused_adam: bool = True, offload: bool = False, **kwargs: Any):
        if fused_adam and offload:
            raise RuntimeError(
                "Cannot use FusedAdam and CPUAdam at the same time! "
                "Please set either `fused_adam` or `offload` to False."
            )

        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = super().configure_optimizers()
        return _get_deepspeed_optimizer(
            optimizer,
            fused_adam=self.hparams.fused_adam,
            cpu_offload=self.hparams.offload,
            learning_rate=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

    def configure_sharded_model(self) -> None:
        self.mingpt = mingpt.model.GPT(self.mingpt_config)


class FSDPMinGPT(MinGPT):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        _register_gpt_strategy()

    def configure_optimizers(self) -> torch.optim.AdamW:
        return _get_fsdp_optimizers(
            self.trainer.model,
            weight_decay=self.mingpt_trainer_config.weight_decay,
            learning_rate=self.mingpt_trainer_config.learning_rate,
            betas=self.mingpt_trainer_config.betas,
        )


class DeepSpeedNanoGPT(NanoGPT):
    # TODO: activation checkpointing (requires overriding forward)
    def __init__(self, fused_adam: bool = True, offload: bool = False, **kwargs: Any):
        if fused_adam and offload:
            raise RuntimeError(
                "Cannot use FusedAdam and CPUAdam at the same time! "
                "Please set either `fused_adam` or `offload` to False."
            )

        kwargs["device_type"] = "cuda" if fused_adam or kwargs.pop("device_type", "cpu") == "cuda" else "cpu"

        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = super().configure_optimizers()

        return _get_deepspeed_optimizer(
            optimizer,
            fused_adam=self.hparams.device_type == "cuda",
            cpu_offload=self.hparams.offload,
            learning_rate=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

    def configure_sharded_model(self) -> None:
        self.nanogpt = nanogpt.model.GPT(self.nanogpt_config)


class FSDPNanoGPT(NanoGPT):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        _register_gpt_strategy()

    def configure_optimizers(self) -> torch.optim.AdamW:
        return _get_fsdp_optimizers(
            self.trainer.model,
            weight_decay=self.nanogpt_trainer_config.weight_decay,
            learning_rate=self.nanogpt_trainer_config.learning_rate,
            betas=self.nanogpt_trainer_config.betas,
        )


def _register_gpt_strategy() -> None:
    from lightning.pytorch.strategies import StrategyRegistry
    from lightning.pytorch.strategies.fully_sharded_native import (
        DDPFullyShardedNativeStrategy,
    )
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if "fsdp-gpt" in StrategyRegistry.available_strategies():
        return

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={nanogpt.model.Block, mingpt.model.Block}
    )

    StrategyRegistry.register(
        name="fsdp-gpt",
        strategy=DDPFullyShardedNativeStrategy,
        description="FSDP strategy with memory optimizations enabled for GPT large scale pretraining.",
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=[nanogpt.model.Block, mingpt.model.Block],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )


def _get_deepspeed_optimizer(
    optimizer: torch.optim.Optimizer,
    cpu_offload: bool,
    fused_adam: bool,
    learning_rate: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    optim_groups = optimizer.param_groups

    # import locally because of https://github.com/Lightning-AI/lightning/pull/15610
    if cpu_offload and _DEEPSPEED_AVAILABLE:
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam(optim_groups, lr=learning_rate, betas=betas)

    elif fused_adam and _DEEPSPEED_AVAILABLE:
        # from deepspeed.ops.adam import FusedAdam

        # return FusedAdam(optim_groups, lr=learning_rate, betas=betas)
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW

        return FusedAdamW(optim_groups, lr=learning_rate)
        # return torch.optim.Adam(optim_groups, lr=learning_rate)

    elif fused_adam or cpu_offload:
        warnings.warn(
            "Deepspeed is not available, so cannot enable fused adam or cpu offloaded adam. Please install deepspeed!"
        )

    return optimizer


def _get_fsdp_optimizers(
    model: torch.nn.Module, learning_rate: float, weight_decay: float, betas: Tuple[float, float]
) -> torch.optim.AdamW:
    # fsdp only supports a single parameter group and requires the parameters from the already wrapped model
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
