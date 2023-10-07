import lightning as L
import pytest
import torch

import mingpt
import nanogpt
from lightning_gpt import models


def test_mingpt_vs_lightning_mingpt():
    vocab_size = 65
    block_size = 128
    model_type = "gpt-mini"

    x = torch.randint(0, vocab_size, (1, 12))

    mingpt_config = mingpt.model.GPT.get_default_config()
    mingpt_config.vocab_size = vocab_size
    mingpt_config.block_size = block_size
    mingpt_config.model_type = model_type

    mingpt_model = mingpt.model.GPT(mingpt_config)

    lit_model = models.MinGPT(vocab_size=vocab_size, block_size=block_size, model_type=model_type)

    for target_param, param in zip(lit_model.parameters(), mingpt_model.parameters()):
        target_param.data.copy_(param.data)

    mingpt_model.eval()
    lit_model.eval()

    mingpt_y, _ = mingpt_model(x)
    lit_y, _ = lit_model(x)

    torch.testing.assert_close(mingpt_y, lit_y)


def test_nanogpt_vs_lightning_nanogpt():
    vocab_size = 65
    block_size = 128
    model_type = "gpt-mini"

    x = torch.randint(0, vocab_size, (1, 12))

    nanogpt_config = nanogpt.model.GPTConfig(**models.MINGPT_PRESETS[model_type])
    nanogpt_config.vocab_size = vocab_size
    nanogpt_config.block_size = block_size
    nanogpt_config.model_type = model_type

    nanogpt_model = nanogpt.model.GPT(nanogpt_config)

    lit_model = models.NanoGPT(vocab_size=vocab_size, block_size=block_size, model_type=model_type)

    for target_param, param in zip(lit_model.parameters(), nanogpt_model.parameters()):
        target_param.data.copy_(param.data)

    nanogpt_model.eval()
    lit_model.eval()

    nanogpt_y, _ = nanogpt_model(x)
    lit_y, _ = lit_model(x)

    torch.testing.assert_close(nanogpt_y, lit_y)


def _get_dummy_data(vocabsize):
    data = [[torch.randint(0, vocabsize, (12,)) for _ in range(2)] for _ in range(10)]
    return torch.utils.data.DataLoader(data)


def _get_minimal_gpt_config():
    return {"vocab_size": 65, "block_size": 128, "model_type": "gpt-nano"}


@pytest.mark.parametrize(
    "model_cls",
    [
        models.MinGPT,
        models.DeepSpeedMinGPT,
        models.FSDPMinGPT,
        models.NanoGPT,
        models.DeepSpeedNanoGPT,
        models.FSDPNanoGPT,
    ],
)
def test_model_instatiation_base_strategy(tmpdir, model_cls):
    trainer = L.pytorch.Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=tmpdir,
    )
    gpt_config = _get_minimal_gpt_config()

    if "deepspeed" in model_cls.__qualname__.lower():
        gpt_config.update(fused_adam=False, offload=False)
    mingpt = model_cls(**gpt_config)
    dataloader_train = _get_dummy_data(gpt_config["vocab_size"])
    trainer.fit(mingpt, dataloader_train)


@pytest.mark.parametrize("model_cls", [models.DeepSpeedMinGPT, models.DeepSpeedNanoGPT])
def test_model_instantiation_error_deepspeed(model_cls):
    with pytest.raises(
        RuntimeError,
        match="Cannot use FusedAdam and CPUAdam at the same time!"
        " Please set either `fused_adam` or `offload` to False.",
    ):
        model_cls(**_get_minimal_gpt_config(), fused_adam=True, offload=True)
