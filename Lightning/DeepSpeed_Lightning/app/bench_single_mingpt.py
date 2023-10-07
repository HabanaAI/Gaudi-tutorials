from typing import TYPE_CHECKING, Any, Optional, Tuple
from urllib.request import urlopen

import lightning as L

if TYPE_CHECKING:
    from lightning import LightningModule

import torch
import torch._dynamo
from torch.utils.data import DataLoader

from lightning_gpt import bench, data, models


class GPTBench(bench.Bench):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_workers = 0
        self.batch_size = 64
        self.max_epochs = 2
        self.precision = 32
        self.model_type = "gpt-micro"
        self.num_runs = 2

    def create(self) -> Tuple[models.MinGPT, torch.utils.data.DataLoader]:
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.suppress_errors = True

        with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as f:
            text = f.read()

        dataset = data.CharDataset(text, block_size=128)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        model = models.MinGPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model, dataloader

    def train(
        self,
        model: "LightningModule",
        dataloader: torch.utils.data.DataLoader,
    ) -> Optional[float]:
        self._check_precision()
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            accelerator="cuda",
            devices=1,
            precision=self.precision,  # type: ignore
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            use_distributed_sampler=False,
            num_sanity_val_steps=0,
            reload_dataloaders_every_n_epochs=1000,
        )

        trainer.fit(model, dataloader)
        final_loss = trainer.fit_loop.running_loss.last()
        return final_loss.item() if final_loss is not None else None

    def run(self) -> None:
        model, dataloader = self.create()

        self.run_benchmark(name="nocompile", fn=self.train, args=(model, dataloader), num_runs=self.num_runs)

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark("compile", self.train, args=(model, dataloader), num_runs=self.num_runs)


app = L.LightningApp(GPTBench(cloud_compute=L.CloudCompute("gpu-fast")))
