from typing import Any, Optional, Tuple
from urllib.request import urlopen

import lightning as L
import torch
import torch._dynamo

import mingpt.model
from lightning_gpt import bench, data, models


class GPTBench(bench.Bench):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_workers = 0
        self.batch_size = 64
        self.max_epochs = 2
        self.precision = 32  # not used
        self.model_type = "gpt-micro"
        self.num_runs = 2

    def create(self) -> Tuple[mingpt.model.GPT, torch.utils.data.DataLoader]:
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.suppress_errors = True

        with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as f:
            text = f.read()

        dataset = data.CharDataset(text, block_size=128)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        model = models.MinGPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model.mingpt, dataloader

    def train(self, model: mingpt.model.GPT, dataset: torch.utils.data.Dataset) -> Optional[float]:
        from mingpt.trainer import Trainer

        train_config = Trainer.get_default_config()
        train_config.device = "cuda"
        train_config.learning_rate = 3e-4
        train_config.max_iters = len(dataset) / self.batch_size * self.max_epochs  # type: ignore
        train_config.num_workers = self.num_workers
        train_config.batch_size = self.batch_size
        train_config.grad_norm_clip = 1.0
        trainer = Trainer(train_config, model, dataset)

        trainer.run()

        final_loss = trainer.loss
        return final_loss.item() if final_loss is not None else None

    def run(self) -> None:
        model, dataloader = self.create()

        self.run_benchmark(name="nocompile", fn=self.train, args=(model, dataloader), num_runs=self.num_runs)

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark("compile", self.train, args=(model, dataloader), num_runs=self.num_runs)


app = L.LightningApp(GPTBench(cloud_compute=L.CloudCompute("gpu-fast")))

# app = L.LightningApp(
#     bench.BenchRun(
#         GPTBench,
#         num_nodes=1,
#         cloud_compute=L.CloudCompute("gpu-fast"),
#     )
# )
