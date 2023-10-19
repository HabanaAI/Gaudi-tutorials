import time

import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import Callback
elif module_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import Callback
# from lightning import LightningModule, Trainer
# from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info


class CUDAMetricsCallback(Callback):
    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")

    def root_gpu(self, trainer: "Trainer") -> int:
        return trainer.strategy.root_device.index
