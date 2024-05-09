# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company

import argparse
import pickle

import deepspeed
import torch
from mingpt.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

set_seed(3407)


def add_argument():
    parser = argparse.ArgumentParser(description="minGPT")

    # Intel Gaudi
    parser.add_argument(
        "--use_hpu", default=True, action="store_true", help="use Intel Gaudi for training"
    )

    # cuda
    parser.add_argument(
        "--with_cuda", default=False, action="store_true", help="use CPU in case there's no GPU support"
    )

    # train
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)")

    parser.add_argument("--steps", default=2000, type=int, help="number of training steps")

    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")

    parser.add_argument("--log-interval", type=int, default=100, help="output logging information at a given interval")

    parser.add_argument(
        "--activation-checkpoint", default=False, action="store_true", help="Enable activation checkpoint"
    )

    parser.add_argument(
        "--dump-memory", default=False, action="store_true", help="Dump the memory usage during training"
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


args = add_argument()

if hasattr(args, "use_hpu") and args.use_hpu:
    use_hpu = True
    dist_backend = "hccl"
    init_method = "env://"
    deepspeed.init_distributed(dist_backend=dist_backend, init_method=init_method)
else:
    deepspeed.init_distributed()
    use_hpu = False


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000  # ...

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = "test" if h % 4 == 0 else "train"  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1
        return x, y


# print an example instance of the dataset
train_dataset = SortDataset("train")
test_dataset = SortDataset("test")
x, y = train_dataset[0]
for a, b in zip(x, y):
    print(int(a), int(b))

# create a GPT instance
from mingpt.model import GPT  # noqa: E402

model_config = GPT.get_default_config()
model_config.model_type = "gpt-nano"
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
if args.activation_checkpoint:
    model_config.activation_checkpoint = True
    # every number of layers/blockes will trigger a checkpoiting
    model_config.checkpoint_num_layers = 1
else:
    model_config.activation_checkpoint = False
model = GPT(model_config)

# wrap it in deepspeed
parameters = filter(lambda p: p.requires_grad, model.parameters())
# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_dataset
)

if hasattr(args, "use_hpu") and args.use_hpu:
    # create a HPU device
    device = torch.device("hpu")
else:
    device = model_engine.local_rank

if args.activation_checkpoint:
    deepspeed.checkpointing.configure(
        None, deepspeed_config=args.deepspeed_config, num_checkpoints=model_config.checkpoint_num_layers
    )

fp16 = model_engine.fp16_enabled()
print(f"fp16={fp16}")

# create a Trainer object
from mingpt.trainer import Trainer  # noqa: E402

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster
train_config.max_iters = args.steps
train_config.num_workers = 0
trainer = Trainer(
    train_config, model_engine, train_dataset, ds_enabled=True, use_hpu=use_hpu, dump_mem=args.dump_memory
)


def batch_end_callback(trainer):
    if trainer.iter_num % args.log_interval == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


trainer.set_callback("on_batch_end", batch_end_callback)

trainer.run()

print("Evaluating the trained model ...")
# now let's perform some evaluation
model_engine.eval()


def eval_split(trainer, split, max_batches):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    n = train_dataset.length  # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(inp, n, do_sample=False)  # using greedy argmax, not sampling
        sol_candidate = cat[:, n:]  # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (sol == sol_candidate).all(1).cpu()  # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 3:  # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print(
                    "GPT claims that %s sorted is %s but gt is %s"
                    % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                )
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100 * rt.mean()))
    return rt.sum()


# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, "train", max_batches=50)
    test_score = eval_split(trainer, "test", max_batches=50)

# let's run a random given sequence through the model as well
n = train_dataset.length  # naugy direct access shrug
inp = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(device)
assert inp[0].nelement() == n
with torch.no_grad():
    cat = model.generate(inp, n, do_sample=False)
sol = torch.sort(inp[0])[0]
sol_candidate = cat[:, n:]
print("input sequence  :", inp.tolist())
print("predicted sorted:", sol_candidate.tolist())
print("gt sort         :", sol.tolist())
print("matches         :", bool((sol == sol_candidate).all()))
