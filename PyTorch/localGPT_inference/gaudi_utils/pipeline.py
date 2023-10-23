#Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

import copy
import torch
import os
from pathlib import Path

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.generation.utils import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import habana_frameworks.torch.hpu as torch_hpu
from optimum.habana.utils import set_seed

from huggingface_hub import list_repo_files, snapshot_download
from transformers.utils import is_offline_mode, is_safetensors_available


def get_optimized_model_name(config):
    for model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
        if model_type == config.model_type:
            return model_type

    return None


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    return get_optimized_model_name(config) is not None


def get_repo_root(model_name_or_path, local_rank=-1):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                print("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        allow_patterns = ["*.bin"]
        # If the model repo contains any .safetensors file and
        # safetensors is installed, only download safetensors weights
        if is_safetensors_available():
            if any(".safetensors" in filename for filename in list_repo_files(model_name_or_path)):
                allow_patterns = ["*.safetensors"]

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
        )


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        super().__init__()
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return self.stop_token_id in input_ids[0]


class GaudiTextGenerationPipeline:
    def __init__(self, model_name_or_path=None, **kwargs):
        self.task = "text-generation"
        self.device = "hpu"

        # Tweak generation so that it runs faster on Gaudi
        adapt_transformers_to_gaudi()
        set_seed(27)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        model_dtype = torch.bfloat16
        get_repo_root(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=model_dtype)
        model = model.eval().to(self.device)
        is_optimized = model_is_optimized(model.config)
        self.model = wrap_in_hpu_graph(model)

        # Used for padding input to fixed length
        self.tokenizer.padding_side = "left"
        self.max_padding_length = kwargs.get("max_padding_length", self.model.config.max_position_embeddings)
        
        # Some models like GPT2 do not have a PAD token so we have to set it if necessary
        if self.model.config.model_type == "llama":
            self.model.generation_config.pad_token_id = 0
            self.model.generation_config.bos_token_id = 1
            self.model.generation_config.eos_token_id = 2
            self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id
            self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.decode(self.tokenizer.pad_token_id)
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            self.tokenizer.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        # Edit generation configuration based on input arguments
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.max_new_tokens = kwargs.get("max_new_tokens", 100)
        self.generation_config.use_cache = kwargs.get("use_kv_cache", True)
        self.generation_config.static_shapes = is_optimized
        self.generation_config.do_sample = kwargs.get("do_sample", False)
        self.generation_config.num_beams = kwargs.get("num_beams", 1)
        self.generation_config.temperature = kwargs.get("temperature", 1.0)
        self.generation_config.top_p = kwargs.get("top_p", 1.0)
        self.generation_config.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.generation_config.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.generation_config.bad_words_ids = None
        self.generation_config.force_words_ids = None

        # Define stopping criteria based on eos token id
        self.stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(self.generation_config.eos_token_id)])

    def __call__(self, prompt: str):
        model_inputs = self.tokenizer.encode_plus(prompt, return_tensors="pt", max_length=self.max_padding_length, padding="max_length", truncation=True)

        for t in model_inputs:
            if torch.is_tensor(model_inputs[t]):
                model_inputs[t] = model_inputs[t].to(self.device)

        output = self.model.generate(**model_inputs, generation_config=self.generation_config, lazy_mode=True, hpu_graphs=True, profiling_steps=0, profiling_warmup_steps=0, stopping_criteria=self.stopping_criteria).cpu()

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        del output, model_inputs

        return [{"generated_text": output_text}]

    def compile_graph(self):
        for _ in range(3):
            self("Here is my prompt")
        torch_hpu.synchronize()