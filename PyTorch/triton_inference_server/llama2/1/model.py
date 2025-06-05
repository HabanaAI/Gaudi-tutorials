# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import logging

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import numpy as np


import torch
from utils import count_hpu_graphs, initialize_model
from optimum.habana.utils import get_hpu_memory_stats
import os


class habana_args:
    device = "hpu"
    model_name_or_path = "huggyllama/llama-7b"  #'meta-llama/Llama-2-7b-chat-hf'
    token = None
    bf16 = False
    use_hpu_graphs = True
    use_lazy_mode = False
    seed = 42
    use_kv_cache = True
    max_new_tokens = 100
    max_inp_tokens = 1024
    batch_size = -1
    do_sample = True
    num_beams = 1
    num_return_sequences = 1
    profiling_steps = 0
    profiling_warmup_steps = 0
    prompt = "I am"
    local_rank = ""
    world_size = ""
    global_rank = ""
    fp8 = False
    model_revision = "main"
    peft_model = None
    skip_hash_with_views = True
    bad_words = None
    force_words = None
    bucket_size = -1
    trim_logits = True
    attn_softmax_bf16 = True
    limit_hpu_graphs = True
    reuse_cache = True
    kv_cache_fp8 = True
    attn_batch_split = 1
    torch_compile = False
    quant_config = os.getenv("QUANT_CONFIG", "")
    load_quantized_model_with_inc = False
    assistant_model = None
    trust_remote_code = False
    disk_offload = False
    show_graphs_count = False
    local_quantized_inc_model_path = None
    load_quantized_model_with_autogptq = False
    load_quantized_model_with_autoawq = False
    bucket_internal = False
    top_k = None
    penalty_alpha = None
    clear_hpu_graphs_cache = False
    reduce_recompile = False
    use_flash_attention = False
    flash_attention_recompute = False
    flash_attention_causal_mask = False
    flash_attention_fast_softmax = False
    const_serialization_path = ""


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print("Initializing")
        model, assistant_model, tokenizer, generation_config = initialize_model(
            habana_args, logger
        )

        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

        input_tokens = tokenizer.batch_encode_plus(
            [habana_args.prompt], return_tensors="pt", padding=True
        )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(habana_args.device)
        outputs = self.model.generate(
            **input_tokens,
            generation_config=generation_config,
            lazy_mode=True,
            hpu_graphs=habana_args.use_hpu_graphs,
            profiling_steps=habana_args.profiling_steps,
            profiling_warmup_steps=habana_args.profiling_warmup_steps,
        ).cpu()
        print("prompt test:")
        prompt_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(prompt_list[0].encode("UTF-8"))
        print(prompt_list[0])

        print("Initialize finished")

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            inp = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            print("xxxxxxxxxxx:", inp)
            input_text = [x.decode("utf8") for x in inp.as_numpy()]
            print(input_text, type(input_text[0]))

            input_tokens = self.tokenizer.batch_encode_plus(
                input_text, return_tensors="pt", padding=True
            )
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(habana_args.device)

            outputs = self.model.generate(
                **input_tokens,
                generation_config=self.generation_config,
                lazy_mode=True,
                hpu_graphs=habana_args.use_hpu_graphs,
                profiling_steps=habana_args.profiling_steps,
                profiling_warmup_steps=habana_args.profiling_warmup_steps,
            ).cpu()
            out_0 = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print([x.encode("utf8") for x in out_0])
            # out_0 = ["asdfasdfasdf", "idiot"]

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT0", np.array(out_0, dtype=self.output0_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
