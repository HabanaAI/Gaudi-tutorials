# Copyright (c) 2025 Habana Labs, Ltd. an Intel Company. SPDX-License-Identifier: Apache-2.0
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

import numpy as np
import time
from tritonclient.utils import *
import tritonclient.http as httpclient
import argparse


def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    prompt = ["I am", "you are"]
    text_obj = np.array(prompt, dtype="object")  # .reshape((-1, 1))

    input_text = httpclient.InferInput("INPUT0", text_obj.shape, np_to_triton_dtype(text_obj.dtype))
    input_text.set_data_from_numpy(text_obj)

    for _model in model_name:
        print(f"----- Processing: model {_model} ----- ")
        output_text = httpclient.InferRequestedOutput("OUTPUT0")
        start = time.time()
        try:
            query_response = client.infer(model_name=_model, inputs=[input_text], outputs=[output_text])
        except:
            print(f"Model {_model} not found! skipping")
            continue

        end = time.time()

        print(f"-----Stats: model {_model} ----- ")
        print(f"http client out: {output_text}")
        print(f"Output: {query_response.as_numpy('OUTPUT0')}")
        print(f"Time taken: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="comma-separated model names",
        default="llama2",
        type=lambda t: [s.strip() for s in t.split(",")],
    )
    args = parser.parse_args()
    main(args.model_name)
