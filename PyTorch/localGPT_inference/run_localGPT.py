import warnings

warnings.filterwarnings("ignore")

import logging
import os
import time

import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from constants import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    LLM_BASE_NAME,
    LLM_ID,
    PERSIST_DIRECTORY,
)
from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma


def load_model(device_type, model_id, temperature, top_p, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU, "cpu" for CPU or "hpu" for Gaudi
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f"temperature set to {temperature}, top_p set to {top_p}")
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    process_rank = -1

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif device_type == "hpu":
        from gaudi_utils.pipeline import GaudiTextGenerationPipeline

        pipe = GaudiTextGenerationPipeline(
            model_name_or_path=model_id,
            max_new_tokens=1000,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            do_sample=True,
        )
        pipe.compile_graph()
        process_rank = pipe.get_process_rank()
    else:
        from transformers import GenerationConfig, pipeline

        if (
            device_type.lower() == "cuda"
        ):  # The code supports all huggingface models that ends with -HF or which have a .bin
            # file in their HF repo.
            logging.info("Using AutoModelForCausalLM for full models")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logging.info("Tokenizer loaded")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
            )
            model.tie_weights()
        else:
            logging.info("Using LlamaTokenizer")
            from transformers import LlamaForCausalLM, LlamaTokenizer

            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(model_id)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            generation_config=generation_config,
        )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm, process_rank


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--temperature",
    default=0.2,
    help="Specify the temperature value for text-generation with LLMs",
)
@click.option(
    "--top_p",
    default=0.95,
    help="Specify the top_p value for text-generation with LLMs",
)
def main(device_type, show_sources, temperature, top_p):
    """
    This function implements the information retrieval task.


    1. Loads a huggingface embedding model
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    use_deepspeed = "deepspeed" in os.environ["_"]

    # model info
    model_id = LLM_ID
    model_basename = LLM_BASE_NAME

    # Load model pipeline object and process rank if using deepspeed
    llm, local_rank = load_model(device_type, model_id, temperature, top_p, model_basename=model_basename)

    # Load embeddings object for vectorstore retrieval
    if device_type == "hpu":
        from optimum.habana.sentence_transformers.modeling_utils import adapt_sentence_transformers_to_gaudi

        adapt_sentence_transformers_to_gaudi()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # Load chroma vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

{context}

{history}
Question: {question}
Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    # Initialize langchain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    if use_deepspeed:
        torch.distributed.barrier()

        # Set up distributed FileStore for deepspeed
        store = torch.distributed.FileStore("filestore", int(os.getenv("WORLD_SIZE")))

        # pre-flight run before starting interactive session
        qa("What is this document about?")

        torch.distributed.barrier()

    # Interactive Session
    while True:
        if local_rank in [-1, 0]:
            query = input("\nEnter a query: ")
            if use_deepspeed:
                store.set("query", query)

        if use_deepspeed:
            torch.distributed.barrier()
            query = str(store.get("query"), encoding="utf-8")

        if query == "exit":
            break

        if local_rank in [-1, 0]:
            start_time = time.perf_counter()

        if use_deepspeed:
            torch.distributed.barrier()

        res = qa(query)

        if local_rank in [-1, 0]:
            end_time = time.perf_counter()
            logging.info(f"Query processing time: {end_time-start_time}s")
            answer, docs = res["result"], res["source_documents"]

            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if show_sources:  # this is a flag that you can set to disable showing answers.
                # # Print the relevant sources used for the answer
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")

        if use_deepspeed:
            torch.distributed.barrier()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
