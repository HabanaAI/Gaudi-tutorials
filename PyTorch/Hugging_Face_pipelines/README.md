# Using Hugging Face Pipelines on Intel&reg; Gaudi&reg; 2 AI Processor
This section contains examples of how to easily incorporate Hugging Face Pipelines with Intel Gaudi and the Optimum for Intel Gaudi (aka Optimum Habana) library. Hugging Face pipelines take advantage of the Hugging Face Tasks in transformer models, such as text generation, translation, and question answering and more. You can read more about Hugging Face pipelines on their main page [here](https://huggingface.co/docs/transformers/main_classes/pipelines)

To ensure that the associated pipline is running on the Intel Gaudi processor, you must set `device_type="hpu"` in the pipeline assignment. 

the basic steps to setup a pipeline for Intel Gaudi are:
* Get access to an Intel Gaudi node and install the associcated version of the Hugging Face Optimum for Intel Gaudi library.  See the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) for more information on basic setup.
* Choose the Hugging Face [task](https://huggingface.co/tasks) and model that you want to use.  Note that the list of optimized Intel Gaudi models can be found [here](https://github.com/huggingface/optimum-habana?tab=readme-ov-file#validated-models)
* Setup the pipeline with the assocaited tasks, model and device set to **"hpu"**; which allows the pipeline to run on Intel Gaudi.

The Jupyter Notebooks in the folder will show different examples of how to setup and run the pipelines with the following tasks
* translation
* image-classification
* text-generation
* visual-question-answering

