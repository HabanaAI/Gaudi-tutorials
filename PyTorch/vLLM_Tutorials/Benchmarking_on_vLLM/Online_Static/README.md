# Static Online Serving Examples
This folder contains scripts and configuration files to enable a user to run automated inference on different models on vLLM in online serving mode.

These scripts are designed to be portable and runnable on any Gaudi machine with minimal pre-requisites. This folder contains :
- Recipe for starting vLLM server.
- Recipe to build container image.
- Perf Test script to run a user-scale request on vLLM server with a report of the results.

## Quick Start
To run these models on your Gaudi machine:
1) Clone this repository.
2) Choose the model directory and cd into it.
3) Edit the docker_envfile.env and enter your HF_TOKEN in Line 26.
3) Launch run.sh which will start vLLM server and load the particular model.
4) (Optional) User can run the perftest.sh script in a separate terminal to run a quick benchmark to get some metrics.

### Important Files 

|File name| Description|
|:--------|------------|
|run.sh |Main launcher script|
|docker_envfile.env |File containing environment variables needed for the chosen model |
|perftest.sh |Simple benchmark script for obtaining metrics
|*.csv |File containing inputs for the client script (perftest.sh)
|

