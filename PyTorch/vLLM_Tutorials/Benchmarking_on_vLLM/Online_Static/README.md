# Static Online Serving Examples
This folder contains scripts and configuration files to enable a user to run automated inference on different models on vLLM in online serving mode.

These scripts are designed to be portable and runnable on any Gaudi machine with minimal pre-requisites. Each folder contains :
- Recipe to build Gaudi container image.
- Recipe for starting vLLM server on Gaudi.
- Perf Test script to run a user-scale request on vLLM server with a report of the results.
- The parameters of the vLLM server are pre-configured in the env files and server_cmd.sh scripts.
- To test using a custom input, ensure that the context length (input + output tokens) does not exceed 2K or 4K, respectively, for this tutorial.

## Quick Start
To run these models on your Gaudi machine:
1) Clone this repository.
2) Choose the model directory and cd into it.
3) Edit the docker_envfile.env and enter your HF_TOKEN in the placeholder variable.
4) Run command `sudo ./run.sh` which will build the Docker container for the vLLM server and load the particular model you have chosen in Step 2.
5) Wait ~15 minutes or more for the server to start up and warmup. Ignore the `pulsecheck   | No successful response. HTTP status code: 000. Retrying in 5 seconds...` messages in the meantime. 
6) When server is finally ready for serving, it will say 
`Application Startup Complete.
INFO:     Uvicorn running on http://0.0.0.0.:8000`
7) (Optional) User can run the `./perftest.sh` script in a separate terminal to run a quick benchmark to get some metrics like example below:
<pre>
============ Serving Benchmark Result ============

Successful requests:               320
Benchmark duration (s):            58.27
Total input tokens:                1199854
Total generated tokens:            40960
Request throughput (req/s):        5.49
Output token throughput (tok/s):   702.88
Total Token throughput (tok/s):    21292.51

---------------Time to First Token----------------

Mean TTFT (ms):                    2294.10
Median TTFT (ms):                  2282.69
P90 TTFT (ms):                     3772.42

-----Time per Output Token (excl. 1st token)------

Mean TPOT (ms):                    27.79
Median TPOT (ms):                  27.72
P90 TPOT (ms):                     39.37

---------------Inter-token Latency----------------

Mean ITL (ms):                     27.77
Median ITL (ms):                   13.21
P90 ITL (ms):                      17.35

==================================================
</pre>

### Important Files 

Here is a list of important files in each model directory:

|File name| Description|
|:--------|------------|
|run.sh |Main launcher script|
|docker_envfile.env |File containing environment variables needed for the chosen model |
|Dockerfile-1.20.0-xxxx| Dockerfile used by run.sh to create the docker container |
|perftest.sh |Simple benchmark script for obtaining metrics
|*.csv |File containing inputs for the client script (perftest.sh)|
