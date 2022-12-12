#!/bin/bash

deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config.json --use_hpu $@
