#!/bin/bash

EXPERIMENTAL_WEIGHT_SHARING=0 deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero1_ac.json --use_hpu --activation-checkpoint $@
