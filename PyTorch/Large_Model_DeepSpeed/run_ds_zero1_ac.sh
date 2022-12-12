#!/bin/bash

deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero1_ac.json --use_hpu --activation-checkpoint $@
