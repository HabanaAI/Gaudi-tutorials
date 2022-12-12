#!/bin/bash

deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero2.json --use_hpu $@
