PT_HPU_LAZY_ACC_PAR_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true python gaudi_spawn.py --use_deepspeed --world_size 8 run_localGPT.py --device_type hpu --temperature 0.7 --top_p 0.95
