#!/bin/bash
source .venv/bin/activate
until grep -qE "stage m2 done|Traceback" logs/run_gpu4.log; do sleep 30; done
PYTORCH_ALLOC_CONF=expandable_segments:True python src/run_gpu.py steer > logs/run_gpu3.log 2>&1
echo AFTER_GPU_DONE
