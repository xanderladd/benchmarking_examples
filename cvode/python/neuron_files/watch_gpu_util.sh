#!/bin/bash


while true; 
do nvidia-smi --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory --format=csv >> gpu_utillization.log; sleep 1; 
done
