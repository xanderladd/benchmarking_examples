#!/bin/bash


source ./experiment_design.txt
IFS=',' read -r -a offspringArray <<< "$offspring"
IFS=',' read -r -a nodeArray <<< "$N"
IFS=',' read -r -a samples <<< "$samples"


if [ "$method" = "neuroGPU" ]; then
    launcher=launch_neuroGPU.sh
elif [ "$method" = "core_neuron" ]; then
    launcher=launch_core_neuron.sh
else
    echo "exp type ${method} did not match supported exp types: neuroGPU, core_neuron"
    exit 1
fi

if [ "${#offspringArray[@]}" -eq "${#nodeArray[@]}" ] \
&& [ "${#offspringArray[@]}" -eq "${#samples[@]}" ]; then
    for i in "${!nodeArray[@]}"; do
        samples="${samples[i]}"
        nnodes="${nodeArray[i]}"
        offspring_trial="${offspringArray[i]}"       
        echo launching @ "$nnodes" nodes and "$offspring_trial" offspring and "$samples" samples from `pwd` 
        sh $launcher $offspring_trial $samples $nnodes
    done
else
    echo experimental design requires nnodes, samples, and offspring size to be same length
fi
exit 0


# bsub -nnodes 1  -W 5 -P nro106  -q debug  -Ep "sh test_post_exec.sh" "sh sleep.sh"