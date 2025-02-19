#!/bin/bash

# Define an array of commands
commands=(
  "CUDA_VISIBLE_DEVICES=1 python main.py --dataset seq-cifar10 --model e2n --buffer_size 200 --load_best_args"


)

# Execute each command in the array
for cmd in "${commands[@]}"; do
  echo "Running command: $cmd"
  eval $cmd
  echo "Command finished"
done

echo "All commands executed."