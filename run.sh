#!/bin/bash

# Define an array of commands
commands=(
  "python main.py --dataset seq-cifar10 --model lord --buffer_size 500 --load_best_args"
  "python main.py --dataset seq-cifar100 --model lord --buffer_size 500 --load_best_args"
  "python main.py --dataset seq-tinyimg --model lord --buffer_size 500 --load_best_args"


)

# Execute each command in the array
for cmd in "${commands[@]}"; do
  echo "Running command: $cmd"
  eval $cmd
  echo "Command finished"
done

echo "All commands executed."