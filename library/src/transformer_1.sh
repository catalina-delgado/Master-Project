#!/bin/bash
CUDA_VISIBLE_DEVICES=1 nohup python transformer_1.py > runs/salida_train_$(date +'%Y-%m-%d_%H-%M-%S').log 2>&1 &