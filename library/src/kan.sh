#!/bin/bash
CUDA_VISIBLE_DEVICES=6 nohup python kan_model.py > runs/salida_train_$(date +'%Y-%m-%d_%H-%M-%S').log 2>&1 &