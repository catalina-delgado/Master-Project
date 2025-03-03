#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python swinTransformer_model.py > runs/salida_train_$(date +'%Y-%m-%d_%H-%M-%S').log 2>&1 &