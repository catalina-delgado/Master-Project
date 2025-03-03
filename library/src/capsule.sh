#!/bin/bash
CUDA_VISIBLE_DEVICES=7 nohup python capsule_model.py > runs/salida_train_$(date +'%Y-%m-%d_%H-%M-%S').log 2>&1 &