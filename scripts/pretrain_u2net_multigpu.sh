#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 horovodrun -np 6 --output-filename logs/u2net python u2net_train.py