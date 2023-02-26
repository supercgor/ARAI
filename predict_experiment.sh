#!/bin/bash
. ./.afmvenv/bin/activate
#python3 afm_go-1/main.py --mode test --dataset AFM_3d/all_fileList --model 22-12-04-21/CP90_O0.8845_H0.7233_0.279314.pkl --gpu 0 --worker 6

#python3 afm_go-1/main.py --mode test --dataset AFM_3d/all_fileList --model 22-12-05-16/CP90_O0.8824_H0.7145_0.254364.pkl --gpu 0 --worker 6

#python3 afm_go-1/main.py --mode test --dataset AFM_3d/fileList/prism_valid.filelist --model 22-12-05-16/CP90_O0.8824_H0.7145_0.254364.pkl --gpu 0 --worker 6

#python3 afm_go-1/main.py --mode test --dataset AFM_3d/fileList/prism_valid.filelist --model 22-12-04-21/CP90_O0.8845_H0.7233_0.279314.pkl --gpu 0 --worker 6

#python3 afm_go-1/main.py --mode predict --dataset exp_ice --model 22-12-05-16/CP90_O0.8824_H0.7145_0.254364.pkl --gpu 0 --worker 6

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/basel_fileList --model 22-12-13-14/CP95_O0.8125_H0.6294_0.270068.pkl

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/T_180_220_fileList --model 22-12-13-14/CP95_O0.8125_H0.6294_0.270068.pkl

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/fileList/prism_valid.filelist --model 22-12-13-14/CP95_O0.8125_H0.6294_0.270068.pkl

python3 afm_go-1/main.py --mode predict --gpu 0 --worker 6 --dataset exp_ice --model 22-12-13-14/CP95_O0.8125_H0.6294_0.270068.pkl

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/basel_fileList --model 22-12-15-17/CP95_O0.8083_H0.6272_0.295024.pkl

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/T_180_220_fileList --model 22-12-15-17/CP95_O0.8083_H0.6272_0.295024.pkl

python3 afm_go-1/main.py --mode test --gpu 0 --worker 6 --dataset AFM_3d/fileList/prism_valid.filelist --model 22-12-15-17/CP95_O0.8083_H0.6272_0.295024.pkl

python3 afm_go-1/main.py --mode predict --gpu 0 --worker 6 --dataset exp_ice --model 22-12-15-17/CP95_O0.8083_H0.6272_0.295024.pkl
