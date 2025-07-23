

import os


os.system(" python hltrainval_mtl2.py --data-dir ./data/soybean_plot_xml --dataset wec --model tasselnetv2plus --exp tasselnetv2plus-soybeanPlot-20-0.5-512-8-v2pp  --data-list ./data/soybean_plot_xml/train.txt --data-val-list ./data/soybean_plot_xml/test.txt --restore-from model_best.pth.tar --image-mean 0.23324092 0.22439253 0.20940149 --image-std 0.17052431 0.1618571 0.15176316 --input-size 64 --output-stride 8 --resize-ratio 0.5 --optimizer sgd --milestones 200 400 --batch-size 1 --crop-size 512 512 --learning-rate 1e-2 --num-epochs 500 --num-workers 0 --print-every 1 --random-seed 2020 --val-every 1 --evaluate-only --save-output")#wec

#