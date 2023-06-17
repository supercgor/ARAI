#bin/bash
a=`ls model/pretrain/ -t | head -n 1`
echo $a
ml
tensorboard --logdir=model/pretrain/${a}/runs/ --port=6787