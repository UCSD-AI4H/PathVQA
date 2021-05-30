name=$pvqa_ignore

# Save logs and models under snap/vqa; make backup.
output=snap/pvqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

PYTHONPATH=$PYTHONPATH:./src \
python3 src/tasks/pvqa_ignore.py \
    --train train --valid val  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 \
    --optim bert --lr 5e-5 --epochs 30 \
    --tqdm --output $output ${@:3}