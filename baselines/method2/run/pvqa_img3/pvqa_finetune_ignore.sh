python finetune_main_ignore.py \
  --task pvqa --epochs 180 --start_epoch 0 \
  --lr 0.01 --cos --train train --val val --tfidf \
  --output saved_models/pvqa/ignore \
  --batch_size 64