imgv=3
pretask=vq
for seed in 1204 1 2 3 4 5
do
  dir=${pretask}_${seed}_imgv${imgv}
  echo $dir
  PYTHONPATH=$PYTHONPATH:./src\
    python pretrain_main.py \
    --task pvqa --epochs 120 --start_epoch 0 \
    --lr 0.1 --cos --train train --tfidf \
    --output saved_models/pvqa/pre/${dir} \
    --batch_size 128 --seed ${seed} --img_v ${imgv} \
    --pretrain_tasks ${pretask}

  PYTHONPATH=$PYTHONPATH:./src\
    python finetune_main.py \
    --task pvqa --epochs 200 --start_epoch 0 \
    --lr 0.01 --cos --train train --val val --tfidf \
    --input saved_models/pvqa/pre/${dir}/model_epoch0.pth \
    --output saved_models/pvqa/pre/${dir} --img_v ${imgv} \
    --batch_size 128 --seed $seed

  PYTHONPATH=$PYTHONPATH:./src\
    python evaluate_main.py \
    --task pvqa --data_split test --tfidf \
    --input saved_models/pvqa/pre/${dir}/model_best.pth \
    --output saved_models/pvqa/pre/${dir} \
    --batch_size 128 --seed $seed --img_v ${imgv}
done
