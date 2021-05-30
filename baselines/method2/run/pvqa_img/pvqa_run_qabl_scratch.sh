imgv=
for seed in 1204 1 2 3 4 5
do
  dir=qabl_${seed}_imgv${imgv}
  echo $dir

  PYTHONPATH=$PYTHONPATH:./src\
    python finetune_main.py \
    --task pvqa --epochs 60 --start_epoch 0 \
    --lr 0.01 --cos --train train --val val --tfidf \
    --output saved_models/pvqa/pre/${dir} \
    --batch_size 128 --seed $seed --qa_bl --img_v ${imgv}

  PYTHONPATH=$PYTHONPATH:./src\
    python evaluate_main.py \
    --task pvqa --data_split test --tfidf \
    --input saved_models/pvqa/pre/${dir}/model_best.pth \
    --output saved_models/pvqa/pre/${dir} \
    --batch_size 128 --seed $seed --qa_bl --img_v ${imgv}
done
