imgv=
dir=qabl_scratch_imgv${imgv}
for seed in 9595 1 2 3 4 5
do


  ft_name=pvqa_pre_from_pretrained_${dir}_${seed}_lxr955 # pvqa_pre

  # Save logs and models under snap/vqa; make backup.
  ft_output=snap/pvqa/$ft_name
  mkdir -p $ft_output

  # See Readme.md for option details.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
      python src/tasks/pvqa.py \
      --train train --valid val  \
      --llayers 9 --xlayers 0 --rlayers 0 \
      --loadLXMERT snap/pretrained/model \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 120 \
      --seed $seed --qa_bl --pvqaimgv $imgv \
      --tqdm --output $ft_output

    # The name of this experiment.
  res_name=pvqa_pre_from_pretrained_${dir}_${seed}_lxr955_results

  # Save logs and models under snap/vqa; make backup.
  res_output=snap/pvqa/$res_name
  mkdir -p $res_output

  # See Readme.md for option details.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
      python src/tasks/pvqa.py \
      --test test  --train val --valid "" \
      --load ${ft_output}/BEST \
      --llayers 9 --xlayers 0 --rlayers 0 \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
      --seed $seed --qa_bl --pvqaimgv $imgv \
      --tqdm --output $res_output

done
