#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


NetNum=0102300000
CP_dir='Task_models'
pCP_dir='TOV_models'
PJ="SSD"

# * Finetune *
f_epochs=200

# *New*
pDataSet='TOV-RS-blanced'
pretrain_config=$pCP_dir'/'$NetNum'_'22061085439_pretrain  #

for fDataSet in "eurorgb"
do
  note=$pDataSet"_fintune_on_"$fDataSet
  for train_scales in 5 20 50 100
  do
  python main_cls.py --mode 410 --pj $PJ --exp_note $note \
    --dataset $fDataSet --input_size 64 --val_scale 1 --train_scale $train_scales \
    --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
    --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
    --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
    --super_prefetch --data_workers 4 \
    --map_keys '' 'features.' \
    --load_pretrained $pretrain_config
  done
done

note=$pDataSet"_fintune_on_tg2rgb"
for train_scales in 5 20 50 100
do
python main_cls.py --mode 410 --pj $PJ --exp_note $note \
  --dataset "tg2rgb" --input_size 128 --val_scale 1 --train_scale $train_scales \
  --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
  --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
  --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
  --super_prefetch --data_workers 4 \
  --map_keys '' 'features.' \
  --load_pretrained $pretrain_config
done


for fDataSet in "aid" "nr" "rsd46" "pnt" "ucm"
do
  note=$pDataSet"_fintune_on_"$fDataSet
  for train_scales in 5 20 50 100
  do
  python main_cls.py --mode 410 --pj $PJ --exp_note $note \
    --dataset $fDataSet --input_size 224 --val_scale 1 --train_scale $train_scales \
    --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
    --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
    --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
    --super_prefetch --data_workers 4 \
    --map_keys '' 'features.' \
    --load_pretrained $pretrain_config
  done
done

# note="Imagenet_pretrain"
# for fDataSet in "eurorgb"
# do
#   note="Imagenet_pretrain_fintune_on_"$fDataSet
#   for train_scales in 5 20 50 100
#   do
#   python main_cls.py --mode 1 --pj $PJ --exp_note $note \
#     --dataset $fDataSet --input_size 64 --val_scale 1 --train_scale $train_scales \
#     --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
#     --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
#     --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
#     --super_prefetch --data_workers 4 \
#     --pretrained --load_pretrained ''  # ImageNet(SL)
#   done
# done
# for fDataSet in "nr"
# do
#   note="Imagenet_pretrain_fintune_on_"$fDataSet
#   for train_scales in 100
#   do
#   python main_cls.py --mode 1 --pj $PJ --exp_note $note \
#     --dataset $fDataSet --input_size 224 --val_scale 1 --train_scale $train_scales \
#     --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
#     --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
#     --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
#     --super_prefetch --data_workers 4 \
#     --pretrained --load_pretrained ''  # ImageNet(SL)
#   done
# done

# for fDataSet in "rsd46" "pnt" "ucm" "eurorgb"
# do
#   note="Imagenet_pretrain_fintune_on_"$fDataSet
#   for train_scales in 5 20 50 100
#   do
#   python main_cls.py --mode 1 --pj $PJ --exp_note $note \
#     --dataset $fDataSet --input_size 224 --val_scale 1 --train_scale $train_scales \
#     --learning_rate 4e-3 --optimizer adam --lr_scheduler cosine --freeze_layers 'features' \
#     --model_name $NetNum --max_epochs $f_epochs --weight_decay 0 \
#     --ckpt $CP_dir --num_workers 4 --gpu 1 --batch_size 64 \
#     --super_prefetch --data_workers 4 \
#     --pretrained --load_pretrained ''  # ImageNet(SL)
#   done
# done
