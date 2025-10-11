python visualize_pred.py \
  --pretrained_ckpt ./outputs/main_exp/pretrain-cls/vit-t-clf-from_pretrained.pt \
  --scratch_ckpt ./outputs/main_exp/scratch-cls/vit-t-clf-from_scratch.pt \
  --indices 2 9 10 11 12 13 14 15 16 17 18 19\
  --save_dir ./outputs/main_exp/cls_viz
