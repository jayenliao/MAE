python visualize_pred.py \
  --pretrained_ckpt ./logs/pretrain-cls/20251008-150155/vit-t-clf-from_pretrained.pt \
  --scratch_ckpt ./logs/scratch-cls/20251008-150202/vit-t-clf-from_scratch.pt \
  --indices 2 9 10 11 12 13 14 15 16 17 18 19\
  --save_dir logs/cls_viz_selected
