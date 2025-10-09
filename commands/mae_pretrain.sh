python mae_pretrain.py \
    --total_epoch 600 --warmup_epoch 60 --visualize_freq 50 \
    --output_root outputs --exp_name main_exp \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512
