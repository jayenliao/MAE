python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/depth/1 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 1 --decoder_dim 192

python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/depth/2 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 2 --decoder_dim 192

python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/depth/8 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 8 --decoder_dim 192
