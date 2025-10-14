python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/width/512 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 4 --decoder_dim 512

python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/width/256 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 4 --decoder_dim 256

python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/width/128 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 4 --decoder_dim 128

python mae_pretrain.py \
    --total_epoch 150 --warmup_epoch 15 --visualize_freq 50 \
    --output_root outputs --exp_name ablation/width/64 \
    --model_fn vit-t-mae.pt --csv_fn metrics.csv \
    --save_images_dir images \
    --mask_ratio 0.75 \
    --batch_size 512 \
    --decoder_layer 4 --decoder_dim 64
