python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name main_exp_150_50 \
    --pretrained_model_path ./outputs/main_exp_150_50/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto \
    --linear_probe
