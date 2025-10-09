python train_classifier.py \
    --total_epoch 100 --warmup_epoch 10 --batch_size 128 \
    --output_root outputs --exp_name main_exp \
    --pretrained_model_path ./outputs/main_exp/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto
