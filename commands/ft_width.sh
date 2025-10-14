python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/width/64 \
    --pretrained_model_path ./outputs/ablation/width/64/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto

python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/width/128 \
    --pretrained_model_path ./outputs/ablation/width/128/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto

python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/width/256 \
    --pretrained_model_path ./outputs/ablation/width/256/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto

python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/width/512 \
    --pretrained_model_path ./outputs/ablation/width/512/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto
