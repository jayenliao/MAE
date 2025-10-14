python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/depth/1 \
    --pretrained_model_path ./outputs/ablation/depth/1/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto \
    --linear_probe

python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/depth/2 \
    --pretrained_model_path ./outputs/ablation/depth/2/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto \
    --linear_probe

python train_classifier.py \
    --total_epoch 50 --warmup_epoch 5 --batch_size 128 \
    --output_root outputs --exp_name ablation/depth/8 \
    --pretrained_model_path ./outputs/ablation/depth/8/mae-pretrain/vit-t-mae.pt \
    --output_model_path auto \
    --linear_probe
