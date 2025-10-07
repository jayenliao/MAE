# Multimodal AI Assignment 1 - Masked Autoencoders

- Name: Jay Chiehen Liao
- ID: R13922210
- E-mail: r13922210@ntu.edu.tw

## Description

The assignment aims to reproduce [*KaiMing He el.al. Masked Autoencoders Are Scalable Vision Learners*](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf7) based on github user [IcarusWizard's code base](https://github.com/IcarusWizard/MAE).

## Envrionment

The project was implemented by Python programming and conducted on 

screenshots

For Python package

## Files

- `requirements.txt` pins the environment: torch==1.10.1, torchvision==0.11.2, timm==0.4.12, plus tensorboard, einops, tqdm.

readme.md — states the replication goal (MAE pretraining improves supervised fine-tuning on CIFAR-10), installation, run commands, and an example result table/log links.

model.py — defines the MAE components:

MAE_Encoder: patchify (conv), add positional embeddings, random patch shuffle/masking, ViT blocks, LN; returns features and unshuffle indices.

MAE_Decoder: adds mask tokens & positional embeddings, small ViT, linear head to pixels, reshapes patches→image, and builds a mask map (1 on masked areas).

MAE_ViT: wires encoder+decoder; forward returns (reconstructed_img, mask).

ViT_Classifier: reuses the pretrained encoder (cls token, pos embedding, patchify, transformer, LN) and adds a linear head for 10 classes.

mae_pretrain.py — self-supervised pretraining on CIFAR-10 with default 2000 epochs, 75% mask, AdamW + warmup+cosine; logs MSE@masked loss and reconstruction images to TensorBoard; saves vit-t-mae.pt.

train_classifier.py — supervised fine-tuning on CIFAR-10 either from scratch or from the pretrained encoder, with TensorBoard curves for loss/accuracy and model checkpointing at best val acc.

utils.py — deterministic training seed setup across PyTorch/CUDA/NumPy. (Important for fair ablations.)

Dataset: both scripts use CIFAR-10 train/val splits from torchvision.datasets.CIFAR10
