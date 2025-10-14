# Multimodal AI Assignment 1 - Masked Autoencoders

- Name: Jay Chiehen Liao (廖傑恩)
- ID: R13922210
- E-mail: r13922210@ntu.edu.tw

This repo reproduces key findings from **Masked Autoencoders Are Scalable Vision Learners (MAE)** on CIFAR-10: self-supervised pretraining improves downstream classification versus training from scratch, and we studied how **decoder depth** and **decoder width** affect MAE pretraining and downstream results.

- Code is adapted from a simplified MAE implementation using ViT blocks from `timm`.
- Dataset is CIFAR-10 (auto-downloaded via `torchvision`).

## Environment

- Install required packages listed in `requirements.txt`.

- You may create a conda env:

    ```bash
    conda create --name mae python=3.12
    conda activate mae
    pip install -r requirements.txt
    ```

## Repo Structure

- `model.py` defines MAE encoder/decoder and a `ViT_Classifier` built on the pretrained encoder.  Encoder masks patches, the decoder reconstructs. The classifier reuses encoder weights and adds a linear head.

- `mae_pretrain.py` self-supervises pretraining on CIFAR-10 with cosine-decayed LR, warmup, random masking. The script also logs loss to a csv file and periodically saves image grids under `images/epoch_XXXX/`.

- `train_classifier.py` trains a classifier on CIFAR-10, either from scratch or loading a pretrained encoder. It supports **linear probe** (with `--linear_probe`) or full fine-tuning.

- `visualize_pred.py` produces side-by-side prediction figures for pretrained vs. scratch classifiers on the same set of test images.

- `utils.py`: seeds and a tiny CSV logger.

- `metrics.ipynb`gets accuracy scores and losses from `metrics.csv` and visualize them.

Outputs are written under:

```
outputs/<EXP_NAME>/
  mae-pretrain/     # pretraining (checkpoint, images/, tensorboard/)
  pretrain-cls/     # classifier from pretrained encoder
  pretrain-cls-lin/ # linear probe from pretrained encoder
  scratch-cls/      # classifier trained from scratch
```

## Usage

### Main Experiments

#### 1. MAE pretraining

- Version of 600 epochs

    You may modify the shell script to adjust the number of epochs.

    ```bash
    bash commands/mae_pretrain.sh
    ```

    This writes the checkpoint to:

    ```
    outputs/main_exp/mae-pretrain/vit-t-mae.pt
    ```

- Version of 600 epochs

    ```bash
    bash commands/mae_pretrain_150.sh
    ```

    This writes the checkpoint to:

    ```
    outputs/main_exp_150_50/mae-pretrain/vit-t-mae.pt
    ```


#### 2-1. Classification from Scratch (without MAE pretraining)

- Version of 600-100 epochs

    ```bash
    bash commands/scratch_cls.sh
    ```

- Version of 150-50 epochs

    ```bash
    bash commands/scratch_cls_50.sh
    ```

#### 2-2. Classification with MAE Pretraining (Fine-tuning)

- Version of 600-100 epochs

    ```bash
    bash commands/pretrain_cls.sh
    ```

- Version of 150-50 epochs

    ```bash
    bash commands/pretrain_cls_50.sh
    ```

#### 3. Visualize Predictions (Pretrained vs. Scratch)

Before running, please ensure the main experiment (`main_exp`) has completed.

```bash
bash visualize_pred.sh
```

### Ablations

This repo explores **decoder depth** and **decoder width** during MAE pretraining:

#### Decoder depth

`--decoder_layer {2,(4),6,8}`, where `depth=4` is the default setting.

1. MAE Pretraining

    ```bash
    bash commands/mae_pretrain_depth.sh
    ```

2. Fine-tuning. Before running the following commands, ensure `commands/mae_pretrain_depth.sh` and experiment `main_exp_150_50` have been completed.

    ```bash
    bash commands/ft_depth.sh
    bash commands/ft_default_d4_w192.sh
    ```

3. Linear probing. Before running the following commands, ensure `commands/mae_pretrain_depth.sh` and experiment `main_exp_150_50` have been completed.

    ```bash
    bash commands/lin_depth.sh
    bash commands/lin_default_d4_w192.sh
    ```

#### Decoder width

`--decoder_dim {64,128,(192),256,512}`, where `decoder_dim=192` is the default setting.

1. MAE Pretraining

    ```bash
    bash commands/mae_pretrain_width.sh
    ```

2. Fine-tuning. Before running the following commands. Before running the following commands, ensure `commands/mae_pretrain_width.sh` has been completed.

    ```bash
    bash commands/ft_width.sh
    ```

3. Linear probing. Before running the following commands. Before running the following commands, ensure `commands/mae_pretrain_width.sh` has been completed.

    ```bash
    bash commands/lin_width.sh
    ```

### Getting result (accuracy scores, loss curves)

- CSV logs: per-epoch metrics are written as `metrics.csv` in each run folder.
- Please run all cells in `metrics.ipynb` to get accuracy scores and training loss curves. Note that you may need to change the file paths if you modify the shell scripts mentioned above.

## Reproducibility

Set `--seed` (default `42`). CuDNN is set to deterministic.

## Known Issues (and how to avoid them)

- Re-running with the same `--exp_name` for pretraining will error on existing directories. Change `--exp_name` or delete the old folder.

- Path assertion: the classifier asserts the pretrained path contains `mae-pretrain`. Use the generated path structure or relax the assertion if you prefer.

## References

- He, Kaiming, et al. “Masked Autoencoders Are Scalable Vision Learners.” *CVPR* (2022).

- Krizhevsky, Alex. “Learning Multiple Layers of Features from Tiny Images.” *Tech Report*, 2009 (CIFAR-10).

