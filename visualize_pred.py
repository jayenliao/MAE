import argparse, os, math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

from model import ViT_Classifier

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def to01(x):  # [-1,1] -> [0,1]
    return ((x.clamp(-1,1) + 1) / 2).clamp(0,1)

@torch.no_grad()
def predict(model, x, device):
    logits = model(x.to(device))
    preds = logits.argmax(dim=1).cpu()
    return preds

def load_classifier(ckpt_path, num_classes=10, device='cuda'):
    model = ViT_Classifier(num_classes=num_classes).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # handle plain state_dict or wrapped
        sd = state.get('model', state)
        model.load_state_dict(sd, strict=False)
        print(f'Loaded: {ckpt_path}')
    else:
        print('WARNING: checkpoint missing or None; using randomly initialized weights')
    model.eval()
    return model

def make_loaders(data_root, batch_size=256, split='test'):
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    ds = CIFAR10(root=data_root, train=(split=='train'), download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return ds, dl

def gather_examples(model_pre, model_scr, dl, device, per_category=6, max_batches=None):
    """
    Build four buckets of (img, label, pred_pre, pred_scr):
      A: pretrained correct, scratch wrong
      B: scratch correct, pretrained wrong
      C: both correct
      D: both wrong
    """
    buckets = {'pre_wins': [], 'scr_wins': [], 'both_correct': [], 'both_wrong': []}
    need_more = lambda: any(len(v) < per_category for v in buckets.values())

    for b, (x, y) in enumerate(dl):
        if not need_more(): break
        pre = predict(model_pre, x, device)
        scr = predict(model_scr, x, device)

        for i in range(x.size(0)):
            xi, yi = x[i], y[i].item()
            pre_ok = (pre[i].item() == yi)
            scr_ok = (scr[i].item() == yi)
            item = (xi, yi, pre[i].item(), scr[i].item())

            if pre_ok and not scr_ok and len(buckets['pre_wins']) < per_category:
                buckets['pre_wins'].append(item)
            elif scr_ok and not pre_ok and len(buckets['scr_wins']) < per_category:
                buckets['scr_wins'].append(item)
            elif pre_ok and scr_ok and len(buckets['both_correct']) < per_category:
                buckets['both_correct'].append(item)
            elif (not pre_ok) and (not scr_ok) and len(buckets['both_wrong']) < per_category:
                buckets['both_wrong'].append(item)

        if max_batches is not None and (b+1) >= max_batches:
            break
    return buckets

def _tile(ax, img01, title_top, title_bottom, color_top='black', color_bottom='black'):
    ax.imshow(img01.permute(1,2,0).numpy())
    ax.axis('off')
    ax.set_title(title_top, color=color_top, fontsize=10, pad=2)
    ax.text(0.5, -0.15, title_bottom, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color=color_bottom)

def plot_bucket(bucket, title, savepath=None, ncols=6):
    n = len(bucket)
    if n == 0:
        print(f'[WARN] No samples for {title}')
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if nrows*ncols>1 else [axes]

    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis('off'); continue
        x, y, pre, scr = bucket[k]
        img01 = to01(x.cpu())
        # captions
        top = f'True: {CLASSES[y]}'
        # color green if correct, red if wrong (pretrained line on top, scratch below)
        pre_ok = (pre == y); scr_ok = (scr == y)
        bottom = f'Pre: {CLASSES[pre]}   |   Scr: {CLASSES[scr]}'
        _tile(ax, img01, title_top=top,
              title_bottom=bottom,
              color_top='green' if (pre_ok or scr_ok) else 'red',
              color_bottom='green' if pre_ok and scr_ok else ('orange' if pre_ok or scr_ok else 'red'))
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
        print(f'Saved: {savepath}')
    plt.show()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./data')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--pretrained_ckpt', type=str, required=True,
                    help='Checkpoint of classifier fine-tuned with MAE pretraining')
    ap.add_argument('--scratch_ckpt', type=str, required=True,
                    help='Checkpoint of classifier trained from scratch')
    ap.add_argument('--per_category', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--save_dir', type=str, default='logs/cifar10/cls_viz')
    args = ap.parse_args()

    ds, dl = make_loaders(args.data_root, batch_size=args.batch_size, split='test')
    pre = load_classifier(args.pretrained_ckpt, device=args.device)
    scr = load_classifier(args.scratch_ckpt, device=args.device)

    buckets = gather_examples(pre, scr, dl, args.device, per_category=args.per_category)

    plot_bucket(buckets['pre_wins'],    'Pretrained wins (Pre correct, Scratch wrong)',
                savepath=os.path.join(args.save_dir, 'pre_wins.png'))
    plot_bucket(buckets['scr_wins'],    'Scratch wins (Scratch correct, Pre wrong)',
                savepath=os.path.join(args.save_dir, 'scratch_wins.png'))
    plot_bucket(buckets['both_correct'],'Both correct',
                savepath=os.path.join(args.save_dir, 'both_correct.png'))
    plot_bucket(buckets['both_wrong'],  'Both wrong',
                savepath=os.path.join(args.save_dir, 'both_wrong.png'))

if __name__ == '__main__':
    main()
