"""
Replica EXATA do treino original do JAGC (Wu et al., ICCV 2019,
github.com/xpwu95/LDL, code/train.py) no split oficial do ACNE04, usando o
codigo do modelo, da loss (genLD) e das transformacoes copiado do proprio
repositorio deles (third_party/jagc), so trocando CUDA por MPS/CPU.

Fiel ao original: ResNet-50 com pesos ImageNet carregados no construtor e
BatchNorm congelado, SGD lr=0.001 momentum=0.9 wd=5e-4, batch 32, 120
epocas, LR x0.5 nas epocas 30/60/90, sem validacao e sem early stopping,
imagens 300x300 -> resize 256 -> crop 224 + flip + rotacao 20.

Como o original avalia no teste a cada epoca (a partir da 9) e nao salva
checkpoint, registramos o historico por epoca e reportamos DOIS numeros:
(a) a ultima epoca (sem nenhuma selecao no teste -- o honesto) e
(b) a melhor epoca no teste (otimista; so pra mostrar quanto a selecao
no teste pesaria).

Uso: python run_jagc_original_replica.py --fold 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from jagc.model.resnet50 import resnet50  # noqa: E402
from jagc.transforms.affine_transforms import RandomRotate  # noqa: E402
from jagc.utils.genLD import genLD  # noqa: E402

LR = 0.001
BATCH_SIZE = 32
LR_STEPS = [30, 60, 90, 120]
SIGMA = 30 * 0.1
LAM = 6 * 0.1
IMG_DIR = "../data/jagc_replica_300"  # sobrescrito por --img-dir


class JAGCDataset(Dataset):
    def __init__(self, items, transform, img_dir):
        self.items = items  # (filename, label, lesion_count)
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, label, lesion = self.items[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        return self.transform(img), torch.tensor(label), torch.tensor(lesion)


class JAGCNetTV(nn.Module):
    """Mesmo desenho do ResNet de third_party/jagc (pesos ImageNet, BN congelado
    e sempre em eval, duas cabecas lineares fc(4)/counting(65), cou2cls por soma de
    baldes), mas com backbone torchvision de outra profundidade (ex: resnet18)."""

    def __init__(self, arch):
        super().__init__()
        from torchvision import models
        ctor, w = {"resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
                   "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT)}[arch]
        base = ctor(weights=w)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = base.fc.in_features
        self.fc = nn.Linear(feat, 4)
        self.counting = nn.Linear(feat, 65)
        for m in (self.fc, self.counting):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                for q in m.parameters():
                    q.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x, x_new=None):
        x = self.pool(self.features(x)).flatten(1)
        cls = torch.softmax(self.fc(x), dim=1) + 1e-4
        cou = torch.softmax(self.counting(x), dim=1) + 1e-4
        cou2cls = torch.stack((cou[:, :5].sum(1), cou[:, 5:20].sum(1), cou[:, 20:50].sum(1), cou[:, 50:].sum(1)), 1)
        return cls, cou, cou2cls


def numpy_collate(batch):
    """Monta o batch em numpy dentro do worker: no Windows (Python 3.15 + torch 2.14) o envio de tensores
    entre processos via memoria compartilhada derruba os workers do DataLoader."""
    return tuple(np.stack([np.asarray(item[k]) for item in batch]) for k in range(len(batch[0])))


class TensorLoader:
    """Converte de volta para tensores no processo principal; o resto do treino fica identico."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield tuple(torch.from_numpy(b) for b in batch)

    def __len__(self):
        return len(self.loader)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--name", default="jagc_original_replica")
    ap.add_argument("--splits", default="results/kfold_splits_official_acne04.json")
    ap.add_argument("--output-dir", default="results/kfold_wholeimage")
    ap.add_argument("--epochs", type=int, default=LR_STEPS[-1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-v2", default=None,
                    help="Dataset combinado Acne04+AcneLevel (manifest + --splits kfold_splits_wholeimage_v2.json); "
                         "contagens via identidade canonica a partir de --count-source. Use com --img-dir.")
    ap.add_argument("--count-source", default="results/kfold_splits_official_acne04.json")
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--only-source", default=None, choices=["a04", "lvl"],
                    help="Com --manifest-v2: usa so imagens de uma fonte (a04=Acne04, lvl=AcneLevel tratado) em treino e teste.")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18", "resnet34"])
    ap.add_argument("--resize", type=int, default=256, help="Lado do resize de treino antes do crop aleatorio.")
    ap.add_argument("--crop", type=int, default=224, help="Lado do crop de treino; o teste usa resize direto para esse lado.")
    ap.add_argument("--tta", action="store_true",
                    help="Ao final, tambem salva probs com TTA (media da imagem original e do flip horizontal).")
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"fold{args.fold}.npz")
    if os.path.exists(npz_path):
        print(f"[{args.name}] fold {args.fold} ja concluido, pulando")
        return

    fold = json.load(open(args.splits))["folds"][args.fold]

    img_dir = args.img_dir or IMG_DIR

    if args.manifest_v2:
        manifest = {m["name"]: m for m in json.load(open(args.manifest_v2))}
        counts_by_canon = {}
        for fd in json.load(open(args.count_source))["folds"]:
            for sk in ("train", "val", "test"):
                counts_by_canon.update(fd[f"{sk}_lesion_counts"])

        def items(keys):
            return [(n, int(manifest[n]["level"]), int(round(counts_by_canon[manifest[n]["canonical"]])))
                    for k in keys for n in fold[k]
                    if n in manifest and (args.only_source is None or manifest[n]["source"] == args.only_source)]
    else:
        def items(keys):
            out = []
            for k in keys:
                labels, counts = fold[f"{k}_labels_official"], fold[f"{k}_lesion_counts"]
                out += [(n, int(labels[n]), int(round(counts[n]))) for n in labels]
            return out

    train_items = items(["train", "val"])  # = NNEW_trainval do original
    test_items = items(["test"])
    print(f"[{args.name}] fold {args.fold}: trainval={len(train_items)} test={len(test_items)}", flush=True)

    np.random.seed(42)
    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266], std=[0.2814769, 0.226306, 0.20132513])
    train_tf = transforms.Compose([
        transforms.Resize((args.resize, args.resize)), transforms.RandomCrop(args.crop), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), RandomRotate(rotation_range=20), normalize,
    ])
    test_tf = transforms.Compose([transforms.Resize((args.crop, args.crop)), transforms.ToTensor(), normalize])

    loader_kw = dict(num_workers=args.num_workers, collate_fn=numpy_collate, persistent_workers=args.num_workers > 0)
    train_loader = TensorLoader(DataLoader(JAGCDataset(train_items, train_tf, img_dir), batch_size=BATCH_SIZE, shuffle=True, **loader_kw))
    test_loader = TensorLoader(DataLoader(JAGCDataset(test_items, test_tf, img_dir), batch_size=20, shuffle=False, **loader_kw))

    device = get_device()
    cnn = (resnet50() if args.arch == "resnet50" else JAGCNetTV(args.arch)).to(device)
    params = [{"params": [v], "lr": LR, "weight_decay": 5e-4} for _, v in cnn.named_parameters() if v.requires_grad]
    optimizer = torch.optim.SGD(params, momentum=0.9)
    kl1, kl2, kl3 = nn.KLDivLoss(), nn.KLDivLoss(), nn.KLDivLoss()

    history = []
    start = time.time()
    final_probs = None
    for epoch in range(args.epochs):
        if epoch in LR_STEPS:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
        cnn.train()
        for b_x, _, b_l in train_loader:
            b_x = b_x.to(device)
            b_l = b_l.numpy() - 1  # contagem -> indice 0..64, como no original
            ld = genLD(b_l, SIGMA, "klloss", 65)
            ld_4 = np.vstack((ld[:, :5].sum(1), ld[:, 5:20].sum(1), ld[:, 20:50].sum(1), ld[:, 50:].sum(1))).T
            ld = torch.from_numpy(ld).float().to(device)
            ld_4 = torch.from_numpy(ld_4).float().to(device)
            cls, cou, cou2cls = cnn(b_x, None)
            loss_cls = kl1(torch.log(cls), ld_4) * 4.0
            loss_cou = kl2(torch.log(cou), ld) * 65.0
            loss_cls_cou = kl3(torch.log(cou2cls), ld_4) * 4.0
            loss = (loss_cls + loss_cls_cou) * 0.5 * LAM + loss_cou * (1.0 - LAM)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch >= 9:
            cnn.eval()
            ys, pm, pc, probs = [], [], [], []
            with torch.no_grad():
                for x, y, _ in test_loader:
                    cls, _, cou2cls = cnn(x.to(device), None)
                    combo = (cls + cou2cls).cpu().numpy()
                    ys.append(y.numpy()); pm.append(combo.argmax(1)); pc.append(cls.argmax(1).cpu().numpy())
                    probs.append(combo / combo.sum(1, keepdims=True))
            ys, pm, pc = np.concatenate(ys), np.concatenate(pm), np.concatenate(pc)
            final_probs = np.concatenate(probs)
            history.append({"epoch": epoch, "acc_combo": float(accuracy_score(ys, pm)),
                            "acc_cls": float(accuracy_score(ys, pc))})
            if epoch % 10 == 9:
                print(f"  epoch {epoch}: acc_combo={history[-1]['acc_combo']:.4f} acc_cls={history[-1]['acc_cls']:.4f} "
                      f"({(time.time() - start) / 60:.0f} min)", flush=True)

    labels = ys
    preds = final_probs.argmax(1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    best = max(h["acc_combo"] for h in history)
    minutes = (time.time() - start) / 60
    print(f"[{args.name}] fold {args.fold}: ULTIMA epoca Acc={acc:.4f} F1m={f1:.4f} QWK={qwk:.4f} | "
          f"melhor-epoca-no-teste Acc={best:.4f} ({minutes:.1f} min)", flush=True)

    extra = {}
    if args.tta:
        cnn.eval()
        tta_probs = []
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x.to(device)
                combo = sum((lambda o: o[0] + o[2])(cnn(v, None)) for v in (x, torch.flip(x, dims=[3]))).cpu().numpy()
                tta_probs.append(combo / combo.sum(1, keepdims=True))
        tta_probs = np.concatenate(tta_probs)
        tp = tta_probs.argmax(1)
        extra["probs_tta"] = tta_probs
        extra["metrics_tta"] = np.array([accuracy_score(labels, tp), f1_score(labels, tp, average="macro"),
                                         cohen_kappa_score(labels, tp, weights="quadratic")])
        print(f"[{args.name}] fold {args.fold}: com TTA Acc={extra['metrics_tta'][0]:.4f} F1m={extra['metrics_tta'][1]:.4f} "
              f"QWK={extra['metrics_tta'][2]:.4f}", flush=True)

    np.savez(npz_path, names=np.array([t[0] for t in test_items]), labels=labels, probs=final_probs,
             metrics=np.array([acc, f1, qwk, best]), minutes=np.array([minutes]), **extra)
    json.dump(history, open(os.path.join(out_dir, f"history_fold{args.fold}.json"), "w"))


if __name__ == "__main__":
    main()
