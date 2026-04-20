import argparse
import dataclasses
import os
import random
import tarfile
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
import heapq
import itertools

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms


SLOT_ORDER: tuple[str, ...] = (
    "top",
    "outwear",
    "dress",
    "jumpsuit",
    "pants",
    "skirt",
    "legwear",
    "shoes",
    "bag",
    "neckwear",
    "hairwear",
    "hats",
    "eyewear",
    "gloves",
    "bracelet",
    "necklace",
    "earrings",
    "rings",
    "brooch",
    "watches",
)


def slot_from_image_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path)).lower()


_SLOT_RANK: dict[str, int] = {slot: i for i, slot in enumerate(SLOT_ORDER)}


def sort_outfit_paths(paths: Sequence[str]) -> list[str]:
    def key(p: str) -> tuple[int, str]:
        slot = slot_from_image_path(p)
        rank = _SLOT_RANK.get(slot, len(_SLOT_RANK))
        return rank, os.path.basename(p).lower()

    return sorted(paths, key=key)


def build_image_index(images_root: str) -> dict[str, str]:
    images_root = os.path.abspath(images_root)
    index: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(images_root):
        for fn in filenames:
            if not fn.lower().endswith(".jpg"):
                continue
            key = fn.lower()
            full = os.path.join(dirpath, fn)
            if key not in index:
                index[key] = full
    return index


def open_polyvore_member_text(tar: tarfile.TarFile, member_name: str) -> Iterable[str]:
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    for raw in f:
        yield raw.decode("utf-8", errors="replace").rstrip("\n")


def resolve_polyvore_member_name(tar: tarfile.TarFile, candidates: Sequence[str]) -> str:
    available = {name.lower(): name for name in tar.getnames()}
    for c in candidates:
        hit = available.get(c.lower())
        if hit is not None:
            return hit
    raise FileNotFoundError(" / ".join(candidates))


def parse_cp_line(line: str) -> Tuple[int, list[str]]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad line: {line!r}")
    label = int(parts[0])
    ids = parts[1:]
    filenames = [f"{item_id}.jpg" for item_id in ids]
    return label, filenames


def roc_auc_score_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank = 1
    sum_ranks_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                sum_ranks_pos += avg_rank
        rank += j - i
        i = j
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


@dataclasses.dataclass(frozen=True)
class CpExample:
    label: int
    image_paths: Tuple[str, ...]


def load_cp_examples(
    *,
    polyvore_tar_path: str,
    image_index: dict[str, str],
    min_items: int,
    max_items: int,
) -> list[CpExample]:
    examples: list[CpExample] = []
    with tarfile.open(polyvore_tar_path) as tar:
        member = resolve_polyvore_member_name(
            tar,
            [
                "fashion_compatibility_prediction.txt",
                "fashion-compatibility-prediction.txt",
            ],
        )
        for line in open_polyvore_member_text(tar, member):
            if not line.strip():
                continue
            label, filenames = parse_cp_line(line)
            paths: list[str] = []
            for fn in filenames[:max_items]:
                p = image_index.get(fn.lower())
                if p is not None:
                    paths.append(p)
            if len(paths) < min_items:
                continue
            examples.append(CpExample(label=label, image_paths=tuple(paths)))
    return examples


class CpDataset(Dataset[CpExample]):
    def __init__(self, examples: Sequence[CpExample], transform: transforms.Compose):
        self._examples = list(examples)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        ex = self._examples[idx]
        images: list[torch.Tensor] = []
        for p in ex.image_paths:
            img = Image.open(p).convert("RGB")
            images.append(self._transform(img))
        x = torch.stack(images, dim=0)
        mask = torch.ones((x.shape[0],), dtype=torch.bool)
        y = int(ex.label)
        return x, mask, y


def collate_cp(batch: Sequence[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, masks, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    c, h, w = xs[0].shape[1], xs[0].shape[2], xs[0].shape[3]
    batch_x = torch.zeros((len(xs), max_len, c, h, w), dtype=xs[0].dtype)
    batch_mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, (x, m) in enumerate(zip(xs, masks)):
        n = x.shape[0]
        batch_x[i, :n] = x
        batch_mask[i, :n] = m
    batch_y = torch.tensor(ys, dtype=torch.float32)
    return batch_x, batch_mask, batch_y


class OutfitCompatModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        embed_dim: int,
        freeze_backbone: bool,
        arch: str,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
    ):
        super().__init__()
        if backbone_name == "resnet18":
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                try:
                    backbone = models.resnet18(pretrained=True)
                except Exception:
                    try:
                        backbone = models.resnet18(weights=None)
                    except Exception:
                        backbone = models.resnet18(pretrained=False)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            try:
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            except Exception:
                try:
                    backbone = models.resnet50(pretrained=True)
                except Exception:
                    try:
                        backbone = models.resnet50(weights=None)
                    except Exception:
                        backbone = models.resnet50(pretrained=False)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if arch not in {"mean", "bilstm"}:
            raise ValueError(f"Unsupported arch: {arch}")
        self.arch = arch
        if self.arch == "mean":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(2 * lstm_hidden, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = x.shape
        x_flat = x.view(b * n, c, h, w)
        feats = self.backbone(x_flat)
        feats = self.proj(feats)
        feats = feats.view(b, n, -1)
        if self.arch == "mean":
            mask_f = mask.to(dtype=feats.dtype).unsqueeze(-1)
            summed = (feats * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            outfit = summed / denom
            logit = self.head(outfit).squeeze(-1)
            return logit

        lengths = mask.sum(dim=1).to(dtype=torch.int64)
        lengths = torch.clamp(lengths, min=1)
        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        outfit = torch.cat([h_fwd, h_bwd], dim=-1)
        logit = self.head(outfit).squeeze(-1)
        return logit


def run_stats(args: argparse.Namespace) -> None:
    image_index = build_image_index(args.images_root)
    examples = load_cp_examples(
        polyvore_tar_path=args.polyvore_tar,
        image_index=image_index,
        min_items=args.min_items,
        max_items=args.max_items,
    )
    labels = [ex.label for ex in examples]
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"images_indexed={len(image_index)}")
    print(f"cp_examples_usable={len(examples)} pos={pos} neg={neg} pos_ratio={(pos / max(1, len(examples))):.3f}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    ys: list[int] = []
    scores: list[float] = []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            logit = model(x, mask)
            loss = loss_fn(logit, y)
            losses.append(float(loss.item()))
            prob = torch.sigmoid(logit)
            pred = (prob >= 0.5).to(torch.int64)
            correct += int((pred == y.to(torch.int64)).sum().item())
            total += int(y.numel())
            ys.extend([int(v) for v in y.tolist()])
            scores.extend([float(v) for v in prob.tolist()])
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "acc": correct / max(1, total),
        "auc": roc_auc_score_binary(ys, scores) if total > 0 else float("nan"),
    }


def run_train_cp(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_index = build_image_index(args.images_root)
    examples = load_cp_examples(
        polyvore_tar_path=args.polyvore_tar,
        image_index=image_index,
        min_items=args.min_items,
        max_items=args.max_items,
    )
    if len(examples) == 0:
        raise RuntimeError("No usable examples found. Check images_root / min_items.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - args.val_ratio))
    train_ex = examples[:split]
    val_ex = examples[split:]

    tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CpDataset(train_ex, tfm)
    val_ds = CpDataset(val_ex, tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cp,
    )

    model = OutfitCompatModel(
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        freeze_backbone=args.freeze_backbone,
        arch=args.arch,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, mask, y in train_loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(x, mask)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(y.numel())
            seen += int(y.numel())
        train_loss = running / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        args_dict = {k: v for k, v in vars(args).items() if k != "func"}
        torch.save({"model_state": model.state_dict(), "args": args_dict}, args.save_path)
        print(f"saved={os.path.abspath(args.save_path)}")


@dataclasses.dataclass(frozen=True)
class FitbExample:
    question_paths: Tuple[str, ...]
    candidate_paths: Tuple[str, ...]


def load_fitb_test_examples(
    *,
    polyvore_tar_path: str,
    image_index: dict[str, str],
    num_candidates: int,
) -> tuple[list[FitbExample], int]:
    import json

    examples: list[FitbExample] = []
    with tarfile.open(polyvore_tar_path) as tar:
        member = resolve_polyvore_member_name(
            tar,
            [
                "fill_in_blank_test.json",
                "fill_in_the_blank_test.json",
            ],
        )
        raw = tar.extractfile(member)
        if raw is None:
            raise FileNotFoundError(member)
        data = json.load(raw)
        total = len(data)
        for q in data:
            question_ids = q["question"]
            answers = q["answers"][:num_candidates]
            if len(answers) != num_candidates:
                continue
            question_paths: list[str] = []
            ok = True
            for item_id in question_ids:
                p = image_index.get(f"{item_id}.jpg".lower())
                if p is None:
                    ok = False
                    break
                question_paths.append(p)
            if not ok:
                continue
            candidate_paths: list[str] = []
            for item_id in answers:
                p = image_index.get(f"{item_id}.jpg".lower())
                if p is None:
                    ok = False
                    break
                candidate_paths.append(p)
            if not ok:
                continue
            examples.append(FitbExample(question_paths=tuple(question_paths), candidate_paths=tuple(candidate_paths)))
    return examples, total


def build_fitb_train_examples(
    *,
    polyvore_tar_path: str,
    image_index: dict[str, str],
    num_candidates: int,
    min_items: int,
    max_items: int,
    num_samples: int,
    seed: int,
) -> list[FitbExample]:
    import json

    rng = random.Random(seed)
    with tarfile.open(polyvore_tar_path) as tar:
        raw = tar.extractfile("train_no_dup.json")
        if raw is None:
            raise FileNotFoundError("train_no_dup.json")
        outfits = json.load(raw)

    slot_pool: dict[str, list[str]] = {}
    usable_outfits: list[list[str]] = []
    for o in outfits:
        sid = str(o.get("set_id"))
        paths: list[str] = []
        for it in o.get("items", [])[:max_items]:
            fn = f"{sid}_{it.get('index')}.jpg".lower()
            p = image_index.get(fn)
            if p is None:
                continue
            paths.append(p)
            slot = slot_from_image_path(p)
            slot_pool.setdefault(slot, []).append(p)
        if len(paths) >= min_items:
            usable_outfits.append(paths)

    for slot, arr in slot_pool.items():
        uniq = list(dict.fromkeys(arr))
        slot_pool[slot] = uniq

    examples: list[FitbExample] = []
    attempts = 0
    max_attempts = num_samples * 50
    while len(examples) < num_samples and attempts < max_attempts:
        attempts += 1
        outfit = rng.choice(usable_outfits)
        outfit_sorted = sort_outfit_paths(outfit)
        if len(outfit_sorted) < min_items:
            continue
        pos_idx = rng.randrange(len(outfit_sorted))
        pos_path = outfit_sorted[pos_idx]
        slot = slot_from_image_path(pos_path)
        pool = slot_pool.get(slot, [])
        if len(pool) < num_candidates:
            continue
        negs: list[str] = []
        while len(negs) < (num_candidates - 1):
            cand = rng.choice(pool)
            if cand == pos_path:
                continue
            if cand in negs:
                continue
            negs.append(cand)
        question = [p for i, p in enumerate(outfit_sorted) if i != pos_idx]
        candidates = [pos_path, *negs]
        examples.append(FitbExample(question_paths=tuple(question), candidate_paths=tuple(candidates)))
    return examples


class FitbDataset(Dataset[FitbExample]):
    def __init__(self, examples: Sequence[FitbExample], transform: transforms.Compose):
        self._examples = list(examples)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self._examples[idx]
        candidate_outfits: list[list[str]] = []
        for cand_path in ex.candidate_paths:
            outfit_paths = sort_outfit_paths([*ex.question_paths, cand_path])
            candidate_outfits.append(outfit_paths)

        num_candidates = len(candidate_outfits)
        lengths = [len(o) for o in candidate_outfits]
        max_len = max(lengths)
        images = torch.zeros((num_candidates, max_len, 3, 0, 0), dtype=torch.float32)
        masks = torch.zeros((num_candidates, max_len), dtype=torch.bool)

        first_img = Image.open(candidate_outfits[0][0]).convert("RGB")
        t0 = self._transform(first_img)
        c, h, w = t0.shape
        images = torch.zeros((num_candidates, max_len, c, h, w), dtype=t0.dtype)

        for i, outfit_paths in enumerate(candidate_outfits):
            for j, p in enumerate(outfit_paths):
                img = Image.open(p).convert("RGB")
                images[i, j] = self._transform(img)
                masks[i, j] = True
        target = torch.tensor(0, dtype=torch.long)
        return images, masks, target


def collate_fitb(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, masks, targets = zip(*batch)
    num_candidates = xs[0].shape[0]
    max_len = max(x.shape[1] for x in xs)
    c, h, w = xs[0].shape[2], xs[0].shape[3], xs[0].shape[4]
    batch_x = torch.zeros((len(xs), num_candidates, max_len, c, h, w), dtype=xs[0].dtype)
    batch_mask = torch.zeros((len(xs), num_candidates, max_len), dtype=torch.bool)
    for i, (x, m) in enumerate(zip(xs, masks)):
        n = x.shape[1]
        batch_x[i, :, :n] = x
        batch_mask[i, :, :n] = m
    batch_y = torch.stack(targets, dim=0)
    return batch_x, batch_mask, batch_y


def run_eval_fitb(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_index = build_image_index(args.images_root)
    examples, total_questions = load_fitb_test_examples(
        polyvore_tar_path=args.polyvore_tar,
        image_index=image_index,
        num_candidates=args.num_candidates,
    )
    if len(examples) == 0:
        raise RuntimeError("No usable FITB test examples found (likely due to missing images).")

    tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = FitbDataset(examples, tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fitb)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = OutfitCompatModel(
        backbone_name=ckpt["args"]["backbone"],
        embed_dim=ckpt["args"]["embed_dim"],
        freeze_backbone=False,
        arch=ckpt["args"].get("arch", "mean"),
        lstm_hidden=ckpt["args"].get("lstm_hidden", 256),
        lstm_layers=ckpt["args"].get("lstm_layers", 1),
        lstm_dropout=ckpt["args"].get("lstm_dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, mask, target in loader:
            b, cands, n, ch, h, w = x.shape
            x = x.to(device).view(b * cands, n, ch, h, w)
            mask = mask.to(device).view(b * cands, n)
            logits = model(x, mask).view(b, cands)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred.cpu() == target).sum().item())
            total += int(target.numel())
    print(
        f"fitb_test_total={total_questions} fitb_test_usable={len(examples)} coverage={(len(examples) / max(1, total_questions)):.4f} acc={correct / max(1, total):.4f}"
    )


def run_train_fitb(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    image_index = build_image_index(args.images_root)
    examples = build_fitb_train_examples(
        polyvore_tar_path=args.polyvore_tar,
        image_index=image_index,
        num_candidates=args.num_candidates,
        min_items=args.min_items,
        max_items=args.max_items,
        num_samples=args.train_samples,
        seed=args.seed,
    )
    if len(examples) == 0:
        raise RuntimeError("No usable FITB train examples found. Check images_root / min_items.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - args.val_ratio))
    train_ex = examples[:split]
    val_ex = examples[split:]

    tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = FitbDataset(train_ex, tfm)
    val_ds = FitbDataset(val_ex, tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fitb,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fitb,
    )

    model = OutfitCompatModel(
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        freeze_backbone=args.freeze_backbone,
        arch=args.arch,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, mask, target in train_loader:
            b, cands, n, ch, h, w = x.shape
            x = x.to(device).view(b * cands, n, ch, h, w)
            mask = mask.to(device).view(b * cands, n)
            target = target.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask).view(b, cands)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(target.numel())
            seen += int(target.numel())
        train_loss = running / max(1, seen)

        model.eval()
        correct = 0
        total = 0
        val_losses: list[float] = []
        with torch.no_grad():
            for x, mask, target in val_loader:
                b, cands, n, ch, h, w = x.shape
                x = x.to(device).view(b * cands, n, ch, h, w)
                mask = mask.to(device).view(b * cands, n)
                target = target.to(device)
                logits = model(x, mask).view(b, cands)
                loss = loss_fn(logits, target)
                val_losses.append(float(loss.item()))
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == target).sum().item())
                total += int(target.numel())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_acc = correct / max(1, total)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        args_dict = {k: v for k, v in vars(args).items() if k != "func"}
        torch.save({"model_state": model.state_dict(), "args": args_dict}, args.save_path)
        print(f"saved={os.path.abspath(args.save_path)}")


def add_args(parser: argparse.ArgumentParser, specs: Sequence[tuple[Sequence[str], dict[str, Any]]]) -> None:
    for flags, kwargs in specs:
        parser.add_argument(*flags, **kwargs)


def add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    add_args(
        parser,
        [
            (["--images-root"], {"required": True}),
            (["--polyvore-tar"], {"default": os.path.join("polyvore-dataset-master", "polyvore.tar.gz")}),
            (["--min-items"], {"type": int, "default": 3}),
            (["--max-items"], {"type": int, "default": 8}),
        ],
    )


def add_model_args(parser: argparse.ArgumentParser, *, default_arch: str) -> None:
    add_args(
        parser,
        [
            (["--image-size"], {"type": int, "default": 224}),
            (["--backbone"], {"choices": ["resnet18", "resnet50"], "default": "resnet18"}),
            (["--embed-dim"], {"type": int, "default": 256}),
            (["--arch"], {"choices": ["mean", "bilstm"], "default": default_arch}),
            (["--lstm-hidden"], {"type": int, "default": 256}),
            (["--lstm-layers"], {"type": int, "default": 1}),
            (["--lstm-dropout"], {"type": float, "default": 0.0}),
        ],
    )


def add_train_args(parser: argparse.ArgumentParser, *, default_batch_size: int) -> None:
    add_args(
        parser,
        [
            (["--freeze-backbone"], {"action": "store_true"}),
            (["--epochs"], {"type": int, "default": 3}),
            (["--batch-size"], {"type": int, "default": default_batch_size}),
            (["--lr"], {"type": float, "default": 3e-4}),
            (["--weight-decay"], {"type": float, "default": 1e-4}),
            (["--val-ratio"], {"type": float, "default": 0.2}),
            (["--seed"], {"type": int, "default": 42}),
            (["--num-workers"], {"type": int, "default": 0}),
            (["--cpu"], {"action": "store_true"}),
            (["--save-path"], {"default": ""}),
        ],
    )


def build_default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def list_closet_images_by_slot(closet_root: str) -> dict[str, list[str]]:
    closet_root = os.path.abspath(closet_root)
    out: dict[str, list[str]] = {}
    for name in os.listdir(closet_root):
        slot_dir = os.path.join(closet_root, name)
        if not os.path.isdir(slot_dir):
            continue
        slot = name.lower()
        paths: list[str] = []
        for dirpath, _, filenames in os.walk(slot_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                paths.append(os.path.join(dirpath, fn))
        paths.sort(key=lambda p: os.path.basename(p).lower())
        if paths:
            out[slot] = paths
    return out


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> tuple[OutfitCompatModel, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = OutfitCompatModel(
        backbone_name=ckpt_args.get("backbone", "resnet18"),
        embed_dim=int(ckpt_args.get("embed_dim", 256)),
        freeze_backbone=False,
        arch=ckpt_args.get("arch", "mean"),
        lstm_hidden=int(ckpt_args.get("lstm_hidden", 256)),
        lstm_layers=int(ckpt_args.get("lstm_layers", 1)),
        lstm_dropout=float(ckpt_args.get("lstm_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_args


def score_outfit_paths(
    *,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    outfit_paths: Sequence[str],
) -> float:
    outfit_paths = sort_outfit_paths(outfit_paths)
    images: list[torch.Tensor] = []
    for p in outfit_paths:
        img = Image.open(p).convert("RGB")
        images.append(transform(img))
    x = torch.stack(images, dim=0).unsqueeze(0).to(device)
    mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
    with torch.no_grad():
        logit = model(x, mask)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def parse_slots_arg(slots: Optional[Sequence[str]]) -> list[str]:
    if not slots:
        return ["top", "pants", "shoes"]
    expanded: list[str] = []
    for s in slots:
        for part in s.split(","):
            p = part.strip().lower()
            if p:
                expanded.append(p)
    uniq: list[str] = []
    for s in expanded:
        if s not in uniq:
            uniq.append(s)
    return uniq


def run_recommend_closet(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rng = random.Random(args.seed)

    closet = list_closet_images_by_slot(args.closet_root)
    slots = parse_slots_arg(args.slots)
    fixed: dict[str, str] = {}
    for pair in args.fixed_item or []:
        slot, path = pair[0].lower(), pair[1]
        fixed[slot] = os.path.abspath(path)

    missing_slots = [s for s in slots if s not in fixed and s not in closet]
    if missing_slots:
        raise RuntimeError(f"Missing closet images for slots: {missing_slots}")

    model, ckpt_args = load_checkpoint_model(args.checkpoint, device)
    image_size = int(args.image_size or ckpt_args.get("image_size", 224))
    transform = build_default_transform(image_size)

    pools: dict[str, list[str]] = {}
    for s in slots:
        if s in fixed:
            pools[s] = [fixed[s]]
            continue
        arr = list(closet.get(s, []))
        if args.max_candidates_per_slot and len(arr) > args.max_candidates_per_slot:
            arr = rng.sample(arr, k=int(args.max_candidates_per_slot))
        pools[s] = arr
        if not pools[s]:
            raise RuntimeError(f"No candidates for slot={s}")

    total_combinations = 1
    for s in slots:
        total_combinations *= max(1, len(pools[s]))

    def sample_outfit() -> list[str]:
        return [rng.choice(pools[s]) for s in slots]

    if args.max_combinations and total_combinations <= int(args.max_combinations):
        iterable = itertools.product(*[pools[s] for s in slots])
        outfits = (list(t) for t in iterable)
    else:
        outfits = (sample_outfit() for _ in range(int(args.num_samples)))

    heap: list[tuple[float, list[str]]] = []
    k = int(args.top_k)
    for outfit in outfits:
        score = score_outfit_paths(model=model, transform=transform, device=device, outfit_paths=outfit)
        if len(heap) < k:
            heapq.heappush(heap, (score, outfit))
        else:
            heapq.heappushpop(heap, (score, outfit))

    best = sorted(heap, key=lambda x: x[0], reverse=True)
    print(f"device={device.type} slots={slots} total_combinations={total_combinations}")
    for i, (score, outfit) in enumerate(best, start=1):
        print(f"rank={i} score={score:.4f} " + " ".join(outfit))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    stats = sub.add_parser("stats")
    add_shared_data_args(stats)
    stats.set_defaults(func=run_stats)

    train_cp = sub.add_parser("train-cp")
    add_shared_data_args(train_cp)
    add_model_args(train_cp, default_arch="mean")
    add_train_args(train_cp, default_batch_size=16)
    train_cp.set_defaults(func=run_train_cp)

    train_fitb = sub.add_parser("train-fitb")
    add_shared_data_args(train_fitb)
    add_args(
        train_fitb,
        [
            (["--num-candidates"], {"type": int, "default": 4}),
            (["--train-samples"], {"type": int, "default": 20000}),
        ],
    )
    add_model_args(train_fitb, default_arch="bilstm")
    add_train_args(train_fitb, default_batch_size=8)
    train_fitb.set_defaults(func=run_train_fitb)

    eval_fitb = sub.add_parser("eval-fitb")
    add_args(
        eval_fitb,
        [
            (["--images-root"], {"required": True}),
            (["--checkpoint"], {"required": True}),
            (["--polyvore-tar"], {"default": os.path.join("polyvore-dataset-master", "polyvore.tar.gz")}),
            (["--num-candidates"], {"type": int, "default": 4}),
            (["--image-size"], {"type": int, "default": 224}),
            (["--batch-size"], {"type": int, "default": 8}),
            (["--num-workers"], {"type": int, "default": 0}),
            (["--cpu"], {"action": "store_true"}),
        ],
    )
    eval_fitb.set_defaults(func=run_eval_fitb)

    recommend = sub.add_parser("recommend-closet")
    recommend.add_argument("--closet-root", required=True)
    recommend.add_argument("--checkpoint", required=True)
    recommend.add_argument("--slots", nargs="*", default=[])
    recommend.add_argument("--fixed-item", nargs=2, action="append", default=[])
    recommend.add_argument("--top-k", type=int, default=10)
    recommend.add_argument("--max-candidates-per-slot", type=int, default=50)
    recommend.add_argument("--max-combinations", type=int, default=5000)
    recommend.add_argument("--num-samples", type=int, default=2000)
    recommend.add_argument("--image-size", type=int, default=0)
    recommend.add_argument("--seed", type=int, default=42)
    recommend.add_argument("--cpu", action="store_true")
    recommend.set_defaults(func=run_recommend_closet)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
