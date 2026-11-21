"""Inspect dataset ZIP archives WITHOUT extracting media or trusting paths.

Halmstad source filenames group multiple frames belonging to one video; all
splits are at video/group level, not frame level. A split is a proposal that
must be frozen before training or tuning.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import json
import re

CLASSES = ('DRONE', 'BIRD', 'AIRPLANE', 'HELICOPTER')
MODES = ('Video_IR', 'Video_V')
_VIDEO_RE = re.compile(r'^(IR|V)_(DRONE|BIRD|AIRPLANE|HELICOPTER)_(\d+)\.mp4$', re.I)


def hash_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open('rb') as f:
        while chunk := f.read(block_bytes):
            h.update(chunk)
    return h.hexdigest()


def _safe_names(z: ZipFile) -> list[str]:
    names = z.namelist()
    for name in names:
        p = PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or '\\' in name or
                not p.parts or name.startswith('/')):
            raise ValueError(f'Unsafe ZIP path: {name!r}')
    return names


def video_split(group: str) -> str:
    """Deterministic strata assignment is performed by stratified_splits."""
    hashnum = int(sha256(('skyguard43-v1|' + group).encode()).hexdigest()[:12], 16)
    value = hashnum % 100
    return 'train' if value < 70 else ('val' if value < 85 else 'test')


def stratified_splits(records: list[dict]) -> None:
    """Preserve source-group independence across modalities. No frame split."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for r in records:
        grouped[r['class']].append(r['group_id'])
    group_splits = {}
    for cls, groups in sorted(grouped.items()):
        unique = sorted(set(groups), key=lambda x: sha256(('skyguard43-v1|' + x).encode()).hexdigest())
        n = len(unique)
        train_end = max(1, round(n * .7))
        val_end = min(n, max(train_end + (1 if n >= 3 else 0), round(n * .85)))
        for i, group in enumerate(unique):
            group_splits[group] = 'train' if i < train_end else ('val' if i < val_end else 'test')
    for r in records:
        r['split'] = group_splits[r['group_id']]


def inspect_halmstad_zip(source: str | Path) -> dict:
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as z:
        names = set(_safe_names(z))
        records = []
        all_videos = 0
        for name in sorted(names):
            if not name.lower().endswith('.mp4'):
                continue
            all_videos += 1
            p = PurePosixPath(name)
            match = _VIDEO_RE.match(p.name)
            if not match or p.parent.name not in MODES:
                continue  # 2 example/demo videos are not ground-truth sequences
            prefix, category, number = match.groups()
            label_path = str(p.with_name(p.stem + '_LABELS.mat'))
            records.append({
                'sequence_id': p.stem,
                'group_id': category.upper() + ':' + number,
                'class': category.upper(),
                'modality': 'IR' if prefix.upper() == 'IR' else 'VISIBLE',
                'video_zip_entry': name,
                'label_zip_entry': label_path if label_path in names else None,
                'video_bytes': z.getinfo(name).file_size,
            })
        stratified_splits(records)
        modality = Counter(r['modality'] for r in records)
        classes = Counter(r['class'] for r in records)
        missing = [r['sequence_id'] for r in records if r['label_zip_entry'] is None]
        if len(records) != 650 or missing:
            # report the evidence instead of silently treating the archive as verified
            integrity = 'FAIL'
        else:
            integrity = 'PASS_INVENTORY_ONLY'
        return {
            'source_kind': 'Halmstad multi-sensor drone dataset',
            'source_zip_path': str(path.resolve()),
            'source_sha256': hash_file(path),
            'license': 'SEE_SOURCE_LICENSE_CC0_1_0',
            'source_reference': 'https://github.com/DroneDetectionThesis/Drone-detection-dataset',
            'annotation_format': 'MATLAB groundTruth MCOS, requires mcos-decoder',
            'counts': {'annotated_videos': len(records), 'all_mp4': all_videos,
                       'modalities': dict(modality), 'classes': dict(classes),
                       'unpaired_videos': len(missing),
                       'splits': dict(Counter(r['split'] for r in records))},
            'inventory_integrity': integrity,
            'records': records,
            'not_validated': ['bbox decoding', 'label/frame agreement', 'model performance', 'field performance'],
        }


def inspect_dut_zip(source: str | Path) -> dict:
    path = Path(source)
    with ZipFile(path) as z:
        names = _safe_names(z)
        media = [n for n in names if n.lower().endswith(('.mp4', '.avi', '.xml', '.txt', '.json'))]
        return {
            'source_kind': 'DUT-Anti-UAV GitHub repository ZIP',
            'source_zip_path': str(path.resolve()),
            'source_sha256': hash_file(path),
            'github_code_license': 'Apache-2.0 (verify external dataset licenses separately)',
            'files': len([n for n in names if not n.endswith('/')]),
            'dataset_annotation_or_video_files': media,
            'status': 'DATASET_NOT_INCLUDED' if not media else 'INSPECTION_REQUIRED',
            'source_reference': 'https://github.com/wangdongdut/DUT-Anti-UAV',
        }


def json_dump_atomic(obj: dict, path: str | Path) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(dst)
