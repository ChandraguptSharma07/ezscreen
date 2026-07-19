from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[1]
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))


def _vec(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('expected three comma-separated numbers')
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('box values must be numeric') from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-id', required=True)
    parser.add_argument('--search-mode', default='balance', choices=['fast', 'balance', 'detail'])
    parser.add_argument('--box-center', type=_vec, default=[15.656, 2.308, 39.483])
    parser.add_argument('--box-size', type=_vec, default=[21.337, 22.566, 22.752])
    parser.add_argument('--accelerator', default='nvidiaTeslaT4')
    parser.add_argument('--receptor-pdbqt', type=Path, default=AUDIT / 'inputs' / 'receptor_fixed.pdbqt')
    parser.add_argument('--redock-ligand-pdbqt', type=Path, default=AUDIT / 'redocking' / 'cocrystal_2znp.pdbqt')
    parser.add_argument('--panel-smi', type=Path, default=AUDIT / 'audit_panel' / 'audit_panel.smi')
    parser.add_argument('--panel-label-csv', type=Path, default=AUDIT / 'audit_panel' / 'audit_panel.csv')
    parser.add_argument('--pose-receptor-pdb', type=Path, default=AUDIT / 'inputs' / 'receptor_prep.pdb')
    args = parser.parse_args()

    from scripts import submit_panel_kaggle, submit_redocking_kaggle

    submit_redocking_kaggle.BOX_CENTER = args.box_center
    submit_redocking_kaggle.BOX_SIZE = args.box_size
    submit_panel_kaggle.BOX_CENTER = args.box_center
    submit_panel_kaggle.BOX_SIZE = args.box_size

    redock = submit_redocking_kaggle.submit(args.ablation_id, args.search_mode, args.accelerator)
    panel = submit_panel_kaggle.submit(args.ablation_id, args.search_mode, args.accelerator)
    out = {
        'ablation_id': args.ablation_id,
        'search_mode': args.search_mode,
        'box_center': args.box_center,
        'box_size': args.box_size,
        'receptor_pdbqt': str(args.receptor_pdbqt),
        'redock_ligand_pdbqt': str(args.redock_ligand_pdbqt),
        'panel_smi': str(args.panel_smi),
        'panel_label_csv': str(args.panel_label_csv),
        'pose_receptor_pdb': str(args.pose_receptor_pdb),
        'redock': redock,
        'panel': panel,
    }
    path = AUDIT / 'logs'
    path.mkdir(parents=True, exist_ok=True)
    (path / f'ablation_pair_{args.ablation_id}.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()