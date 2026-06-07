#!/usr/bin/env python3
"""
Filter trajectories by path_id set loaded from an indices JSON file.

Usage:
  python filter_trajs_by_pathid.py <indices.json> <trajectories.json> <out_dir>

Writes: <out_dir>/<trajectory_filename>_filtered_by_indices.json
"""
import sys
import json
from pathlib import Path


def load_indices(p):
    data = json.loads(Path(p).read_text())
    s = set()
    for v in data:
        if isinstance(v, (int, float)):
            s.add(int(v))
        elif isinstance(v, str) and v.isdigit():
            s.add(int(v))
        elif isinstance(v, dict):
            for key in ('index', 'path_id', 'pathId', 'id', 'pathid'):
                if key in v:
                    try:
                        s.add(int(v[key])); break
                    except Exception:
                        pass
    return s


def main():
    if len(sys.argv) < 4:
        print("Usage: python filter_trajs_by_pathid.py <indices.json> <trajectories.json> <out_dir>")
        sys.exit(1)
    idx_p = Path(sys.argv[1])
    traj_p = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    if not idx_p.exists():
        print('Indices file not found:', idx_p); sys.exit(2)
    if not traj_p.exists():
        print('Trajectories file not found:', traj_p); sys.exit(3)

    indices = load_indices(idx_p)
    print(f'Loaded {len(indices)} indices')

    data = json.loads(traj_p.read_text())
    if not isinstance(data, list):
        print('Expected trajectories file to be a JSON array'); sys.exit(4)

    out = []
    for item in data:
        pid = None
        if isinstance(item, dict):
            for k in ('path_id', 'pathId', 'pathid', 'id'):
                if k in item:
                    pid = item[k]; break
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except Exception:
            continue
        if pid_int in indices:
            out.append(item)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = out_dir / (traj_p.name + '_filtered_by_indices.json')
    out_name.write_text(json.dumps(out, ensure_ascii=False))
    print(f'Wrote {len(out)} trajectories to {out_name}')


if __name__ == '__main__':
    main()
