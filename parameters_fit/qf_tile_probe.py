"""Offline closed-loop SSIMULACRA2 optimizer for 64x64 quant-field tiles."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "target/release/jixel-tuner"


@dataclass(frozen=True)
class Result:
    key: str
    bytes: int
    ss2: float
    tx: int = -1
    ty: int = -1
    delta: int = 0


def run(image: Path, distance: float, key: str, deltas: str = "") -> Result:
    with tempfile.TemporaryDirectory(prefix="jixel-qf-") as tmp:
        tmp = Path(tmp)
        jxl = tmp / "out.jxl"
        png = tmp / "out.png"
        env = dict(os.environ)
        if deltas:
            env["JIXEL_QF_TILE_DELTAS"] = deltas
        else:
            env.pop("JIXEL_QF_TILE_DELTAS", None)
        enc = subprocess.run(
            [str(ENCODER), str(image), str(jxl), "-d", str(distance), "--threads", "1", "--speed", "slow"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        info = json.loads(enc.stdout.strip().splitlines()[-1])
        subprocess.run(["djxl", str(jxl), str(png)], capture_output=True, check=True)
        metric = subprocess.run(
            ["ssimulacra2", str(image), str(png)],
            text=True,
            capture_output=True,
            check=True,
        )
        return Result(key, int(info["bytes"]), float(metric.stdout.split()[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--greedy-iterations", type=int, default=0)
    args = parser.parse_args()
    image = args.image.resolve()
    from PIL import Image

    with Image.open(image) as im:
        tiles_x = (im.width + 63) // 64
        tiles_y = (im.height + 63) // 64

    center = run(image, args.distance, "center")
    finer = run(image, args.distance - 0.5, "finer")
    coarser = run(image, args.distance + 0.5, "coarser")
    slope = (finer.ss2 - coarser.ss2) / math.log(finer.bytes / coarser.bytes)
    print(f"baseline bytes={center.bytes} ss2={center.ss2:.6f} slope={slope:.4f}")

    jobs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for delta in (-1, 1):
                    key = f"{tx}:{ty}:{delta}"
                    future = pool.submit(run, image, args.distance, key, key)
                    jobs.append((future, tx, ty, delta))
        results = []
        for future in as_completed([job[0] for job in jobs]):
            job = next(job for job in jobs if job[0] is future)
            result = future.result()
            results.append(Result(result.key, result.bytes, result.ss2, *job[1:]))

    def objective(result: Result) -> float:
        return result.ss2 - center.ss2 - slope * math.log(result.bytes / center.bytes)

    spends = [r for r in results if r.delta == 1]
    saves = [r for r in results if r.delta == -1]
    pairs = []
    for spend in spends:
        for save in saves:
            if (spend.tx, spend.ty) == (save.tx, save.ty):
                continue
            rate = math.log(spend.bytes / center.bytes) + math.log(save.bytes / center.bytes)
            quality = (spend.ss2 - center.ss2) + (save.ss2 - center.ss2)
            gain = quality - slope * rate
            if rate <= 0.0005 and gain > 0:
                pairs.append((gain, spend, save))
    pairs.sort(key=lambda p: p[0], reverse=True)
    selected = []
    used = set()
    for gain, spend, save in pairs:
        coords = {(spend.tx, spend.ty), (save.tx, save.ty)}
        if coords & used:
            continue
        selected.append((gain, spend, save))
        used |= coords

    print("best single candidates:")
    for result in sorted(results, key=objective, reverse=True)[:12]:
        print(
            f"  {result.key:8s} db={result.bytes-center.bytes:+5d} "
            f"dss2={result.ss2-center.ss2:+.6f} obj={objective(result):+.6f}"
        )
    print(f"positive disjoint pair predictions: {len(selected)}")
    for count in (1, 2, 4, 8, 12, 16):
        chosen = selected[:count]
        if len(chosen) < count:
            break
        entries = []
        predicted = 0.0
        for gain, spend, save in chosen:
            predicted += gain
            entries.extend([spend.key, save.key])
        combined = run(image, args.distance, f"pairs-{count}", ",".join(entries))
        actual = objective(combined)
        print(
            f"pairs={count:2d} bytes={combined.bytes} ({combined.bytes-center.bytes:+d}) "
            f"ss2={combined.ss2:.6f} ({combined.ss2-center.ss2:+.6f}) "
            f"pred={predicted:+.6f} actual_obj={actual:+.6f}"
        )
        if count == 12:
            print(f"pairs=12 map={','.join(entries)}")

    if args.greedy_iterations:
        current = center
        current_map: dict[tuple[int, int], int] = {}
        candidates = results
        for iteration in range(args.greedy_iterations):
            def step_objective(result: Result) -> float:
                return result.ss2 - current.ss2 - slope * math.log(result.bytes / current.bytes)

            best = max(candidates, key=step_objective)
            gain = step_objective(best)
            if gain <= 0:
                print(f"greedy stopped at iteration {iteration}: no positive move")
                break
            coord = (best.tx, best.ty)
            current_map[coord] = current_map.get(coord, 0) + best.delta
            if current_map[coord] == 0:
                del current_map[coord]
            current = best
            print(
                f"greedy={iteration+1:2d} move={best.key:8s} bytes={current.bytes} "
                f"ss2={current.ss2:.6f} step_obj={gain:+.6f}"
            )
            jobs = []
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for ty in range(tiles_y):
                    for tx in range(tiles_x):
                        for delta in (-1, 1):
                            candidate_map = dict(current_map)
                            candidate_map[(tx, ty)] = candidate_map.get((tx, ty), 0) + delta
                            candidate_map = {k: v for k, v in candidate_map.items() if v}
                            deltas = ",".join(
                                f"{x}:{y}:{value}" for (x, y), value in sorted(candidate_map.items())
                            )
                            key = f"{tx}:{ty}:{delta}"
                            future = pool.submit(run, image, args.distance, key, deltas)
                            jobs.append((future, tx, ty, delta))
                candidates = []
                for future in as_completed([job[0] for job in jobs]):
                    job = next(job for job in jobs if job[0] is future)
                    result = future.result()
                    candidates.append(Result(result.key, result.bytes, result.ss2, *job[1:]))
        print(
            "greedy final "
            f"bytes={current.bytes} ({current.bytes-center.bytes:+d}) "
            f"ss2={current.ss2:.6f} ({current.ss2-center.ss2:+.6f}) "
            "map="
            + ",".join(f"{x}:{y}:{v}" for (x, y), v in sorted(current_map.items()))
        )


if __name__ == "__main__":
    main()
