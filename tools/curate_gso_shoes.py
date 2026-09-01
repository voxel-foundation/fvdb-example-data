# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Curate the Google Scanned Objects (GSO) 'Shoe' subset for fvdb-example-data.

Downloads shoe-category models by GoogleResearch from Gazebo Fuel (all CC-BY 4.0),
extracts the scan mesh, decimates it to a lean face budget, and writes binary PLYs
plus an ATTRIBUTION.json manifest recording name, source URL, license, and face counts.

Usage:
    python curate_gso_shoes.py --out DIR [--limit N] [--max-faces 20000]
"""

import argparse
import io
import json
import logging
import pathlib
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import point_cloud_utils as pcu

FUEL_API = "https://fuel.gazebosim.org/1.0"
OWNER = "GoogleResearch"
CATEGORY = "Shoe"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"


def http_get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries - 1:
                raise  # 404 is definitive (also marks end of pagination); retry other codes
            logging.warning(f"retrying {url} after HTTP {e.code}")
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:  # noqa: BLE001 - retry any transient network error
            if attempt == retries - 1:
                raise
            logging.warning(f"retrying {url} after error: {e}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_shoe_models() -> list[dict]:
    """Paginate the full GoogleResearch model list and keep Shoe-category entries."""
    models, page = [], 1
    while True:
        try:
            data = http_get(f"{FUEL_API}/{OWNER}/models?page={page}&per_page=100")
        except urllib.error.HTTPError as e:
            if e.code == 404:  # Fuel returns 404 past the last page
                break
            raise
        batch = json.loads(data)
        if not batch:
            break
        for m in batch:
            if (
                m.get("owner") == OWNER
                and CATEGORY in (m.get("categories") or [])
                and m.get("license_name") == LICENSE_NAME
            ):
                models.append(m)
        page += 1
    # Deterministic order for reproducible curation.
    models.sort(key=lambda m: m["name"])
    return models


def extract_obj(zip_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Pull the scan mesh (meshes/model.obj) out of a Fuel model zip and load it."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        obj_names = [n for n in zf.namelist() if n.lower().endswith("model.obj")]
        if not obj_names:
            obj_names = [n for n in zf.namelist() if n.lower().endswith(".obj")]
        if not obj_names:
            raise ValueError("no .obj in archive")
        # pcu loads from a path, so write to a temp buffer file.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".obj") as tmp:
            tmp.write(zf.read(obj_names[0]))
            tmp.flush()
            v, f = pcu.load_mesh_vf(tmp.name)
    if v is None or f is None:
        raise ValueError("mesh missing vertices or faces")
    return np.ascontiguousarray(v, dtype=np.float32), np.ascontiguousarray(f, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="curate only the first N shoes (0 = all)")
    parser.add_argument("--max-faces", type=int, default=20_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    models = list_shoe_models()
    logging.info(f"Fuel lists {len(models)} CC-BY 4.0 '{CATEGORY}' models owned by {OWNER}")
    if args.limit:
        models = models[: args.limit]

    manifest = {
        "source": "Google Scanned Objects via Gazebo Fuel",
        "collection_url": f"https://app.gazebosim.org/{OWNER}",
        "license": LICENSE_NAME,
        "license_url": "http://creativecommons.org/licenses/by/4.0/",
        "attribution": "Scanned Objects by Google Research, used under CC-BY 4.0",
        "category": CATEGORY,
        "curated_with": "curate_gso_shoes.py (fvdb-core)",
        "models": [],
    }

    for i, m in enumerate(models):
        name = m["name"]
        url = f"{FUEL_API}/{OWNER}/models/{name}/tip/{name}.zip"
        try:
            v, f = extract_obj(http_get(url))
        except Exception as e:  # noqa: BLE001 - skip individual bad archives, keep curating
            logging.warning(f"[{i + 1}/{len(models)}] SKIP {name}: {e}")
            continue
        n_faces_in = f.shape[0]
        if n_faces_in > args.max_faces:
            v_d, f_d, _, _ = pcu.decimate_triangle_mesh(v.astype(np.float64), f, args.max_faces)
            v, f = v_d.astype(np.float32), np.ascontiguousarray(f_d, dtype=np.int32)
        ply_name = f"{name}.ply"
        pcu.save_mesh_vf(str(args.out / ply_name), v, f)
        manifest["models"].append(
            {
                "file": ply_name,
                "name": name,
                "description": (m.get("description") or "").split("\n")[0],
                "source_url": f"https://app.gazebosim.org/{OWNER}/fuel/models/{name}",
                "faces_original": n_faces_in,
                "faces_stored": int(f.shape[0]),
            }
        )
        logging.info(f"[{i + 1}/{len(models)}] {name}: {n_faces_in} -> {f.shape[0]} faces")

    with open(args.out / "ATTRIBUTION.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    total_mb = sum(p.stat().st_size for p in args.out.glob("*.ply")) / 2**20
    logging.info(f"curated {len(manifest['models'])} meshes, {total_mb:.1f} MB total -> {args.out}")


if __name__ == "__main__":
    main()
