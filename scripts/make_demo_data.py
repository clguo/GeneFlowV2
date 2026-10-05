import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import tifffile


def main():
    parser = argparse.ArgumentParser(description="Create synthetic data for a CPU smoke run.")
    parser.add_argument("--output_dir", default="demo_data")
    args = parser.parse_args()
    output = Path(args.output_dir)
    (output / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    ids = [f"cell_{i:03d}" for i in range(24)]
    expression = rng.poisson(1.0, size=(24, 16)).astype(np.float32)
    adata = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(index=[f"gene_{i:02d}" for i in range(16)]),
    )
    adata.obsm["spatial"] = np.array(
        [(i % 6 * 64, i // 6 * 64) for i in range(24)], dtype=np.float32
    )
    adata.write_h5ad(output / "expression.h5ad")
    paths = {}
    for i, cell_id in enumerate(ids):
        color = expression[i, :3] / (expression[i, :3].max() + 1)
        image = np.clip(color + rng.normal(0, 0.1, (16, 16, 3)), 0, 1)
        relative = f"images/{cell_id}.tif"
        tifffile.imwrite(output / relative, (image * 255).astype(np.uint8), photometric="rgb")
        paths[cell_id] = relative
    (output / "image_paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    print(f"Synthetic dataset saved to {output.resolve()}")


if __name__ == "__main__":
    main()
