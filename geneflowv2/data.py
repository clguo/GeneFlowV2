from __future__ import annotations
import logging
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import torch
from PIL import Image as PILImage
from torchvision import transforms
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)
def normalize_rgb(rgb_image):
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = ((rgb_image - np.min(rgb_image) + 1e-6) / (np.max(rgb_image) - np.min(rgb_image) + 1e-6))
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def parse_adata(args=None, 
                adata=None,
                layer=None,
                cell_type=None, 
                exclude_cell_type=None,
                cell_type_label=None, 
                min_total_counts=None, 
                max_total_counts=None, 
                min_total_pct=None, 
                max_total_pct=None,

                gene_symbols=None,
                missing_gene_symbols=None,
                nsamples_test=None,
                ):

    if args is not None:
        if adata is None and args.adata is not None:
            adata = args.adata
        if layer is None and args.layer is not None:
            layer = args.layer
        if cell_type is None and args.cell_type is not None:
            cell_type = args.cell_type
        if exclude_cell_type is None and args.exclude_cell_type is not None:
            exclude_cell_type = args.exclude_cell_type
        if cell_type_label is None:
            cell_type_label = args.cell_type_label
        if min_total_counts is None and args.min_total_counts is not None:
            min_total_counts = args.min_total_counts
        if max_total_counts is None and args.max_total_counts is not None:
            max_total_counts = args.max_total_counts
        if min_total_pct is None and args.min_total_pct is not None:
            min_total_pct = args.min_total_pct
        if max_total_pct is None and args.max_total_pct is not None:
            max_total_pct = args.max_total_pct
        if gene_symbols is None:
            gene_symbols = args.gene_symbols
        if missing_gene_symbols is None:
            missing_gene_symbols = args.missing_gene_symbols
        if nsamples_test is None:
            nsamples_test = args.nsamples_test
    

    if type(adata) is str:
        adata = sc.read_h5ad(adata)
        logger.info(f"Loaded AnnData object from {adata}")
        logger.info(f"AnnData object has {adata.n_obs} cells and {adata.n_vars} genes")
    
    if layer is not None:
        adata.X = adata.layers[layer]

    if cell_type is not None:
        logger.info(f"Filtering cells with cell type {cell_type}")
        adata = adata[adata.obs[cell_type_label].isin(cell_type)]
        logger.info(f"{len(adata)} cells with cell type {cell_type} passed the filter")

    if exclude_cell_type is not None:
        logger.info(f"Filtering cells other than cell type {exclude_cell_type}")
        adata = adata[~adata.obs[cell_type_label].isin(exclude_cell_type)]
        logger.info(f"{len(adata)} cells other than cell type {exclude_cell_type} passed the filter")

    if min_total_counts is not None and min_total_counts > 0:
        logger.info(f"Filtering cells with total counts < {min_total_counts}")
        adata = adata[adata.obs["total_counts"] >= min_total_counts]
        logger.info(f"{len(adata)} cells with total counts > {min_total_counts} passed the filter")
    
    if max_total_counts is not None and max_total_counts < np.inf:
        logger.info(f"Filtering cells with total counts > {max_total_counts}")
        adata = adata[adata.obs["total_counts"] <= max_total_counts]
        logger.info(f"{len(adata)} cells with total counts < {max_total_counts} passed the filter")
    
    if min_total_pct is not None and min_total_pct > 0.0:
        logger.info(f"Filtering cells with total pct < {min_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], min_total_pct * 100)
        adata = adata[adata.obs["total_counts"] >= threshold]
    
    if max_total_pct is not None and max_total_pct < 1.0:
        logger.info(f"Filtering cells with total pct > {max_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], max_total_pct * 100)
        adata = adata[adata.obs["total_counts"] <= threshold]
        logger.info(f"{len(adata)} cells with total pct < {max_total_pct * 100}% passed the filter")

    if missing_gene_symbols is not None and os.path.isfile(missing_gene_symbols):
        missing_gene_symbols = pd.read_csv(missing_gene_symbols, header=None)[0].tolist()
        logger.info(f"Loaded {len(missing_gene_symbols)} missing gene symbols from {args.missing_gene_symbols}")
    else:
        missing_gene_symbols = set()

    if gene_symbols is not None and os.path.isfile(gene_symbols):
        gene_symbols = pd.read_csv(gene_symbols, header=None)[0].tolist()

    if nsamples_test is not None and nsamples_test > 0:
        logger.info(f"Subsampling {nsamples_test} cells for testing")
        sc.pp.subsample(adata, n_obs=nsamples_test)
        logger.info(f"Subsampled {len(adata)} cells for testing")

    ngenes = adata.n_vars
    genes = adata.var_names.tolist()
    expr = adata.to_df()


    if gene_symbols is not None and len(gene_symbols) > 0:
        ngenes = len(gene_symbols)
        genes = gene_symbols
        expr = pd.DataFrame(np.zeros((adata.n_obs, ngenes)), index=adata.obs_names, columns=gene_symbols)
        expr.update(adata.to_df())
        missing_gene_symbols = list(set(missing_gene_symbols) | (set(gene_symbols) - set(adata.var_names)))
    
    return expr, missing_gene_symbols

class CellImageGeneDataset(Dataset):


    def __init__(self, expr_df, image_paths, img_size=256, img_channels=3,
                 transform=None, missing_gene_symbols=None, normalize_aux=False,
                 strict_images=True):
        self.expr_df = expr_df
        self.gene_list = expr_df.columns.tolist()
        self.normalize_aux = normalize_aux
        self.strict_images = strict_images


        self.image_paths = image_paths


        common_cells = set(self.expr_df.index) & set(self.image_paths.keys())


        self.cell_ids = sorted(common_cells, key=str)
        logger.info(f"Dataset contains {len(self.cell_ids)} cells with both expression data and images")

        self.img_size = img_size
        self.img_channels = img_channels

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ])
        else:
            self.transform = transform

        self.missing_gene_symbols = missing_gene_symbols
        self.missing_gene_indices = None
        if self.missing_gene_symbols:
            self.missing_gene_indices = {gene: idx for idx, gene in enumerate(self.gene_list)
                                         if gene in self.missing_gene_symbols}
            logger.info(f"Initialized dataset with {len(self.missing_gene_indices)} missing gene indices identified.")
        else:
            logger.info("Initialized dataset with no missing gene symbols provided or found in data.")

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        gene_expr = self.expr_df.loc[cell_id].values.astype(np.float32)
        gene_mask = np.ones_like(gene_expr)
        if self.missing_gene_indices:
            indices_to_zero = list(self.missing_gene_indices.values())
            if indices_to_zero:
                gene_mask[indices_to_zero] = 0

        img_source = self.image_paths[cell_id]

        if isinstance(img_source, np.ndarray):
            patch = img_source
            if self.strict_images and (
                patch.ndim != 3 or patch.shape[-1] < self.img_channels
            ):
                raise ValueError(
                    f"Image for cell {cell_id} has shape {patch.shape}; expected "
                    f"[H, W, >= {self.img_channels}]"
                )
            if patch.shape[-1] != self.img_channels:
                patch = patch[..., :self.img_channels]
            if patch.dtype != np.uint8:
                patch = normalize_rgb(patch)
            pil_img = PILImage.fromarray(patch)
            image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        else:

            try:
                if str(img_source).lower().endswith(('.h5', '.hdf5')):
                    with h5py.File(img_source, 'r') as handle:
                        image = handle['image'][:]
                else:
                    image = tifffile.imread(img_source)
                if self.strict_images and (
                    image.ndim != 3 or image.shape[-1] < self.img_channels
                ):
                    raise ValueError(
                        f"Image has shape {image.shape}; expected "
                        f"[H, W, >= {self.img_channels}]"
                    )
                image = image[:, :, :self.img_channels]
                if image.dtype != np.uint8:
                    image = normalize_rgb(image)
                pil_img = PILImage.fromarray(image)
                image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
            except Exception as e:
                logger.error(f"Error loading image {img_source}: {e}")
                if self.strict_images:
                    raise RuntimeError(
                        f"Strict image loading failed for cell {cell_id}: {img_source}"
                    ) from e
                pil_img = PILImage.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
                image = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)

        if self.strict_images:
            if image.ndim != 3 or image.shape[0] != self.img_channels:
                raise ValueError(
                    f"Transformed image for cell {cell_id} has shape {tuple(image.shape)}; "
                    f"expected [{self.img_channels}, H, W]"
                )
            if not torch.isfinite(image).all():
                raise ValueError(f"Transformed image for cell {cell_id} is non-finite")

        return {
            'cell_id': cell_id,
            'gene_expr': gene_expr,
            'gene_mask': gene_mask,
            'image': image
        }
