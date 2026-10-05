import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import tifffile

from geneflowv2.datasets import INPUT_DIR, SAMPLES, check, extract_archive, prepare
from geneflowv2.run import build_command


class ReleasedDatasetTests(unittest.TestCase):
    def make_sample(self, root, dataset):
        sample = root / 'raw' / SAMPLES[dataset][0]
        images = sample / INPUT_DIR / 'cell_images'
        images.mkdir(parents=True)
        ids = [f'cell_{i:02d}' for i in range(13)]
        genes = [f'g{i}' for i in range(6)]
        a = ad.AnnData(np.arange(78, dtype=np.float32).reshape(13, 6),
                       obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=genes))
        a.obsm['spatial'] = np.array([(i * 512, i % 3 * 512) for i in range(13)])
        a.write_h5ad(sample / 'adata.h5ad')
        mapping = {}
        for i, cell in enumerate(ids[:12]):
            name = f'{cell}_original.tif'
            channels = 9 if dataset == 'P1' else 6
            tifffile.imwrite(images / name, np.full((8, 8, channels), i, dtype=np.uint8),
                             photometric='minisblack', planarconfig='contig')
            mapping[cell] = f'/publisher/server/images/{name}'
        mapping['unmatched_release_cell'] = '/publisher/missing.tif'
        (sample / INPUT_DIR / 'cell_image_paths.json').write_text(json.dumps(mapping))
        return sample

    def test_all_datasets_repair_stale_paths_and_keep_gene_order(self):
        for dataset in SAMPLES:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                sample = self.make_sample(root, dataset)
                prepared = prepare(dataset, sample, root / 'prepared' / dataset)
                m = check(prepared, dataset, image_checks=-1)
                self.assertEqual(m['retained_cells'], 12)
                self.assertEqual(m['unavailable_cells'], 1)
                self.assertEqual(m['overlap_audit'], 0)
                paths = json.loads((prepared / 'cell_image_paths.json').read_text())
                self.assertTrue(all(not Path(p).is_absolute() for p in paths.values()))
                self.assertEqual((prepared / 'gene_symbols.txt').read_text().splitlines(), [f'g{i}' for i in range(6)])

    def test_relative_paths_survive_moving_complete_data_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'before'
            sample = self.make_sample(root, 'C1')
            prepare('C1', sample, root / 'prepared/C1')
            moved = root.with_name('after')
            shutil.move(root, moved)
            self.assertEqual(check(moved / 'prepared/C1', 'C1')['retained_cells'], 12)

    def test_subset_preserves_actual_expression_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = self.make_sample(root, 'P1')
            prepared = prepare('P1', sample, root / 'prepared/P1', max_cells=10, max_genes=3)
            original = ad.read_h5ad(sample / 'adata.h5ad')
            subset = ad.read_h5ad(prepared / 'expression.h5ad')
            expected = original[subset.obs_names, subset.var_names]
            np.testing.assert_array_equal(subset.X, expected.X)
            np.testing.assert_array_equal(subset.obsm['spatial'], expected.obsm['spatial'])
            self.assertEqual(check(prepared, 'P1')['gene_count'], 3)

    def test_wrong_dataset_and_changed_index_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = self.make_sample(root, 'C2')
            prepared = prepare('C2', sample, root / 'prepared/C2')
            with self.assertRaises(ValueError):
                check(prepared, 'C1')
            (prepared / 'cell_image_paths.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                check(prepared, 'C2')

    def test_extractor_supports_public_wrappers_and_skips_patch_images(self):
        for dataset in SAMPLES:
            for wrapper in ['', 'processed_data/']:
                with self.subTest(dataset=dataset, wrapper=wrapper), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    sample = self.make_sample(root, dataset)
                    patches = sample / INPUT_DIR / 'patch_images'
                    patches.mkdir()
                    (patches / 'unused.txt').write_text('not used for single-cell training')
                    archive = root / 'sample.tar.gz'
                    with tarfile.open(archive, 'w:gz') as handle:
                        handle.add(sample, arcname=wrapper + sample.name)
                    extracted = extract_archive(archive, root / 'extracted', dataset)
                    self.assertTrue((extracted / 'adata.h5ad').is_file())
                    self.assertFalse((extracted / INPUT_DIR / 'patch_images').exists())
                    self.assertEqual(len(list((extracted / INPUT_DIR / 'cell_images').glob('*.tif'))), 12)

    def test_extractor_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                info = tarfile.TarInfo('../escape')
                info.size = 1
                handle.addfile(info, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                extract_archive(archive, root / 'out', 'C1')
            self.assertFalse((root / 'escape').exists())

    def test_launcher_resolves_each_dataset_and_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dataset in SAMPLES:
                sample = self.make_sample(root, dataset)
                prepared = prepare(dataset, sample, root / 'prepared' / dataset)
                manifest = json.loads((prepared / 'dataset.json').read_text())
                for action in ['train', 'validate']:
                    args = argparse.Namespace(dataset=dataset, prepared_dir=prepared, data_root=root,
                                              output_dir=root / 'out' / dataset, checkpoint=None,
                                              action=action, smoke=False)
                    cmd = build_command(args, manifest, ['--batch_size', '2'])
                    self.assertIn(f'geneflowv2.{action}', cmd)
                    self.assertEqual(Path(cmd[cmd.index('--adata') + 1]), sample / 'adata.h5ad')
                    self.assertEqual(cmd[-2:], ['--batch_size', '2'])


if __name__ == '__main__':
    unittest.main()
