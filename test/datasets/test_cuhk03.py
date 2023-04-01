from unittest import TestCase

class TestCUHK03(TestCase):
    def test_init(self):
        from reid.utils.serialization import read_json
        from reid.datasets.cuhk03 import CUHK03
        import os.path as osp

        root, split_id, num_val = '/tmp/open-reid/cuhk03', 0, 100
        dataset = CUHK03(root, split_id = split_id, num_val = num_val, download = True)

        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 1467)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 20)

        self.assertDictEqual(meta, dataset.meta)
        self.assertDictEqual(splits[split_id], dataset.split)
