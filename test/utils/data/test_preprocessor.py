from unittest import TestCase

class TestPreprocessor(TestCase):
    def test_getitem(self):
        from reid.utils.data.preprocessor import Preprocessor
        from reid.datasets.viper import VIPeR
        import torchvision.transforms as t

        root, split_id, num_val = '/tmp/open-reid/viper', 0, 100
        dataset = VIPeR(root, split_id = split_id, num_val = num_val, download = True)

        preproc = Preprocessor(dataset.train, root=dataset.images_dir,
                               transform=t.Compose([
                                   t.Scale(256),
                                   t.CenterCrop(224),
                                   t.ToTensor(),
                                   t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]))
        
        self.assertEquals(len(preproc), len(dataset.train))
        img, pid, camid = preproc[0]
        self.assertEquals(img.size(), (3, 224, 224))
