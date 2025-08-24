from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SWDataset(CustomDataset):
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [0, 0, 255555]]

    def __init__(self, **kwargs):
        super(SWDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)