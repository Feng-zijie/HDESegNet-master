from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SurfaceWaterDataset(CustomDataset):
    CLASSES = ('background', 'lake')
    PALETTE = [[0, 0, 0], [50, 50, 50]]

    def __init__(self, **kwargs):
        super(SurfaceWaterDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
