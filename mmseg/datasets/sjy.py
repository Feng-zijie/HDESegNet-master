from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SJYDataset(CustomDataset):
    CLASSES = ('background', 'lake','river')
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 127, 191]]

    def __init__(self, **kwargs):
        super(SJYDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)