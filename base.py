import typing

import numpy as np


class SeamCarving(typing.Protocol):

    def __call__(
        self,
        image: typing.Union[str, np.ndarray],
        vertical_seams: int = 0,
        horizontal_seams: int = 0,
        retain_mask: typing.Union[str, np.ndarray] = None,
        remove_mask: typing.Union[str, np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError()
