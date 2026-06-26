import unittest

import numpy as np

from slide2vec.wsi.wsi import WholeSlideImage


class NormalizeBinaryMaskTest(unittest.TestCase):
    def test_rgb_mask_is_normalized_to_single_channel_uint8(self):
        mask = np.array(
            [
                [[1, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 1]],
            ],
            dtype=np.uint8,
        )

        normalized = WholeSlideImage._normalize_binary_mask(
            mask,
            tissue_pixel_value=1,
            foreground_value=255,
        )

        expected = np.array(
            [
                [255, 0],
                [0, 255],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(normalized, expected)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.ndim, 2)


if __name__ == "__main__":
    unittest.main()
