import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from slide2vec.tiling import (
    get_slides_with_tiles,
    get_wsi_name,
    mark_tiling_success_from_coordinates,
)


class TilingCoordinateNameTest(unittest.TestCase):
    def test_space_containing_slide_stem_matches_sanitized_coordinate_file(self):
        slide_path = Path("/slides/T19-019550 I 001 HE 1-2-3 I 6516476.mrxs")

        with TemporaryDirectory() as tmpdir:
            coordinates_dir = Path(tmpdir)
            coordinates_file = coordinates_dir / "T19-019550_I_001_HE_1-2-3_I_6516476.npy"
            coordinates_file.touch()

            self.assertEqual(
                get_wsi_name(slide_path),
                "T19-019550_I_001_HE_1-2-3_I_6516476",
            )
            self.assertEqual(
                get_slides_with_tiles([slide_path], coordinates_dir),
                [str(slide_path)],
            )

    def test_read_coordinates_marks_unsanitized_process_list_name_success(self):
        process_df = pd.DataFrame(
            {
                "wsi_name": ["T19-019550 I 001 HE 1-2-3 I 6516476"],
                "tiling_status": ["tbp"],
            }
        )

        with TemporaryDirectory() as tmpdir:
            coordinates_dir = Path(tmpdir)
            coordinates_file = coordinates_dir / "T19-019550_I_001_HE_1-2-3_I_6516476.npy"
            coordinates_file.touch()

            updated = mark_tiling_success_from_coordinates(process_df, coordinates_dir)

        self.assertEqual(updated.loc[0, "tiling_status"], "success")


if __name__ == "__main__":
    unittest.main()
