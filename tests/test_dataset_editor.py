import tempfile
import unittest
from pathlib import Path

import dataset_editor as de


class DatasetEditorTests(unittest.TestCase):
    def test_copy_dataset_rejects_destination_inside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            (src / "meta").mkdir(parents=True)
            (src / "meta" / "info.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ValueError):
                de.copy_dataset(src, src / "copy")


if __name__ == "__main__":
    unittest.main()
