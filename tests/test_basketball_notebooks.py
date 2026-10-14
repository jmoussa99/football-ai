"""Checks for the basketball Colab notebooks' setup and import cells."""

import ast
import contextlib
import importlib.metadata
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
except ImportError:  # pragma: no cover - both ship with Jupyter
    nbformat = TransformerManager = None


ROOT = Path(__file__).resolve().parents[1]
THREE_SECOND_PATH = ROOT / "basketball_ai_automatic_detection_of_3_second_violations.ipynb"
JUMPSHOT_PATH = ROOT / "Copy_of_basketball_ai_make_or_miss_jumpshot_detection.ipynb"
NOTEBOOK_PATHS = (THREE_SECOND_PATH, JUMPSHOT_PATH)
NOTEBOOKS = {path.name: json.loads(path.read_text()) for path in NOTEBOOK_PATHS}
THREE_SECOND = NOTEBOOKS[THREE_SECOND_PATH.name]
JUMPSHOT = NOTEBOOKS[JUMPSHOT_PATH.name]


def source(notebook, cell_id):
    cell = next(cell for cell in notebook["cells"] if cell["metadata"].get("id") == cell_id)
    return "".join(cell["source"])


def code_sources(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def all_code():
    return "\n".join(code for notebook in NOTEBOOKS.values() for code in code_sources(notebook))


def shell_commands(code):
    return [line.strip() for line in code.splitlines() if line.strip().startswith("!")]


def as_python(code):
    """Turn `!shell` and `%magic` lines into plain Python, as IPython does."""
    return TransformerManager().transform_cell(code)


@unittest.skipIf(TransformerManager is None, "needs IPython and nbformat")
class NotebookTests(unittest.TestCase):
    def test_notebooks_are_valid_and_every_code_cell_compiles(self):
        for path in NOTEBOOK_PATHS:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertEqual(cell.outputs, [])
                    compile(as_python(cell.source), cell.metadata["id"], "exec")

    def test_installs_keep_the_versions_inference_gpu_requires(self):
        code = "\n".join(shell_commands(all_code())).lower()
        self.assertIn("inference-gpu", code)
        self.assertNotIn("pillow==", code)
        self.assertNotIn("roboflow/supervision.git", code)
        self.assertNotIn("inference[yolo-world]", code)

    def test_restart_cell_restarts_only_when_preloaded_packages_changed(self):
        for notebook, cell_id in ((THREE_SECOND, "RestartCell3s"), (JUMPSHOT, "RestartCellJs")):
            for loaded, installed, restarts in (("11.3.0", "12.3.0", True), ("12.3.0", "12.3.0", False)):
                with self.subTest(cell=cell_id, loaded=loaded):
                    shutdowns = []
                    kernel = types.SimpleNamespace(do_shutdown=shutdowns.append)
                    ipython = types.ModuleType("IPython")
                    ipython.Application = types.SimpleNamespace(
                        instance=lambda: types.SimpleNamespace(kernel=kernel))
                    pil = types.ModuleType("PIL")
                    pil.__version__ = loaded
                    versions = {"pillow": installed, "numpy": np.__version__}
                    with patch.dict(sys.modules, {"PIL": pil, "IPython": ipython}), \
                            patch.object(importlib.metadata, "version", versions.__getitem__), \
                            contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(source(notebook, cell_id), cell_id, "exec"), {})
                    self.assertEqual(shutdowns, [True] if restarts else [])

    def test_restart_cell_runs_right_after_the_installs(self):
        for notebook, install_id, restart_id in (
            (THREE_SECOND, "U4zROHmjgMgo", "RestartCell3s"),
            (JUMPSHOT, "u2Iyla4lfedz", "RestartCellJs"),
        ):
            ids = [cell["metadata"].get("id") for cell in notebook["cells"]
                   if cell["cell_type"] == "code"]
            self.assertEqual(ids.index(restart_id), ids.index(install_id) + 1)

    def test_sam2_import_uses_the_cloned_fork_over_the_installed_package(self):
        with tempfile.TemporaryDirectory() as home:
            package = Path(home) / "segment-anything-2-real-time" / "sam2"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "build_sam.py").write_text(
                "def build_sam2_camera_predictor(config, checkpoint):\n"
                "    return ('fork', config, checkpoint)\n"
            )
            installed = types.ModuleType("sam2")
            installed.__path__ = []
            installed_build = types.ModuleType("sam2.build_sam")
            original_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {"sam2": installed, "sam2.build_sam": installed_build}):
                    namespace = {"HOME": home}
                    exec(compile(source(THREE_SECOND, "oxLaOgcIrw-1"), "sam2_import", "exec"), namespace)
            finally:
                sys.path[:] = original_path
        self.assertEqual(namespace["predictor"], (
            "fork",
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            f"{home}/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt",
        ))

    def test_sam2_clone_can_be_rerun(self):
        code = source(THREE_SECOND, "BDCmntH2UtHE")
        self.assertIn("[ -d $HOME/segment-anything-2-real-time ] || git clone", code)
        self.assertFalse([line for line in shell_commands(code) if "pip install -e" in line])

    def test_source_video_paths_are_path_objects(self):
        for notebook in NOTEBOOKS.values():
            for code in code_sources(notebook):
                for node in ast.walk(ast.parse(as_python(code))):
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "SOURCE_VIDEO_PATH"
                        for target in node.targets
                    ):
                        value = node.value.left if isinstance(node.value, ast.BinOp) else node.value
                        self.assertIsInstance(value, ast.Call)
                        self.assertEqual(value.func.id, "Path")

    def test_sam2_masks_keep_the_object_axis(self):
        masks = np.zeros((1, 1, 4, 6), dtype=bool)
        self.assertEqual(np.squeeze(masks, axis=1).shape, (1, 4, 6))
        squeezes = re.findall(r"np\.squeeze\(masks[^)]*\)", all_code())
        self.assertTrue(squeezes)
        self.assertTrue(all("axis=1" in call for call in squeezes), squeezes)

    def test_mask_cleanup_compares_unfiltered_and_filtered_masks(self):
        self.assertNotIn("masks_after = masks =", source(THREE_SECOND, "wYk3Xz6v9j8Q"))

    def test_ffmpeg_paths_are_quoted(self):
        lines = [line for line in all_code().splitlines() if "ffmpeg" in line and " -i " in line]
        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, r'-i "\{[^}]+\}" .* "\{[^}]+\}"$')

    def test_key_points_use_non_deprecated_confidence(self):
        self.assertIsNone(re.search(r"key_points(\[[^\]]*\])?\.confidence\b", all_code()))

    def test_jumpshot_downloads_every_file_it_reads(self):
        downloads = source(JUMPSHOT, "polkWGq4gFzX")
        self.assertNotIn("--folder", downloads)
        names = ast.literal_eval(
            ast.parse(source(JUMPSHOT, "ZwHeNnglhoMV")).body[1].value)
        source_video = ast.parse(source(JUMPSHOT, "sqjj2VVYRpm9")).body[1].value.right.value
        for relative_path in [f"boston-celtics-new-york-knicks-game-1/{name}" for name in names] + [
            source_video, "fonts/Staatliches-Regular.ttf"
        ]:
            self.assertIn(f"-O $HOME/{relative_path}\n", downloads + "\n")


if __name__ == "__main__":
    unittest.main()
