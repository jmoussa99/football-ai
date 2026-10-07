"""Checks for the Colab notebook's standalone analysis helpers."""

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np


NOTEBOOK = json.loads(
    (Path(__file__).resolve().parents[1] / "Copy_of_NU_Soccer_Analysis.ipynb").read_text()
)


def source(index):
    return "".join(NOTEBOOK["cells"][index]["source"])


def function_from_cell(index, name, namespace):
    module = ast.parse(source(index))
    node = next(node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"cell_{index}", "exec"), namespace)
    return namespace[name]


class NotebookTests(unittest.TestCase):
    def test_all_cells_compile_and_only_one_video_path_is_assigned(self):
        assignments = []
        for index, cell in enumerate(NOTEBOOK["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = source(index)
            compile(code, f"cell_{index}", "exec")
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SOURCE_VIDEO_PATH"
                    for target in node.targets
                ):
                    assignments.append(index)
        self.assertEqual(assignments, [16])

    def test_outlier_filter_keeps_frame_alignment_and_uses_elapsed_seconds(self):
        namespace = {"np": np, "fps": 30, "speeds": [], "List": List}
        clean = function_from_cell(60, "replace_outliers_based_on_distance", namespace)
        positions = [np.array([0, 0]), np.empty(0), np.array([300, 0]),
                     np.array([10000, 0]), np.array([600, 0])]
        result = clean(positions, 5000)
        self.assertEqual([len(position) for position in result], [2, 0, 2, 0, 2])
        self.assertEqual(len(namespace["speeds"]), len(positions))
        self.assertAlmostEqual(namespace["speeds"][2], 45.0)
        self.assertAlmostEqual(namespace["speeds"][4], 45.0)

    def test_speed_estimator_keeps_pixel_speed_without_calibration(self):
        sv_stub = SimpleNamespace(Position=SimpleNamespace(BOTTOM_CENTER=object()))
        namespace = {"np": np, "sv": sv_stub}
        estimator_type = function_from_cell(63, "PlayerSpeedEstimator", namespace)

        class Detections:
            tracker_id = np.array([7])

            def __init__(self, x):
                self.x = x

            def get_anchors_coordinates(self, _):
                return np.array([[self.x, 0]], dtype=float)

        class Transformer:
            def transform_points(self, points):
                return points / 100.0

        estimator = estimator_type(fps=30, sample_stride=30)
        estimator.update(0, Detections(0), Transformer())
        pixels, meters = estimator.update(30, Detections(30))
        self.assertAlmostEqual(pixels[7], 30.0)
        self.assertIsNone(meters[7])
        _, meters = estimator.update(60, Detections(60), Transformer())
        self.assertIsNone(meters[7])
        _, meters = estimator.update(90, Detections(90), Transformer())
        self.assertAlmostEqual(meters[7], 0.3)

    def test_pitch_transformer_uses_matching_confident_points_and_scale(self):
        class Transformer:
            def __init__(self, source, target):
                self.source = source
                self.target = target
                self.m = np.eye(3)

        namespace = {
            "np": np,
            "cv2": SimpleNamespace(error=ValueError),
            "ViewTransformer": Transformer,
            "CONFIG": SimpleNamespace(vertices=[(0, 0), (100, 0), (100, 100), (0, 100)]),
        }
        get_transformer = function_from_cell(39, "get_pitch_transformer", namespace)
        keypoints = SimpleNamespace(
            xy=np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=float),
            confidence=np.ones((1, 4)),
        )
        transformer = get_transformer(keypoints, target_scale=0.01)
        np.testing.assert_allclose(transformer.target, [[0, 0], [1, 0], [1, 1], [0, 1]])
        keypoints.confidence[:] = 0
        self.assertIsNone(get_transformer(keypoints))


if __name__ == "__main__":
    unittest.main()
