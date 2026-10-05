"""Regression checks for the notebook helpers; uses its pinned sports/supervision APIs."""

import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "football_ai.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text())


def source(index):
    return "".join(NOTEBOOK["cells"][index]["source"])


def helpers():
    namespace = {"sv": sv, "np": np}
    crop_module = ast.parse(source(20))
    crop_module.body = [node for node in crop_module.body if isinstance(node, ast.FunctionDef)]
    exec(compile(crop_module, "crop_helpers", "exec"), namespace)
    for index in (22, 23, 24):
        exec(compile(source(index), f"cell_{index}", "exec"), namespace)
    return namespace


def detections(points, tracker_ids=None, classes=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    boxes = np.array([[x - 1, y - 2, x + 1, y] for x, y in points]).reshape(-1, 4)
    return sv.Detections(
        xyxy=boxes,
        confidence=np.ones(len(points)),
        class_id=np.asarray(classes if classes is not None else [0] * len(points), dtype=int),
        tracker_id=np.asarray(tracker_ids if tracker_ids is not None else range(len(points)), dtype=int),
    )


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.namespace = helpers()
        self.Estimator = self.namespace["PlayerSpeedEstimator"]
        self.transformer = self.namespace["ViewTransformer"](
            source=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            target=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        )

    def test_every_code_cell_compiles(self):
        for index, entry in enumerate(NOTEBOOK["cells"]):
            if entry["cell_type"] == "code":
                compile(source(index), f"cell_{index}", "exec")

    def test_device_selection_checks_pytorch_in_addition_to_onnx(self):
        for cuda_available, available_providers, expected_device, expected_providers in (
            (False, ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "cpu", "[CPUExecutionProvider]"),
            (True, ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "cuda", "[CUDAExecutionProvider,CPUExecutionProvider]"),
            (True, ["CPUExecutionProvider"], "cuda", "[CPUExecutionProvider]"),
        ):
            with self.subTest(cuda_available=cuda_available, providers=available_providers):
                torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))
                ort_stub = SimpleNamespace(get_available_providers=lambda: available_providers)
                namespace = {}
                with patch.dict("sys.modules", {"torch": torch_stub, "onnxruntime": ort_stub}), \
                        patch.dict(os.environ, {}), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(source(14), "device_setup", "exec"), namespace)
                    self.assertEqual(namespace["DEVICE"], expected_device)
                    self.assertEqual(os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"], expected_providers)

    def test_speed_uses_sample_interval_and_persistent_metric_positions(self):
        estimator = self.Estimator(30, sample_stride=30)
        estimator.update(0, detections([[10, 20]], [7]), self.transformer)
        estimator.update(15, detections([[25, 20]], [7]))
        pixels, meters = estimator.update(30, detections([[40, 20]], [7]), self.transformer)
        self.assertAlmostEqual(pixels[7], 30)
        self.assertAlmostEqual(meters[7], 0.3, places=6)
        pixels, meters = estimator.update(60, detections([[70, 20]], [7]), self.transformer)
        self.assertAlmostEqual(pixels[7], 30)
        self.assertAlmostEqual(meters[7], 0.3, places=6)

    def test_each_frame_can_use_its_own_camera_calibration(self):
        estimator = self.Estimator(30, 30)
        estimator.update(0, detections([[10, 20]], [7]), self.transformer)
        moved_camera = self.namespace["ViewTransformer"](
            source=np.array([[50, 0], [150, 0], [150, 100], [50, 100]]),
            target=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        )
        pixels, meters = estimator.update(30, detections([[60, 20]], [7]), moved_camera)
        self.assertAlmostEqual(pixels[7], 50)
        self.assertAlmostEqual(meters[7], 0, places=6)

    def test_missing_calibration_keeps_pixel_speed_and_resets_metric_baseline(self):
        estimator = self.Estimator(30, 30)
        estimator.update(0, detections([[10, 20]], [7]), self.transformer)
        pixels, meters = estimator.update(30, detections([[40, 20]], [7]))
        self.assertAlmostEqual(pixels[7], 30)
        self.assertIsNone(meters[7])
        _, meters = estimator.update(60, detections([[70, 20]], [7]), self.transformer)
        self.assertIsNone(meters[7])
        _, meters = estimator.update(90, detections([[100, 20]], [7]), self.transformer)
        self.assertAlmostEqual(meters[7], 0.3, places=6)

    def test_lost_track_does_not_reuse_old_position_or_speed(self):
        estimator = self.Estimator(30, 30)
        estimator.update(0, detections([[10, 20]], [7]), self.transformer)
        pixels, meters = estimator.update(1, sv.Detections.empty())
        self.assertEqual(pixels, {})
        self.assertEqual(meters, {})
        pixels, meters = estimator.update(30, detections([[80, 20]], [7]), self.transformer)
        self.assertEqual(pixels[7], 0)
        self.assertIsNone(meters[7])

    def test_pitch_coordinates_are_converted_from_centimeters_to_meters(self):
        points = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=float)
        keypoints = sv.KeyPoints(xy=points, confidence=np.ones((1, 4)))
        config = SimpleNamespace(vertices=points[0].tolist())
        transformer = self.namespace["get_pitch_transformer"](keypoints, config)
        np.testing.assert_allclose(transformer.transform_points(np.array([[50, 50]])), [[0.5, 0.5]])

    def test_missing_low_confidence_and_collinear_keypoints(self):
        transform = self.namespace["get_pitch_transformer"]
        self.assertIsNone(transform(None))
        self.assertIsNone(transform(sv.KeyPoints.empty()))
        points = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=float)
        config = SimpleNamespace(vertices=points[0].tolist())
        self.assertIsNone(transform(sv.KeyPoints(xy=points, confidence=np.zeros((1, 4))), config))
        line = np.array([[[0, 0], [1, 1], [2, 2], [3, 3]]], dtype=float)
        self.assertIsNone(transform(sv.KeyPoints(xy=line, confidence=np.ones((1, 4))), config))

    def test_wrong_pitch_model_is_reported(self):
        points = sv.KeyPoints(xy=np.ones((1, 4, 2)), confidence=np.ones((1, 4)))
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.namespace["get_pitch_transformer"](points)

    def test_goalkeepers_handle_missing_teams_and_empty_detections(self):
        resolve = self.namespace["resolve_goalkeepers_team_id"]
        keepers = detections([[2, 10], [98, 10]])
        players = detections([[0, 10], [100, 10]], classes=[0, 1])
        np.testing.assert_array_equal(resolve(players, keepers), [0, 1])
        np.testing.assert_array_equal(resolve(players[:1], keepers), [3, 3])
        np.testing.assert_array_equal(resolve(sv.Detections.empty(), keepers), [3, 3])
        result = resolve(players, sv.Detections.empty())
        self.assertEqual(result.dtype.kind, "i")
        self.assertEqual(result.size, 0)

    def test_invalid_crops_are_skipped_without_losing_alignment(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        boxes = sv.Detections(
            xyxy=np.array([[-2, -2, 3, 3], [20, 20, 30, 30], [5, 5, 5, 7],
                           [np.nan, 1, 3, 3], [6, 6, 9, 9]]),
            class_id=np.zeros(5, dtype=int), tracker_id=np.arange(5),
        )
        kept, crops = self.namespace["crop_players"](frame, boxes)
        np.testing.assert_array_equal(kept.tracker_id, [0, 4])
        self.assertEqual([crop.shape for crop in crops], [(3, 3, 3), (3, 3, 3)])

    def test_invalid_frame_rate_or_stride_is_reported(self):
        for fps, stride in ((0, 30), (np.nan, 30), (30, 0)):
            with self.assertRaises(ValueError):
                self.Estimator(fps, stride)

    def test_connections_handle_identical_player_positions(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        output = self.namespace["draw_all_connections_on_frame"](
            frame, detections([[10, 10], [10, 10]]), sv.Color.RED)
        self.assertTrue(output.any())
        self.assertFalse(frame.any())

    def run_video_pipeline(self, predictions, frame_count):
        class Model:
            def __init__(self, predictions):
                self.predictions = predictions
                self.calls = 0

            def infer(self, frame, confidence):
                self.calls += 1
                return [{"predictions": self.predictions,
                         "image": {"width": frame.shape[1], "height": frame.shape[0]}}]

        class Classifier:
            calls = 0

            def predict(self, crops):
                if not crops:
                    raise AssertionError("Do not call the classifier with empty crops")
                self.calls += 1
                return np.arange(len(crops)) % 2

        with tempfile.TemporaryDirectory() as directory:
            input_path = str(Path(directory) / "input.mp4")
            output_path = str(Path(directory) / "output.mp4")
            writer = cv2.VideoWriter(input_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(frame_count):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()

            people_model = Model(predictions)
            field_model = Model([])
            classifier = Classifier()
            namespace = self.namespace | {
                "SOURCE_VIDEO_PATH": input_path, "OUTPUT_VIDEO_PATH": output_path,
                "video_info": sv.VideoInfo.from_video_path(input_path),
                "BALL_ID": 0, "GOALKEEPER_ID": 1, "PLAYER_ID": 2, "REFEREE_ID": 3,
                "PLAYER_DETECTION_MODEL": people_model, "FIELD_DETECTION_MODEL": field_model,
                "team_classifier": classifier, "tqdm": lambda items, **kwargs: items,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source(26), "tracking_cell", "exec"), namespace)
            self.assertEqual(people_model.calls, frame_count)
            self.assertEqual(field_model.calls, (frame_count - 1) // 30 + 1)
            self.assertEqual(sv.VideoInfo.from_video_path(output_path).total_frames, frame_count)
            return namespace, classifier

    def test_video_loop_writes_frames_with_no_detections(self):
        namespace, classifier = self.run_video_pipeline([], 3)
        self.assertEqual(classifier.calls, 0)
        self.assertEqual(namespace["frames_without_calibration"], 1)

    def test_video_loop_handles_players_goalkeeper_referee_and_ball(self):
        predictions = []
        for x, y, class_id, name in ((4, 8, 2, "player"), (24, 8, 2, "player"),
                                    (2, 18, 1, "goalkeeper"), (16, 18, 3, "referee"),
                                    (16, 2, 0, "ball")):
            predictions.append({"x": x, "y": y, "width": 3, "height": 4,
                                "confidence": 0.95, "class_id": class_id, "class": name})
        namespace, classifier = self.run_video_pipeline(predictions, 31)
        self.assertGreater(classifier.calls, 0)
        self.assertEqual(len(namespace["player_speeds_pps"]), 3)
        self.assertTrue(all(value is None for value in namespace["player_speeds_mps"].values()))


if __name__ == "__main__":
    unittest.main()
