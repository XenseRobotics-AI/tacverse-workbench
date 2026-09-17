import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import viewer_service as vsvc


class _Response:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


class ViewerDoctorTests(unittest.TestCase):
    def test_doctor_parses_progress_and_result_stream(self):
        events = [
            {"type": "progress", "progress": {"overall_percent": 42,
                                                  "message": "Running"}},
            {"type": "result", "result": {"ok": True, "report": {}}},
        ]
        progress = []
        with patch("viewer_service.urllib.request.urlopen",
                   return_value=_Response(
                       [json.dumps(event) + "\n" for event in events])) as open_url:
            result, error = vsvc.ViewerService(port=3001).doctor(
                "TacVerse/example", max_episodes=25,
                on_progress=progress.append)
        self.assertIsNone(error)
        self.assertEqual(result["ok"], True)
        self.assertEqual(progress[0]["overall_percent"], 42)
        request = open_url.call_args.args[0]
        self.assertEqual(request.method, "POST")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["maxEpisodes"], 25)

    def test_doctor_reports_stream_error(self):
        with patch("viewer_service.urllib.request.urlopen",
                   return_value=_Response([
                       json.dumps({"type": "error", "error": "failed"}) + "\n"
                   ])):
            result, error = vsvc.ViewerService(port=3001).doctor("TacVerse/example")
        self.assertIsNone(result)
        self.assertEqual(error, "failed")

    def test_doctor_retries_incomplete_chunked_read(self):
        events = [{"type": "result", "result": {"ok": True}}]
        error = RuntimeError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)")
        with patch("viewer_service.urllib.request.urlopen",
                   side_effect=[error, _Response(
                       [json.dumps(event) + "\n" for event in events])]) as open_url, \
                patch("viewer_service.time.sleep"):
            result, err = vsvc.ViewerService(port=3001).doctor("TacVerse/example")

        self.assertIsNone(err)
        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, open_url.call_count)


class ViewerServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "pulls"
        self.viewer_dir = Path(self.tmp.name) / "viewer"
        self.root.mkdir()
        self.viewer_dir.mkdir()
        (self.viewer_dir / "package.json").write_text("{}", encoding="utf-8")
        (self.viewer_dir / "node_modules").mkdir()
        self.service = vsvc.ViewerService(self.viewer_dir, port=39001)
        self.addCleanup(self.service._state_path.unlink, missing_ok=True)

    @patch.object(vsvc, "_port_in_use", return_value=True)
    def test_start_reuses_viewer_with_matching_root(self, _port):
        info = {"root": str(self.root), "datasets": []}
        with patch.object(self.service, "service_info", return_value=info):
            ok, message = self.service.start(self.root)

        self.assertTrue(ok)
        self.assertIn("现有 Viewer", message)
        self.assertTrue(self.service._attached)

    @patch.object(vsvc, "_port_in_use", return_value=True)
    def test_start_rejects_non_viewer_port_conflict(self, _port):
        with patch.object(self.service, "service_info", return_value=None):
            ok, message = self.service.start(self.root)

        self.assertFalse(ok)
        self.assertIn("其他程序占用", message)
        self.assertIsNone(self.service.proc)

    @patch.object(vsvc, "_port_in_use", return_value=True)
    def test_status_reports_wrong_root_as_conflict(self, _port):
        info = {"root": str(self.root / "old"), "datasets": []}
        with patch.object(self.service, "service_info", return_value=info):
            self.service.root = str(self.root)
            status = self.service.status()

        self.assertEqual("conflict", status["state"])
        self.assertFalse(status["ready"])
        self.assertEqual(info["root"], status["actual_root"])

    @patch.object(vsvc, "_port_in_use", side_effect=[True, False])
    def test_start_replaces_viewer_with_stale_root(self, _port):
        info = {"root": str(self.root / "old"), "datasets": []}
        process = MagicMock()
        process.pid = 12345
        with patch.object(self.service, "service_info", return_value=info), \
                patch.object(self.service, "_stop_processes") as stop_processes, \
                patch.object(vsvc, "find_bun", return_value="bun"), \
                patch.object(vsvc.subprocess, "Popen", return_value=process):
            ok, message = self.service.start(self.root)

        self.assertTrue(ok)
        self.assertEqual("启动中…（bun）", message)
        stop_processes.assert_called_once_with(info["root"])
        self.assertIs(process, self.service.proc)

    def test_js_runner_falls_back_to_npm_with_modern_node(self):
        with patch.object(vsvc, "find_bun", return_value=None), \
                patch.object(vsvc.shutil, "which", return_value="npm"), \
                patch.object(vsvc, "node_version", return_value=(20, "20.0.0")):
            cmd, runner = vsvc.find_js_runner()

        self.assertEqual(["npm", "run", "dev"], cmd)
        self.assertEqual("npm", runner)

    def test_install_hint_points_to_runtime_copy(self):
        hint = vsvc.install_hint(self.viewer_dir / "missing")

        self.assertIn("scripts/install_viewer.py", hint)

    def test_prepare_viewer_runtime_copies_source_without_touching_it(self):
        source = Path(self.tmp.name) / "source"
        runtime = Path(self.tmp.name) / "runtime"
        source.mkdir()
        (source / "package.json").write_text("{}", encoding="utf-8")
        (source / ".git").write_text("ignored", encoding="utf-8")

        copied = vsvc.prepare_viewer_runtime(source, runtime)

        self.assertEqual(runtime, copied)
        self.assertTrue((runtime / "package.json").is_file())
        self.assertFalse((runtime / ".git").exists())
        self.assertTrue((source / "package.json").is_file())

    def test_prepare_viewer_runtime_rejects_paths_inside_source(self):
        source = Path(self.tmp.name) / "source"
        source.mkdir()
        (source / "package.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(ValueError):
            vsvc.prepare_viewer_runtime(source, source / "runtime")

    @patch.object(vsvc, "_port_in_use", return_value=True)
    def test_stop_does_not_kill_viewer_with_wrong_root(self, _port):
        self.service.root = str(self.root)
        self.service.proc = MagicMock()
        info = {"root": str(self.root / "other"), "datasets": []}
        with patch.object(self.service, "service_info", return_value=info), \
                patch.object(self.service, "_stop_processes") as stop_processes:
            ok, message = self.service.stop()

        self.assertFalse(ok)
        self.assertIn("未关闭外部服务", message)
        stop_processes.assert_not_called()

    @patch.object(vsvc, "_port_in_use", return_value=True)
    def test_stop_does_not_kill_unrecognized_port_owner(self, _port):
        self.service.root = str(self.root)
        self.service.proc = MagicMock()
        with patch.object(self.service, "service_info", return_value=None), \
                patch.object(self.service, "_stop_processes") as stop_processes:
            ok, message = self.service.stop()

        self.assertFalse(ok)
        self.assertIn("不是可识别的 Viewer", message)
        stop_processes.assert_not_called()

    def test_service_info_requires_viewer_response_shape(self):
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({"hello": "world"}).encode()
        response.__enter__.return_value = response
        with patch.object(vsvc.urllib.request, "urlopen", return_value=response):
            self.assertIsNone(self.service.service_info(max_age=0))

    def test_report_retries_incomplete_chunked_read(self):
        response = MagicMock()
        response.read.return_value = json.dumps({"ok": True}).encode()
        response.__enter__.return_value = response
        error = RuntimeError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)")

        with patch.object(
                vsvc.urllib.request, "urlopen",
                side_effect=[error, response]) as open_url, \
                patch.object(vsvc.time, "sleep"):
            report, err = self.service.report("TacVerse/example")

        self.assertIsNone(err)
        self.assertEqual({"ok": True}, report)
        self.assertEqual(2, open_url.call_count)


if __name__ == "__main__":
    unittest.main()
