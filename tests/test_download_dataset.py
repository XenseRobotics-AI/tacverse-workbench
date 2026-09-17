import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import call, patch

import download_dataset as dd


class DatasetLogTests(unittest.TestCase):
    def test_present_arrays_are_compact_and_round_trip(self):
        log = {
            "dataset_index": [f"org/dataset-{i}" for i in range(40)],
            "daily_totals": [{
                "pulled_at": "2026-07-24T12:00:00",
                "date": "260724",
                "org": "TacVerse",
                "total_datasets": 40,
                "total_episodes": 100,
                "total_frames": 1000,
                "total_hours": 1.0,
                "present": list(range(40)),
            }],
            "datasets": {},
        }

        text = dd._dataset_log_text(log)

        self.assertEqual(json.loads(text), log)
        present_lines = []
        inside = False
        for line in text.splitlines():
            if line.strip() == '"present": [':
                inside = True
                continue
            if inside and line.strip() == "]":
                break
            if inside:
                present_lines.append(line)
        self.assertEqual([16, 16, 8], [
            len(line.strip().rstrip(",").split(", "))
            for line in present_lines
        ])

    def test_dataset_changes_and_metadata_use_compact_lines(self):
        change = {
            "date": "260724",
            "pulled_at": "2026-07-24T14:04:19",
            "total_episodes": 10,
            "total_frames": 6017,
            "duration_hours": 0.056,
            "d_episodes": 10,
            "d_frames": 6017,
            "d_hours": 0.056,
        }
        log = {
            "dataset_index": ["TacVerse/taccap-g1-cucumber-0702"],
            "daily_totals": [],
            "datasets": {
                "TacVerse/taccap-g1-cucumber-0702": {
                    "changes": [change],
                    "fps": 30,
                    "robot_type": "bi_taccap_gripper",
                    "total_tasks": 1,
                    "uploader": "WBH333",
                    "last_modified": "2026-07-02T09:16:55+00:00",
                },
            },
        }

        text = dd._dataset_log_text(log)

        self.assertEqual(log, json.loads(text))
        self.assertIn(
            '        {"date": "260724", "pulled_at": '
            '"2026-07-24T14:04:19", "total_episodes": 10,', text)
        self.assertIn(
            '      "fps": 30, "robot_type": "bi_taccap_gripper", '
            '"total_tasks": 1, "uploader": "WBH333", '
            '"last_modified": "2026-07-02T09:16:55+00:00"}', text)


    def test_manual_snapshot_upserts_and_real_pull_replaces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"
            today = dt.date(2026, 7, 24)
            dd.upsert_manual_totals(
                "260723", "TacVerse", 156, 7898, 18289577, 169.311,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "TacVerse", 157, 8000, 19000000, 170.0,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "Xense", 20, 500, 1000000, 9.0,
                path=path, today=today,
            )
            log = dd.load_dataset_log(path)
            manual = [r for r in log["daily_totals"]
                      if r.get("source") == "manual"]
            self.assertEqual(2, len(manual))
            tacverse = next(r for r in manual if r["org"] == "TacVerse")
            self.assertEqual(8000, tacverse["total_episodes"])
            self.assertEqual([], tacverse["present"])

            report = {
                "pulled_at": "2026-07-23T20:00:00",
                "date": "260723",
                "org": "TacVerse",
                "total_datasets": 1,
                "total_episodes": 10,
                "total_frames": 10800,
                "total_hours": 0.1,
                "datasets": [{
                    "dataset_name": "TacVerse/example",
                    "total_episodes": 10,
                    "total_frames": 10800,
                    "duration_hours": 0.1,
                }],
            }
            dd.append_pull(report, path=path)
            log = dd.load_dataset_log(path)
            remaining_manual = [r for r in log["daily_totals"]
                                if r.get("source") == "manual"]
            self.assertEqual(["Xense"], [r["org"] for r in remaining_manual])
            real = next(r for r in log["daily_totals"]
                        if r.get("source") != "manual")
            self.assertEqual([0], real["present"])

    def test_manual_history_drives_daily_totals_without_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dataset_log.json"
            today = dt.date(2026, 7, 24)
            dd.upsert_manual_totals(
                "260722", "TacVerse", 150, 7030, 17000000, 149.911,
                path=path, today=today,
            )
            dd.upsert_manual_totals(
                "260723", "TacVerse", 156, 7898, 18289577, 169.311,
                path=path, today=today,
            )

            history = dd.load_history(root / "pulls", log_file=path)
            self.assertEqual(["260722", "260723"],
                             [row["date"] for row in history])
            self.assertEqual([], history[-1]["datasets"])
            self.assertEqual("manual", history[-1]["source"])
            series = dd.daily_series(history)
            self.assertEqual(19.4, series[-1]["new_hours"])
            self.assertEqual(7898, series[-1]["total_episodes"])
            self.assertIs(history[-2], dd.find_baseline(history[-1], history))
            self.assertEqual((19.4, 868),
                             dd.aggregate_deltas(history[-1], history))

    def test_future_manual_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dd.upsert_manual_totals(
                    "260725", "TacVerse", 1, 1, 1, 1.0,
                    path=Path(tmp) / "dataset_log.json",
                    today=dt.date(2026, 7, 24),
                )

    def test_only_latest_pull_is_kept_for_each_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"

            def report(pulled_at, date, episodes):
                return {
                    "pulled_at": pulled_at,
                    "date": date,
                    "org": "TacVerse",
                    "total_datasets": 1,
                    "total_episodes": episodes,
                    "total_frames": episodes * 100,
                    "total_hours": episodes / 10,
                    "datasets": [{
                        "dataset_name": "TacVerse/example",
                        "total_episodes": episodes,
                        "total_frames": episodes * 100,
                        "duration_hours": episodes / 10,
                    }],
                }

            dd.append_pull(report("2026-07-23T18:00:00", "260723", 5), path)
            early = report("2026-07-24T14:04:19", "260724", 10)
            latest = report("2026-07-24T15:12:00", "260724", 15)
            dd.append_pull(early, path)
            dd.append_pull(latest, path)

            log = dd.load_dataset_log(path)
            today_rows = [r for r in log["daily_totals"]
                          if r["date"] == "260724" and r["org"] == "TacVerse"]
            self.assertEqual(1, len(today_rows))
            self.assertEqual("2026-07-24T15:12:00", today_rows[0]["pulled_at"])
            changes = log["datasets"]["TacVerse/example"]["changes"]
            today_changes = [c for c in changes if c["date"] == "260724"]
            self.assertEqual(1, len(today_changes))
            self.assertEqual(15, today_changes[0]["total_episodes"])
            self.assertEqual(10, today_changes[0]["d_episodes"])

            # Replaying an older result must not roll the day back.
            dd.append_pull(early, path)
            log = dd.load_dataset_log(path)
            today_row = next(r for r in log["daily_totals"]
                             if r["date"] == "260724")
            self.assertEqual("2026-07-24T15:12:00", today_row["pulled_at"])
            today_change = next(c for c in
                                log["datasets"]["TacVerse/example"]["changes"]
                                if c["date"] == "260724")
            self.assertEqual(15, today_change["total_episodes"])

    def test_compaction_keeps_metadata_and_recomputes_daily_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_log.json"
            log = {
                "dataset_index": ["TacVerse/example", "TacVerse/orphan"],
                "daily_totals": [
                    {"pulled_at": "2026-07-23T18:00:00", "date": "260723",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 5, "total_frames": 500,
                     "total_hours": 0.5, "present": [0]},
                    {"pulled_at": "2026-07-24T14:04:19", "date": "260724",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 10, "total_frames": 1000,
                     "total_hours": 1.0, "present": [0]},
                    {"pulled_at": "2026-07-24T15:12:00", "date": "260724",
                     "org": "TacVerse", "total_datasets": 1,
                     "total_episodes": 15, "total_frames": 1500,
                     "total_hours": 1.5, "present": [0]},
                ],
                "datasets": {
                    "TacVerse/example": {
                        "robot_type": "test_robot",
                        "changes": [
                            {"pulled_at": "2026-07-23T18:00:00", "date": "260723",
                             "total_episodes": 5, "total_frames": 500,
                             "duration_hours": 0.5, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                            {"pulled_at": "2026-07-24T14:04:19", "date": "260724",
                             "total_episodes": 10, "total_frames": 1000,
                             "duration_hours": 1.0, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                            {"pulled_at": "2026-07-24T15:12:00", "date": "260724",
                             "total_episodes": 15, "total_frames": 1500,
                             "duration_hours": 1.5, "d_episodes": 5,
                             "d_frames": 500, "d_hours": 0.5},
                        ],
                    },
                    "TacVerse/orphan": {
                        "robot_type": "keep_me",
                        "changes": [],
                    },
                },
            }
            dd.write_dataset_log(log, path)

            self.assertEqual((3, 2), dd.compact_dataset_log(path))
            compacted = dd.load_dataset_log(path)
            self.assertIn("TacVerse/orphan", compacted["datasets"])
            self.assertEqual("keep_me",
                             compacted["datasets"]["TacVerse/orphan"]["robot_type"])
            changes = compacted["datasets"]["TacVerse/example"]["changes"]
            self.assertEqual(2, len(changes))
            self.assertEqual("2026-07-24T15:12:00", changes[-1]["pulled_at"])
            self.assertEqual(10, changes[-1]["d_episodes"])
            self.assertEqual(1000, changes[-1]["d_frames"])
            self.assertEqual(1.0, changes[-1]["d_hours"])

    def _detailed_report(self, date, pulled_at, episodes, frames, hours,
                         last_modified="2026-07-24T10:00:00+00:00"):
        return {
            "pulled_at": pulled_at,
            "date": date,
            "org": "TacVerse",
            "total_datasets": 1,
            "total_episodes": episodes,
            "total_frames": frames,
            "total_hours": hours,
            "datasets": [{
                "dataset_name": "TacVerse/example",
                "total_episodes": episodes,
                "total_frames": frames,
                "duration_hours": hours,
                "last_modified": last_modified,
                "uploader": "tester",
            }],
        }

    def test_matching_hf_cache_overrides_dataset_log_delta(self):
        previous = self._detailed_report(
            "260723", "2026-07-23T12:00:00", 10, 1000, 1.0)
        current = self._detailed_report(
            "260724", "2026-07-24T12:00:00", 30, 3000, 3.0)
        cache = {"version": 1, "repos": {"TacVerse/example": {
            "last_modified": current["datasets"][0]["last_modified"],
            "changes": [{
                "dataset_name": "TacVerse/example",
                "date": "260724",
                "created_at": "2026-07-24T10:00:00+00:00",
                "hours": 0.5,
                "episodes": 5,
                "frames": 500,
            }],
        }}}
        history = [previous, current]

        delta = dd.hf_last_modified_dataset_deltas(
            current, history, cache)["TacVerse/example"]
        totals = dd.hf_last_modified_totals(current, history, cache)

        self.assertTrue(dd.hf_report_has_matching_change_cache(current, cache))
        self.assertEqual((0.5, 5, 500), (
            delta["d_hours"], delta["d_episodes"], delta["d_frames"]))
        self.assertEqual(("260724", 0.5, 5), (
            totals["date"], totals["hours"], totals["episodes"]))

    def test_stale_or_missing_hf_cache_falls_back_to_dataset_log(self):
        previous = self._detailed_report(
            "260723", "2026-07-23T12:00:00", 10, 1000, 1.0)
        current = self._detailed_report(
            "260724", "2026-07-24T12:00:00", 30, 3000, 3.0)
        stale = {"version": 1, "repos": {"TacVerse/example": {
            "last_modified": "2026-07-23T10:00:00+00:00",
            "changes": [{"date": "260723", "hours": 99, "episodes": 99}],
        }}}

        totals = dd.hf_last_modified_totals(current, [previous, current], stale)

        self.assertFalse(dd.hf_report_has_matching_change_cache(current, stale))
        self.assertEqual(("260724", 2.0, 20), (
            totals["date"], totals["hours"], totals["episodes"]))

        current["datasets"][0]["last_modified"] = None
        no_hf_date = dd.hf_last_modified_totals(
            current, [previous, current], {"version": 1, "repos": {}})
        self.assertEqual(("260724", 2.0, 20), (
            no_hf_date["date"], no_hf_date["hours"], no_hf_date["episodes"]))

    def test_baseline_ignores_other_organisations(self):
        tacverse = self._detailed_report(
            "260722", "2026-07-22T12:00:00", 10, 1000, 1.0)
        other = self._detailed_report(
            "260723", "2026-07-23T12:00:00", 900, 90000, 90.0)
        other["org"] = "OtherOrg"
        current = self._detailed_report(
            "260724", "2026-07-24T12:00:00", 30, 3000, 3.0)

        self.assertIs(tacverse, dd.find_baseline(
            current, [tacverse, other, current]))
        self.assertEqual((2.0, 20), dd.aggregate_deltas(
            current, [tacverse, other, current]))

    def test_load_history_can_filter_organisation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "dataset_log.json"
            pulls = root / "pulls"
            pulls.mkdir()
            tacverse = self._detailed_report(
                "260723", "2026-07-23T12:00:00", 10, 1000, 1.0)
            other = self._detailed_report(
                "260724", "2026-07-24T12:00:00", 20, 2000, 2.0)
            other["org"] = "OtherOrg"
            other["datasets"][0]["dataset_name"] = "OtherOrg/example"
            dd.append_pull(tacverse, log_path)
            dd.append_pull(other, log_path)

            history = dd.load_history(pulls, log_file=log_path, org="TacVerse")

            self.assertEqual(1, len(history))
            self.assertEqual("TacVerse", history[0]["org"])

    def test_manual_totals_use_dataset_log_without_group_attribution(self):
        previous = {
            "pulled_at": "2026-07-23T23:59:59", "date": "260723",
            "org": "TacVerse", "total_datasets": 10,
            "total_episodes": 100, "total_frames": 1000,
            "total_hours": 10.0, "datasets": [], "source": "manual",
        }
        current = {
            "pulled_at": "2026-07-24T23:59:59", "date": "260724",
            "org": "TacVerse", "total_datasets": 12,
            "total_episodes": 125, "total_frames": 1300,
            "total_hours": 12.5, "datasets": [], "source": "manual",
        }
        history = [previous, current]

        totals = dd.hf_last_modified_totals(current, history, {})
        groups = dd.hf_last_modified_daily_group_series(
            current, history, {}, lambda dataset: dataset.get("uploader"))

        self.assertEqual(
            {"date": "260724", "hours": 2.5, "episodes": 25, "datasets": 2},
            totals)
        self.assertEqual([], groups)

    def test_migration_merges_config_local_and_existing_log_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            legacy_path = root / "pull_history.local.json"
            log_path = root / "dataset_log.json"
            pulls = root / "pulls"
            pulls.mkdir()

            day22 = self._detailed_report(
                "260722", "2026-07-22T12:00:00", 5, 500, 0.5)
            day23 = self._detailed_report(
                "260723", "2026-07-23T12:00:00", 10, 1000, 1.0)
            day24 = self._detailed_report(
                "260724", "2026-07-24T12:00:00", 30, 3000, 3.0)
            config_path.write_text(json.dumps({
                "checks": {}, "uploader_names": {}, "pull_history": [day22],
            }), encoding="utf-8")
            legacy_path.write_text(json.dumps([day23]), encoding="utf-8")
            dd.append_pull(day24, log_path)

            dd.migrate_pull_history_to_log(
                config_path, log_path, legacy_path)
            history = dd.load_history(pulls, log_file=log_path)

            self.assertEqual(
                ["260722", "260723", "260724"],
                [row["date"] for row in history])
            self.assertNotIn("pull_history", dd.load_config(config_path))
            self.assertEqual(30, history[-1]["total_episodes"])

            with mock.patch.object(dd, "write_dataset_log") as write_log:
                dd.migrate_pull_history_to_log(
                    config_path, log_path, legacy_path)
            write_log.assert_not_called()

    def test_migration_prefers_real_details_over_later_manual_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            legacy_path = root / "pull_history.local.json"
            log_path = root / "dataset_log.json"
            config_path.write_text("{}", encoding="utf-8")

            real = self._detailed_report(
                "260724", "2026-07-24T18:17:00", 9670, 23660286, 219.037)
            real["total_datasets"] = 174
            manual = {
                "pulled_at": "2026-07-24T23:59:59", "date": "260724",
                "org": "TacVerse", "total_datasets": 9670,
                "total_episodes": 6093, "total_frames": 23660282,
                "total_hours": 219.037, "datasets": [], "source": "manual",
            }
            dd.upsert_manual_totals(
                "260724", "TacVerse", 9670, 6093, 23660282, 219.037,
                path=log_path, today=dt.date(2026, 7, 24))
            legacy_path.write_text(json.dumps([real]), encoding="utf-8")

            dd.migrate_pull_history_to_log(
                config_path, log_path, legacy_path)
            history = dd._reconstruct_history(dd.load_dataset_log(log_path))

            self.assertEqual(1, len(history))
            self.assertNotEqual(manual["source"], history[0].get("source"))
            self.assertEqual(174, history[0]["total_datasets"])
            self.assertEqual(9670, history[0]["total_episodes"])
            self.assertEqual(1, len(history[0]["datasets"]))


class SnapshotLockTests(unittest.TestCase):
    def test_different_target_dirs_get_distinct_locks(self):
        first = dd._snapshot_lock_for(Path("datasets/TacVerse/a"))
        second = dd._snapshot_lock_for(Path("datasets/TacVerse/b"))

        self.assertIsNot(first, second)
        self.assertIs(first, dd._snapshot_lock_for(Path("datasets/TacVerse/a")))


class HubEndpointTests(unittest.TestCase):
    class _Response:
        def __init__(self, status_code=401, url="https://huggingface.co/api/datasets/example",
                     history=()):
            self.status_code = status_code
            self.url = url
            self.history = list(history)

    class _HistoryItem:
        def __init__(self, status_code):
            self.status_code = status_code

    def _redirected_auth_error(self):
        response = self._Response(
            history=[self._HistoryItem(308)],
        )
        error_type = type("RepositoryNotFoundError", (Exception,), {})
        error = error_type("401 Repository Not Found")
        error.response = response
        return error

    def test_configured_mirror_precedes_official_fallback(self):
        with patch.dict(dd.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}):
            self.assertEqual(
                ["https://hf-mirror.com", dd.HF_OFFICIAL_ENDPOINT],
                dd._hub_endpoint_candidates(),
            )

    def test_redirected_mirror_auth_error_falls_back(self):
        error = self._redirected_auth_error()
        calls = []

        def read(endpoint):
            calls.append(endpoint)
            if endpoint == "https://hf-mirror.com":
                raise error
            return "official"

        with patch.dict(dd.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}), \
                patch.object(dd, "_configure_hub_endpoint"):
            self.assertEqual("official", dd._call_hub_with_fallback(read))
        self.assertEqual(
            ["https://hf-mirror.com", dd.HF_OFFICIAL_ENDPOINT], calls)

    def test_wrapped_metadata_error_falls_back(self):
        metadata_type = type("FileMetadataError", (Exception,), {})
        outer = RuntimeError("metadata lookup failed")
        outer.__cause__ = metadata_type("missing ETag")
        calls = []

        def read(endpoint):
            calls.append(endpoint)
            if endpoint == "https://hf-mirror.com":
                raise outer
            return "official"

        with patch.dict(dd.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}), \
                patch.object(dd, "_configure_hub_endpoint"), \
                patch.object(dd.time, "sleep"):
            self.assertEqual("official", dd._call_hub_with_fallback(read))
        self.assertEqual(
            ["https://hf-mirror.com"] * 3
            + [dd.HF_OFFICIAL_ENDPOINT], calls)

    def test_default_endpoint_preserves_none_argument(self):
        calls = []

        with patch.dict(dd.os.environ, {}, clear=False):
            dd.os.environ.pop("HF_ENDPOINT", None)
            with patch.object(dd, "_configure_hub_endpoint"):
                self.assertEqual(
                    "ok",
                    dd._call_hub_with_fallback(
                        lambda endpoint: calls.append(endpoint) or "ok"))
        self.assertEqual([None], calls)

    def test_malformed_redirect_history_is_ignored(self):
        response = self._Response(history=[self._HistoryItem("not-a-status")])
        self.assertFalse(dd._response_redirected(response))

    def test_fetch_username_uses_fallback_helper(self):
        with patch.object(
                dd, "_call_hub_with_fallback", return_value={"name": "tester"}) as call_hub:
            self.assertEqual("tester", dd.fetch_username("token"))
        call_hub.assert_called_once()

    def test_existing_local_dir_is_probed_before_snapshot_fallback(self):
        error = self._redirected_auth_error()
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "example"
            local_dir.mkdir()
            (local_dir / "old-file").write_text("old", encoding="utf-8")
            logs = []
            with patch.dict(
                    dd.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}), \
                    patch.object(dd, "_configure_hub_endpoint"), \
                    patch.object(
                        dd, "_probe_snapshot_endpoint",
                        side_effect=[error, object()]) as probe, \
                    patch.object(dd, "_snapshot_with_retries") as snapshot:
                dd._snapshot_to_local_with_fallback(
                    "TacVerse/example",
                    None,
                    local_dir,
                    "token",
                    log=logs.append,
                )

        self.assertEqual(2, probe.call_count)
        self.assertEqual(
            "https://hf-mirror.com",
            probe.call_args_list[0].args[3],
        )
        self.assertEqual(
            dd.HF_OFFICIAL_ENDPOINT,
            probe.call_args_list[1].args[3],
        )
        snapshot.assert_called_once()
        self.assertEqual(
            dd.HF_OFFICIAL_ENDPOINT,
            snapshot.call_args.kwargs["endpoint"],
        )
        self.assertTrue(any("切换到官方端点" in line for line in logs))


class PullDatasetRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset_dir = Path(self.tmp.name)
        self.repo_id = "TacVerse/example-dataset"
        self.local_dir = self.dataset_dir / "example-dataset"
        self.cache_error = FileNotFoundError(
            2, "No such file or directory",
            str(self.local_dir / ".cache" / "huggingface" / "download" /
                "meta" / "episodes" / "blob.12345678.incomplete"),
        )

    @patch.object(dd, "build_summary", return_value={"ok": True})
    @patch.object(dd.time, "sleep")
    def test_retries_cache_temp_error_serially(self, _sleep, summary):
        with patch.object(
                dd, "_snapshot_to_local",
                side_effect=[self.cache_error, str(self.local_dir)]) as download:
            logs = []
            result = dd.pull_dataset(
                self.repo_id, self.dataset_dir, None, "token", log=logs.append)

        self.assertEqual({"ok": True}, result)
        self.assertEqual(
            [call(self.repo_id, None, self.local_dir, "token", max_workers=8),
             call(self.repo_id, None, self.local_dir, "token", max_workers=1)],
            download.call_args_list,
        )
        self.assertTrue(any("单线程续传" in line for line in logs))
        summary.assert_called_once_with(self.repo_id, str(self.local_dir))

    @patch.object(dd, "build_summary", return_value={"ok": True})
    @patch.object(dd.time, "sleep")
    def test_retries_incomplete_chunked_read(self, _sleep, summary):
        error = RuntimeError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)")
        with patch.object(
                dd, "_snapshot_to_local",
                side_effect=[error, str(self.local_dir)]) as download:
            logs = []
            result = dd.pull_dataset(
                self.repo_id, self.dataset_dir, None, "token", log=logs.append)

        self.assertEqual({"ok": True}, result)
        self.assertEqual(
            [call(self.repo_id, None, self.local_dir, "token", max_workers=8),
             call(self.repo_id, None, self.local_dir, "token", max_workers=1)],
            download.call_args_list,
        )
        self.assertTrue(any("网络读取中断" in line for line in logs))
        summary.assert_called_once_with(self.repo_id, str(self.local_dir))

    @patch.object(dd, "build_summary")
    def test_does_not_retry_unrelated_download_error(self, summary):
        error = PermissionError("denied")
        with patch.object(dd, "_snapshot_to_local", side_effect=error) as download:
            with self.assertRaises(PermissionError):
                dd.pull_dataset(self.repo_id, self.dataset_dir, None, None, log=lambda _: None)

        self.assertEqual(1, download.call_count)
        summary.assert_not_called()

    def test_only_matches_huggingface_incomplete_cache_paths(self):
        self.assertTrue(dd._is_local_cache_temp_error(self.cache_error))
        self.assertFalse(dd._is_local_cache_temp_error(FileNotFoundError("info.json")))
        self.assertFalse(dd._is_local_cache_temp_error(RuntimeError("blob.incomplete")))

    @patch.object(dd.os, "name", "nt")
    def test_windows_hub_local_dir_uses_extended_path_prefix(self):
        path = dd._hub_local_dir(self.local_dir)
        self.assertTrue(path.startswith("\\\\?\\"))
        self.assertTrue(path.endswith("example-dataset"))


class LastModifiedRangeTests(unittest.TestCase):
    def test_group_range_totals_filter_and_sum_by_date_range(self):
        rows = [
            {"date": "260101", "group": "alice", "hours": 1.0, "episodes": 10, "datasets": 2},
            {"date": "260102", "group": "alice", "hours": 2.0, "episodes": 20, "datasets": 1},
            {"date": "260103", "group": "bob", "hours": 3.0, "episodes": 30, "datasets": 1},
        ]
        report = {"datasets": []}
        with patch.object(dd, "hf_last_modified_daily_group_series", return_value=rows):
            grouped = dd.hf_last_modified_group_range_totals(
                report, [], {}, lambda d: d.get("uploader"),
                date_from="260102", date_to="260103")
            totals = dd.hf_last_modified_range_totals(
                report, [], {}, date_from="260102", date_to="260103")

        self.assertEqual([
            {"group": "bob", "hours": 3.0, "episodes": 30, "datasets": 1},
            {"group": "alice", "hours": 2.0, "episodes": 20, "datasets": 1},
        ], grouped)
        self.assertEqual({
            "date_from": "260102",
            "date_to": "260103",
            "hours": 5.0,
            "episodes": 50,
            "datasets": 2,
        }, totals)


class PullDatasetRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.day_dir = Path(self.tmp.name)
        self.repo_id = "TacVerse/example-dataset"
        self.local_dir = self.day_dir / "example-dataset"
        self.cache_error = FileNotFoundError(
            2, "No such file or directory",
            str(self.local_dir / ".cache" / "huggingface" / "download" /
                "meta" / "episodes" / "blob.12345678.incomplete"),
        )

    @patch.object(dd, "build_summary", return_value={"ok": True})
    @patch.object(dd.time, "sleep")
    def test_retries_cache_temp_error_serially(self, _sleep, summary):
        with patch.object(
                dd, "_snapshot_to_local",
                side_effect=[self.cache_error, str(self.local_dir)]) as download:
            logs = []
            result = dd.pull_dataset(
                self.repo_id, self.day_dir, None, "token", log=logs.append)

        self.assertEqual({"ok": True}, result)
        self.assertEqual(
            [call(self.repo_id, None, self.local_dir, "token",
                  max_workers=8),
             call(self.repo_id, None, self.local_dir, "token",
                  max_workers=1)],
            download.call_args_list,
        )
        self.assertTrue(any("单线程续传" in line for line in logs))
        summary.assert_called_once_with(self.repo_id, str(self.local_dir))

    @patch.object(dd, "build_summary")
    def test_does_not_retry_unrelated_download_error(self, summary):
        error = PermissionError("denied")
        with patch.object(dd, "_snapshot_to_local", side_effect=error) as download:
            with self.assertRaises(PermissionError):
                dd.pull_dataset(self.repo_id, self.day_dir, None, None, log=lambda _: None)

        self.assertEqual(1, download.call_count)
        summary.assert_not_called()

    def test_only_matches_huggingface_incomplete_cache_paths(self):
        self.assertTrue(dd._is_local_cache_temp_error(self.cache_error))
        self.assertFalse(dd._is_local_cache_temp_error(FileNotFoundError("info.json")))
        self.assertFalse(dd._is_local_cache_temp_error(RuntimeError("blob.incomplete")))

    @patch.object(dd.os, "name", "nt")
    def test_windows_hub_local_dir_uses_extended_path_prefix(self):
        path = dd._hub_local_dir(self.local_dir)
        self.assertTrue(path.startswith("\\\\?\\"))
        self.assertTrue(path.endswith("example-dataset"))


if __name__ == "__main__":
    unittest.main()
