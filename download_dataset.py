#!/usr/bin/env python3
"""Pull one or more Hugging Face dataset repos into an organization folder.

Datasets are re-pulled on every run (`snapshot_download` syncs incrementally,
so newly merged files are fetched and unchanged ones are skipped). Each run
stores each dataset directly under the configured organization folder:

    <out-dir>/<dataset-name>/...                       # dataset files
    <out-dir>/pull_result_<YYMMDD>_<HHMMSS>.json      # aggregate summary

The list of datasets lives in DATASETS and can be overridden with repeated
`--repo-id` flags. The fields lifted from each dataset's meta/info.json are
declared in INFO_FIELDS, so extending the report is a one-line change.
"""

import argparse
import datetime as dt
import http.client
import json
import os
import sys
import threading
import time
from urllib.parse import urlsplit
from pathlib import Path

# By default every dataset under this org is discovered and pulled. Override
# with --org, or pass explicit --repo-id flags to pull a specific subset.
ORG = "TacVerse"

HF_DATASET_URL = "https://huggingface.co/datasets/{repo_id}"
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"

# `HF_ENDPOINT` is useful when a local mirror is available, but a mirror can
# also answer the metadata request while failing to serve the actual file.
# Keep the configured endpoint first, then use the official Hub as a
# transparent fallback so an interrupted or stale mirror cannot strand a
# partially downloaded dataset.
_HUB_METADATA_ERROR_NAMES = frozenset({
    "FileMetadataError",
    "LocalEntryNotFoundError",
})
_HUB_ENDPOINT_ERROR_MARKERS = (
    "distant resource does not seem to be on huggingface.co",
    "cannot find the requested files in the local cache",
)

# Fields copied verbatim from meta/info.json into each dataset's summary.
# Extend this list to surface more of info.json (e.g. "fps", "total_tasks",
# "robot_type") with no other change. `key` is the output name, `source` the
# info.json key; `required=False` skips the field for datasets that lack it.
INFO_FIELDS = [
    {"key": "total_episodes", "source": "total_episodes"},
    {"key": "total_frames", "source": "total_frames"},
    {"key": "fps", "source": "fps", "required": False},
    {"key": "robot_type", "source": "robot_type", "required": False},
    {"key": "total_tasks", "source": "total_tasks", "required": False},
]

# Assumed capture rate (frames per second) when a dataset's info.json omits fps.
DEFAULT_FPS = 30

# Serialize snapshot calls per target directory so different datasets can pull
# concurrently while a single pull and a batch pull cannot write the same dir.
_SNAPSHOT_LOCKS = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


def _snapshot_lock_for(local_dir):
    key = os.path.normcase(os.path.abspath(os.fspath(local_dir)))
    with _SNAPSHOT_LOCKS_GUARD:
        return _SNAPSHOT_LOCKS.setdefault(key, threading.RLock())


_TRANSIENT_NETWORK_MARKERS = (
    "incomplete chunked read",
    "peer closed connection without sending complete message body",
    "incomplete read",
    "response ended prematurely",
    "remote end closed connection without response",
    "connection reset by peer",
    "connection aborted",
)


def _related_exceptions(exc):
    """Yield an exception plus common wrapped causes/reasons."""
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("__cause__", "__context__", "reason"):
            nested = getattr(current, attr, None)
            if isinstance(nested, BaseException):
                stack.append(nested)
        for arg in getattr(current, "args", ()) or ():
            if isinstance(arg, BaseException):
                stack.append(arg)


def _is_transient_network_error(exc):
    """Detect short-lived HTTP body/read interruptions worth retrying."""
    transient_classes = {
        "ChunkedEncodingError",
        "ContentTooShortError",
        "IncompleteRead",
        "ProtocolError",
        "ReadTimeout",
        "ReadTimeoutError",
        "RemoteDisconnected",
    }
    for current in _related_exceptions(exc):
        if isinstance(current, (ConnectionError, TimeoutError,
                               http.client.IncompleteRead,
                               http.client.RemoteDisconnected)):
            return True
        if current.__class__.__name__ in transient_classes:
            return True
        text = str(current).lower()
        if any(marker in text for marker in _TRANSIENT_NETWORK_MARKERS):
            return True
    return False


def _is_hub_metadata_error(exc):
    """Return whether the Hub could not resolve file metadata.

    `LocalEntryNotFoundError` is often only the outer error.  The useful
    diagnosis is commonly its wrapped `FileMetadataError`, so inspect the
    complete exception chain just as the transient-network detector does.
    """
    return any(
        current.__class__.__name__ in _HUB_METADATA_ERROR_NAMES
        for current in _related_exceptions(exc)
    )


def _response_url(response):
    """Return the final URL associated with an httpx/requests response."""
    if response is None:
        return ""
    try:
        return str(response.url)
    except (AttributeError, RuntimeError):
        return ""


def _response_redirected(response):
    """Return whether a response followed a redirect."""
    history = getattr(response, "history", None) or ()
    for item in history:
        try:
            status = int(getattr(item, "status_code", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 300 <= status < 400:
            return True
    return False


def _is_hub_endpoint_error(exc, endpoint=None):
    """Return whether retrying another Hub endpoint is worthwhile.

    A mirror that redirects to the official Hub can turn an authenticated
    request into a 401, which huggingface_hub reports as
    `RepositoryNotFoundError`.  The response history/final URL and the
    non-official endpoint are important context here; a bare 401 from the
    official Hub must still be treated as an authentication error.
    """
    if _is_hub_metadata_error(exc) or _is_transient_network_error(exc):
        return True
    non_official = (
        endpoint
        and _normalize_hf_endpoint(endpoint).lower()
        != HF_OFFICIAL_ENDPOINT.lower()
    )
    endpoint_host = urlsplit(str(endpoint or "")).netloc.lower()
    for current in _related_exceptions(exc):
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        # A redirect without Hub metadata and a temporary server failure are
        # endpoint problems.  Auth/permission/not-found responses from the
        # official endpoint are not.
        if status is not None and (300 <= status < 400 or status >= 500):
            return True
        if non_official and status is not None and 400 <= status < 500:
            response_url = _response_url(response)
            response_host = urlsplit(response_url).netloc.lower()
            if (
                _response_redirected(response)
                or (response_host and response_host != endpoint_host)
                or current.__class__.__name__ in {
                    "RepositoryNotFoundError",
                    "EntryNotFoundError",
                    "HfHubHTTPError",
                }
            ):
                return True
        text = str(current).lower()
        if any(marker in text for marker in _HUB_ENDPOINT_ERROR_MARKERS):
            return True
    return False


def _retry_transient(call, *, label="", log=None, attempts=3, delay=0.6):
    """Retry idempotent Hub reads when the peer drops a chunked response."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_network_error(exc):
                raise
            if log:
                prefix = f"{label}: " if label else ""
                log(f"{prefix}网络读取中断，正在重试 {attempt}/{attempts - 1} ...")
            time.sleep(delay * attempt)


def _retry_hub_call(call, *, label="", log=None, attempts=3, delay=0.6):
    """Retry transient reads and incomplete Hub metadata lookups."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            retryable = (
                _is_transient_network_error(exc)
                or _is_hub_metadata_error(exc)
            )
            if attempt >= attempts or not retryable:
                raise
            if log:
                prefix = f"{label}: " if label else ""
                if _is_transient_network_error(exc):
                    reason = "网络读取中断"
                else:
                    reason = "Hub 文件元数据暂时不可用"
                log(f"{prefix}{reason}，正在重试 {attempt}/{attempts - 1} ...")
            time.sleep(delay * attempt)


def _normalize_hf_endpoint(endpoint):
    """Return a comparable, slash-free Hub endpoint."""
    value = str(endpoint or "").strip().rstrip("/")
    return value or HF_OFFICIAL_ENDPOINT


def _hub_endpoint_candidates():
    """Return the configured Hub endpoint followed by the official fallback."""
    configured = _normalize_hf_endpoint(os.environ.get("HF_ENDPOINT"))
    if configured.lower() == HF_OFFICIAL_ENDPOINT.lower():
        return [HF_OFFICIAL_ENDPOINT]
    return [configured, HF_OFFICIAL_ENDPOINT]


def _hub_endpoint_argument(endpoint):
    """Avoid changing legacy default call shapes when no endpoint is set."""
    if (
        not os.environ.get("HF_ENDPOINT", "").strip()
        and endpoint.lower() == HF_OFFICIAL_ENDPOINT.lower()
    ):
        return None
    return endpoint


def _is_mirror_endpoint(endpoint):
    return "hf-mirror.com" in str(endpoint or "").lower()


def _configure_hub_endpoint(endpoint):
    """Apply endpoint-specific options before importing Hub transfer code."""
    if not _is_mirror_endpoint(endpoint):
        return

    # hf-mirror does not proxy Xet's CAS service.  Force plain HTTP for both
    # fresh imports and a huggingface_hub package that was imported earlier.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_DISABLE_XET = True
    except (ImportError, AttributeError):
        pass


def _call_hub_with_fallback(call, *, label="", log=None):
    """Call a Hub operation and fall back to the official endpoint if needed.

    `call` receives the endpoint argument expected by huggingface_hub.  When
    no `HF_ENDPOINT` was configured, the official endpoint is passed as
    `None` to preserve the library default and compatibility with older Hub
    versions.
    """
    endpoints = _hub_endpoint_candidates()
    for index, endpoint in enumerate(endpoints):
        _configure_hub_endpoint(endpoint)
        try:
            return _retry_hub_call(
                lambda: call(_hub_endpoint_argument(endpoint)),
                label=label,
                log=log,
            )
        except Exception as exc:
            if (
                index == len(endpoints) - 1
                or not _is_hub_endpoint_error(exc, endpoint)
            ):
                raise
            if log:
                log(
                    f"{label + ': ' if label else ''}{endpoint} "
                    f"无法提供文件，切换到官方端点 {HF_OFFICIAL_ENDPOINT} ..."
                )
    raise AssertionError("unreachable")


def fetch_username(token):
    """Resolve the account name for ``token`` using the configured Hub.

    The GUI uses this as a lightweight authentication check.  It deliberately
    lives next to the other Hub reads so a mirror that redirects or rejects
    ``whoami`` gets the same official-endpoint fallback as dataset metadata.
    """
    if not token:
        return ""
    from huggingface_hub import HfApi

    result = _call_hub_with_fallback(
        lambda endpoint: HfApi(
            **({"endpoint": endpoint} if endpoint else {}),
        ).whoami(token=token),
        label="验证 Hugging Face 登录",
    )
    return result.get("name", "") or ""


def normalize_proxy_env() -> None:
    """Normalize proxy and Hub endpoint settings for huggingface_hub.

    httpx rejects a schemeless `socks://` proxy URL. The http(s)_proxy vars
    already cover HTTPS traffic to the Hub, so drop the offending ALL_PROXY
    vars and normalize any remaining socks:// value to socks5://.  Preserve a
    caller-provided HF_ENDPOINT, while endpoint-specific calls can still fall
    back to the official Hub.
    """
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        val = os.environ.get(var)
        if val and val.startswith("socks://"):
            os.environ[var] = "socks5://" + val[len("socks://"):]
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint is not None:
        os.environ["HF_ENDPOINT"] = _normalize_hf_endpoint(endpoint)
    _configure_hub_endpoint(_normalize_hf_endpoint(endpoint))


def _apply_info(summary: dict, info: dict) -> dict:
    """Fill INFO_FIELDS + derived duration_hours into `summary` from an info dict."""
    for field in INFO_FIELDS:
        src = field["source"]
        if src in info:
            summary[field["key"]] = info[src]
        elif field.get("required", True):
            summary[field["key"]] = None
    # Recording duration in hours = frames / fps / 3600.
    frames = summary.get("total_frames")
    if frames is not None:
        fps = summary.get("fps") or DEFAULT_FPS
        summary["duration_hours"] = round(frames / fps / 3600, 3)
    return summary


def build_summary(repo_id: str, local_dir: str) -> dict:
    """Assemble a per-dataset summary from a *downloaded* dataset directory.

    Derived fields (name, link, local_dir) plus every entry in INFO_FIELDS read
    from meta/info.json. Missing info.json or missing keys degrade gracefully.
    """
    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
        "local_dir": str(local_dir),
    }
    info_path = Path(local_dir) / "meta" / "info.json"
    if info_path.is_file():
        _apply_info(summary, json.loads(info_path.read_text()))
    else:
        # Not a LeRobot-style dataset (no meta/info.json); leave the
        # info-derived fields absent rather than guessing.
        print(f"Note: {info_path} not found; summary limited to name and link.")
    tasks_path = Path(local_dir) / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        import tasks_reader

        rows, _ = tasks_reader.load(tasks_path)
        summary["tasks"] = rows
    return summary


def fetch_tasks(repo_id: str, token=None) -> list:
    """Fetch a dataset's task instructions from meta/tasks.parquet.

    tasks.parquet is a tiny file (a few KB) carrying the natural-language task
    string(s) the dataset was recorded against — the base "prompt". Fetched on
    the stats-only path so the dashboard can show prompts without a full pull.
    Returns [{"index", "task"}] (sorted), or [] if absent/unreadable.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = _call_hub_with_fallback(
            lambda endpoint: hf_hub_download(
                repo_id=repo_id,
                filename="meta/tasks.parquet",
                repo_type="dataset",
                token=token,
                **({"endpoint": endpoint} if endpoint else {}),
            ),
            label=f"读取 {repo_id} tasks",
        )
    except Exception:
        return []  # dataset has no tasks.parquet (or no access)
    import tasks_reader

    rows, _ = tasks_reader.load(path)
    return rows


def fetch_summary(repo_id: str, token=None) -> dict:
    """Summarize a dataset by fetching only small meta files (no full download).

    Used for the stats-only path: downloads meta/info.json (+ meta/tasks.parquet
    for the task prompt) instead of the whole (potentially huge) dataset. Falls
    back to name+link if info.json is absent.
    """
    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
    }
    try:
        from huggingface_hub import hf_hub_download

        info_path = _call_hub_with_fallback(
            lambda endpoint: hf_hub_download(
                repo_id=repo_id,
                filename="meta/info.json",
                repo_type="dataset",
                token=token,
                **({"endpoint": endpoint} if endpoint else {}),
            ),
            label=f"读取 {repo_id} info",
        )
    except Exception:
        return summary  # no info.json -> name+link only
    _apply_info(summary, json.loads(Path(info_path).read_text()))
    summary["tasks"] = fetch_tasks(repo_id, token)
    return summary


def discover_datasets_meta(org, token):
    """Return [{"id", "last_modified"}] for every dataset under an org/user.

    Ordered most-recently-updated first, matching the Hugging Face org page's
    default "Recently updated" sort. Datasets missing a timestamp sort last.
    """
    # Ask the Hub for its own "Recently updated" ranking when available; older
    # huggingface_hub versions (e.g. 1.23.x) do not expose a `direction` kwarg,
    # so we sort client-side as the source of truth and also pin timestamp-less
    # repos last.
    from huggingface_hub import HfApi

    ds = _call_hub_with_fallback(
        lambda endpoint: list(
            HfApi(
                **({"endpoint": endpoint} if endpoint else {}),
            ).list_datasets(
                author=org,
                token=token,
                sort="lastModified",
            )
        ),
        label=f"发现 {org} 数据集",
    )
    ds.sort(key=lambda d: (d.last_modified is not None, d.last_modified), reverse=True)
    out = []
    for d in ds:
        lm = d.last_modified
        out.append({"id": d.id, "last_modified": lm.isoformat() if lm else None})
    return out


def discover_datasets(org, token):
    """Return every dataset repo id under an org/user (recently-updated first)."""
    return [d["id"] for d in discover_datasets_meta(org, token)]


def fetch_uploader(repo_id, token=None):
    """Return uploader info from the dataset's HF commit history.

    `uploader` is the author of the earliest (initial) commit — i.e. who created
    the dataset. `uploaders` lists every distinct commit author. Degrades to an
    empty dict on any error (private repo, network, etc.).
    """
    try:
        from huggingface_hub import HfApi

        commits = _call_hub_with_fallback(
            lambda endpoint: HfApi(
                **({"endpoint": endpoint} if endpoint else {}),
            ).list_repo_commits(
                repo_id, repo_type="dataset", token=token),
            label=f"读取 {repo_id} commits",
        )
    except Exception:
        return {}
    if not commits:
        return {}
    # Commits come newest-first; the last one is the initial commit.
    authors, seen = [], set()
    for c in commits:
        for a in (getattr(c, "authors", None) or []):
            if a not in seen:
                seen.add(a)
                authors.append(a)
    initial = commits[-1]
    creator = (getattr(initial, "authors", None) or [None])[0]
    last_at = getattr(commits[0], "created_at", None)
    return {
        "uploader": creator,
        "uploaders": authors,
        "last_commit_at": last_at.isoformat() if last_at else None,
    }


def _is_local_cache_temp_error(exc):
    """Whether an exception is a recoverable local_dir temp-file failure."""
    if not isinstance(exc, FileNotFoundError):
        return False
    path = str(getattr(exc, "filename", "") or exc).replace("\\", "/").lower()
    return "/.cache/huggingface/download/" in path and ".incomplete" in path


def _hub_local_dir(local_dir):
    """Use Win32's extended path form for Hub cache files over MAX_PATH."""
    path = os.path.abspath(os.fspath(local_dir))
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path


def _snapshot_to_local(
        repo_id, revision, local_dir, token, *, max_workers, endpoint=None):
    _configure_hub_endpoint(endpoint or _normalize_hf_endpoint(
        os.environ.get("HF_ENDPOINT")))
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "local_dir": _hub_local_dir(local_dir),
        "token": token,
        "max_workers": max_workers,
    }
    if endpoint:
        kwargs["endpoint"] = endpoint
    return snapshot_download(**kwargs)


def _probe_snapshot_endpoint(repo_id, revision, token, endpoint, *, log=None):
    """Resolve the remote repo before syncing an existing local directory.

    ``snapshot_download`` deliberately returns a non-empty ``local_dir`` when
    its initial Hub request fails. That offline convenience is unsafe for a
    sync operation: a mirror may have redirected to the official Hub and
    returned 401, while the caller is told that an old directory is current.
    A small ``repo_info`` request makes the endpoint choice explicit before
    that fallback can happen.
    """
    from huggingface_hub import HfApi

    endpoint_arg = _hub_endpoint_argument(endpoint)
    api_kwargs = {"token": token}
    if endpoint_arg:
        api_kwargs["endpoint"] = endpoint_arg
    api = HfApi(**api_kwargs)

    def resolve():
        try:
            return api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except TypeError:
            # Older huggingface_hub releases may not expose ``token`` on the
            # method; the token supplied to HfApi still covers those versions.
            return api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )

    return _retry_hub_call(
        resolve,
        label=f"检查 {repo_id} 远端",
        log=log,
    )


def _snapshot_with_retries(
        repo_id, revision, local_dir, token, *, endpoint, log):
    """Run one endpoint's snapshot with worker and transfer retries."""
    attempts = [8, 1, 1]
    retry_note = None
    endpoint_arg = _hub_endpoint_argument(endpoint)
    for index, workers in enumerate(attempts):
        if retry_note:
            log(retry_note)
            time.sleep(0.25)
            retry_note = None
        try:
            kwargs = {
                "max_workers": workers,
            }
            if endpoint_arg:
                kwargs["endpoint"] = endpoint_arg
            _snapshot_to_local(
                repo_id, revision, local_dir, token, **kwargs,
            )
            return
        except Exception as exc:
            if _is_local_cache_temp_error(exc):
                retry_note = "检测到缓存临时文件异常，正在单线程续传…"
            elif _is_transient_network_error(exc):
                retry_note = "网络读取中断，正在续传…"
            elif _is_hub_metadata_error(exc):
                retry_note = "Hub 文件元数据暂时不可用，正在重试…"
            if not retry_note or index == len(attempts) - 1:
                raise


def _snapshot_to_local_with_fallback(
        repo_id, revision, local_dir, token, *, log):
    """Download a snapshot, switching to the official Hub when necessary."""
    endpoints = _hub_endpoint_candidates()
    for index, endpoint in enumerate(endpoints):
        _configure_hub_endpoint(endpoint)
        try:
            # When local_dir is non-empty, snapshot_download can turn an
            # endpoint/auth failure into a successful-looking return. Probe
            # first so an old local copy never masks a failed remote sync.
            local_path = Path(local_dir)
            if local_path.is_dir() and any(local_path.iterdir()):
                _probe_snapshot_endpoint(
                    repo_id, revision, token, endpoint, log=log,
                )
            _snapshot_with_retries(
                repo_id,
                revision,
                local_dir,
                token,
                endpoint=endpoint,
                log=log,
            )
            return
        except Exception as exc:
            if (
                index == len(endpoints) - 1
                or not _is_hub_endpoint_error(exc, endpoint)
            ):
                raise
            log(
                f"{endpoint} 无法提供 {repo_id} 的文件，"
                f"切换到官方端点 {HF_OFFICIAL_ENDPOINT} ..."
            )


def pull_dataset(repo_id, dataset_dir, revision, token, log=print):
    """Download one dataset into <dataset_dir>/<dataset-name> and summarize it."""
    local_dir = Path(dataset_dir) / repo_id.split("/")[-1]
    log(f"Downloading {repo_id} -> {local_dir}")
    with _snapshot_lock_for(local_dir):
        _snapshot_to_local_with_fallback(
            repo_id, revision, local_dir, token, log=log,
        )
    return build_summary(repo_id, str(local_dir))


def build_report(summaries, failures, now, org, requested):
    """Build the aggregate report dict (totals first, then per-dataset list)."""
    agg_episodes = sum(x.get("total_episodes") or 0 for x in summaries)
    agg_frames = sum(x.get("total_frames") or 0 for x in summaries)
    agg_hours = round(sum(x.get("duration_hours") or 0 for x in summaries), 3)
    report = {
        "total_datasets": len(summaries),
        "total_episodes": agg_episodes,
        "total_frames": agg_frames,
        "total_hours": agg_hours,
        "pulled_at": now.isoformat(timespec="seconds"),
        "date": now.strftime("%y%m%d"),
        "org": org,
        "requested": requested,
        "count": len(summaries),
        "datasets": summaries,
    }
    if failures:
        report["failures"] = failures
    return report


def run_pull(repo_ids, out_dir, org, revision=None, token=None, now=None,
             log=print, progress=None, write_summary=True,
             meta_map=None, with_uploader=True):
    """Pull every repo in `repo_ids` into one organization folder and write a report.

    `log(msg)` receives human-readable progress lines (same text as the CLI).
    `progress(done, total)` is called before and after each dataset so a UI can
    drive a progress bar. Returns (report_dict, out_path_or_None).
    """
    now = now or dt.datetime.now()
    dataset_dir = Path(out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = pull_dataset(repo_id, dataset_dir, revision, token, log=log)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:  # keep pulling the rest if one fails
            log(f"ERROR pulling {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)

    report = build_report(summaries, failures, now, org, total)
    out_path = None
    if write_summary:
        out_path = dataset_dir / f"pull_result_{now.strftime('%y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        log(f"Wrote summary -> {out_path}")
    try:
        append_pull(report)  # git-committed change-log; survives datasets/ being ignored
        log(f"Updated history -> {DATASET_LOG_FILE}")
    except OSError as exc:
        log(f"WARN: could not update {DATASET_LOG_FILE}: {exc}")
    return report, out_path


def _enrich(summary, repo_id, meta_map, with_uploader, token):
    """Attach last_modified (from meta_map) and uploader fields to a summary."""
    if meta_map and repo_id in meta_map:
        summary["last_modified"] = meta_map[repo_id]
    if with_uploader:
        summary.update(fetch_uploader(repo_id, token))
    return summary


def collect_stats(repo_ids, org, token=None, now=None, log=print, progress=None,
                  meta_map=None, with_uploader=True):
    """Build a report from meta/info.json only — no dataset files downloaded.

    Same report shape as run_pull (totals + per-dataset list), so a UI can show
    it in the exact same dashboard. Per-dataset entries have no local_dir but do
    carry last_modified + uploader when meta_map/with_uploader are supplied.
    """
    now = now or dt.datetime.now()
    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = fetch_summary(repo_id, token)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:
            log(f"ERROR reading {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)
    return build_report(summaries, failures, now, org, total)


def find_latest_report(out_dir):
    """Return the newest pull_result_*.json directly under an org directory."""
    files = sorted(Path(out_dir).glob("pull_result_*.json"))
    if not files:
        files = sorted(Path(out_dir).glob("*/pull_result_*.json"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# Analytics helpers (pure functions over report dicts — used by the GUI)
# --------------------------------------------------------------------------- #
# Two git-committed json files at the repo root, both travelling with the code so
# a fresh clone gets the collection trend / 每日新增 WITHOUT syncing the multi-GB
# datasets/ folder (which is .gitignore'd):
#
#   config.json       — hand-edited only:
#                        { "checks": {...}, "uploader_names": {"<hf_id>": "<中文名>"} }
#   dataset_log.json  — auto-appended per-DATASET change log (no time-major dupes):
#     { "dataset_index": [ "<name>", ... ],                     # names stored once
#       "daily_totals": [ {pulled_at,date,org,total_*,present:[idx],
#                            source?:"manual"} ],  # 1 row/pull/manual snapshot
#       "datasets": { "<name>": { <meta>, "changes": [ {date,pulled_at,
#                                total_*,d_*} ] } } }  # a row ONLY when totals moved
CONFIG_FILE = str(Path(__file__).parent / "config.json")
DATASET_LOG_FILE = str(Path(__file__).parent / "dataset_log.json")
CHANGE_HISTORY_FILE = str(Path(__file__).parent / "hf_change_history.local.json")
LEGACY_HISTORY_FILE = str(Path(__file__).parent / "pull_history.local.json")

# Per-dataset fields exposed to the GUI in a reconstructed snapshot row.
_HISTORY_DS_FIELDS = (
    "dataset_name", "total_episodes", "total_frames", "duration_hours",
    "fps", "robot_type", "total_tasks", "uploader", "last_modified",
)
# Slow-changing metadata stored once per dataset (latest value wins).
_DS_META_FIELDS = ("fps", "robot_type", "total_tasks", "uploader", "last_modified")
# Cumulative totals whose movement triggers a new `changes` entry.
_DS_TOTAL_FIELDS = ("total_episodes", "total_frames", "duration_hours")


def _load_json(path):
    """Read a JSON file; returns None if missing/corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_config(path=CONFIG_FILE):
    """Read the committed config; returns {} if missing/corrupt."""
    cfg = _load_json(path)
    return cfg if isinstance(cfg, dict) else {}


def load_uploader_names(path=CONFIG_FILE):
    """The hand-edited HF id -> Chinese name map from the config file."""
    return load_config(path).get("uploader_names", {}) or {}


# --------------------------------------------------------------------------- #
# Dataset change-log (dataset_log.json) — read / append / reconstruct
# --------------------------------------------------------------------------- #
def load_dataset_log(path=DATASET_LOG_FILE):
    """Read the dataset change-log; returns an empty skeleton on missing/corrupt."""
    try:
        log = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log = None
    if not isinstance(log, dict):
        log = {}
    log.setdefault("dataset_index", [])
    log.setdefault("daily_totals", [])
    log.setdefault("datasets", {})
    return log


def _dataset_log_text(log, present_per_line=16):
    """Serialize the change-log in a compact but still reviewable layout.

    Normal ``json.dumps(..., indent=2)`` places every integer on its own line,
    and expands every small dataset/change object across many lines.  Keep the
    top-level collections readable, group ``present`` indices, render each
    change object on one line, and render each dataset's metadata on one line.
    The result remains plain JSON and round-trips through any standard parser.
    """
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    def append_regular_property(lines, key, value, trailing_comma):
        dumped = json.dumps(value, indent=2, ensure_ascii=False).splitlines()
        key_text = json.dumps(key, ensure_ascii=False)
        if len(dumped) == 1:
            lines.append(f"  {key_text}: {dumped[0]}" + ("," if trailing_comma else ""))
            return
        lines.append(f"  {key_text}: {dumped[0]}")
        lines.extend("  " + line for line in dumped[1:-1])
        lines.append("  " + dumped[-1] + ("," if trailing_comma else ""))

    def append_datasets(lines, datasets, trailing_comma):
        lines.append('  "datasets": {')
        items = list(datasets.items())
        for dataset_i, (name, section) in enumerate(items):
            dataset_comma = dataset_i < len(items) - 1
            lines.append(f"    {json.dumps(name, ensure_ascii=False)}: {{")
            changes = section.get("changes", []) or []
            metadata = [(key, value) for key, value in section.items()
                        if key != "changes"]
            if changes:
                lines.append('      "changes": [')
                for change_i, change in enumerate(changes):
                    if change_i < len(changes) - 1:
                        suffix = ","
                    else:
                        suffix = "]" + ("," if metadata else "")
                    lines.append("        " + compact(change) + suffix)
            else:
                lines.append('      "changes": []' + ("," if metadata else ""))

            if metadata:
                meta_text = ", ".join(
                    f"{json.dumps(key, ensure_ascii=False)}: {compact(value)}"
                    for key, value in metadata)
                lines.append("      " + meta_text + "}" + ("," if dataset_comma else ""))
            else:
                lines.append("    }" + ("," if dataset_comma else ""))
        lines.append("  }" + ("," if trailing_comma else ""))

    lines = ["{"]
    properties = list(log.items())
    for prop_i, (key, value) in enumerate(properties):
        trailing_comma = prop_i < len(properties) - 1
        if key == "datasets" and isinstance(value, dict):
            append_datasets(lines, value, trailing_comma)
        else:
            append_regular_property(lines, key, value, trailing_comma)
    lines.append("}")

    # Compact only the already-rendered multi-line present arrays; empty arrays
    # stay as ``[]`` and all other arrays retain their dedicated layouts above.
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != '"present": [':
            out.append(line)
            i += 1
            continue

        values = []
        j = i + 1
        while j < len(lines) and lines[j].strip() not in ("]", "],"):
            values.append(lines[j].strip().rstrip(","))
            j += 1
        if not values or j >= len(lines):
            out.append(line)
            i += 1
            continue

        out.append(line)
        value_indent = line[:len(line) - len(line.lstrip())] + "  "
        for start in range(0, len(values), present_per_line):
            chunk = values[start:start + present_per_line]
            suffix = "," if start + present_per_line < len(values) else ""
            out.append(value_indent + ", ".join(chunk) + suffix)
        out.append(lines[j])
        i = j + 1
    return "\n".join(out) + "\n"


def write_dataset_log(log, path=DATASET_LOG_FILE):
    """Write ``log`` using the repository's readable compact JSON format."""
    Path(path).write_text(_dataset_log_text(log), encoding="utf-8")
    return path


def _fold_report_into_log(log, report):
    """Merge one report into `log` in place. Appends the per-pull aggregate row
    and, per dataset, a change entry ONLY when its totals moved — recording both
    the resulting totals and the +delta vs the previous entry.

    Dataset names are stored once in the shared `dataset_index`; each daily_totals
    row lists the datasets present at that pull as integer indices into it
    (`present`), so names aren't re-listed in full on every pull. Reports must be
    folded oldest-first so deltas chain correctly."""
    at = report.get("pulled_at")
    date = report.get("date")
    org = report.get("org")
    totals = log.setdefault("daily_totals", [])

    # One authoritative snapshot per organisation and calendar day.  Ignore an
    # out-of-order older report instead of allowing it to replace newer data.
    same_day = [t for t in totals
                if t.get("date") == date and t.get("org") == org]
    if same_day and max((t.get("pulled_at") or "") for t in same_day) > (at or ""):
        return log

    # Remove the earlier aggregate row and its per-dataset changes before
    # calculating this pull's deltas.  The remaining previous change is then
    # from an earlier day, so d_* represents the whole day's growth.
    totals[:] = [t for t in totals
                 if not (t.get("date") == date and t.get("org") == org)]
    report_names = {d.get("dataset_name") for d in report.get("datasets", [])
                    if d.get("dataset_name")}
    prefix = f"{org}/" if org else None
    for name, sect in log.setdefault("datasets", {}).items():
        if name not in report_names and not (prefix and name.startswith(prefix)):
            continue
        changes = sect.setdefault("changes", [])
        changes[:] = [c for c in changes if c.get("date") != date]

    index = log.setdefault("dataset_index", [])
    pos = {n: i for i, n in enumerate(index)}
    dsets = log.setdefault("datasets", {})

    present = []
    for d in report.get("datasets", []):
        name = d.get("dataset_name")
        if not name:
            continue
        if name not in pos:                    # register the name once
            pos[name] = len(index)
            index.append(name)
        present.append(pos[name])              # present even if unchanged below
        sect = dsets.setdefault(name, {"changes": []})
        for k in _DS_META_FIELDS:              # keep newest slow-changing metadata
            sect[k] = d.get(k)
        changes = sect.setdefault("changes", [])
        changes[:] = [c for c in changes if c.get("pulled_at") != at]  # idempotent
        cur = {k: d.get(k) for k in _DS_TOTAL_FIELDS}
        prev = changes[-1] if changes else None
        if prev and all(prev.get(k) == cur.get(k) for k in _DS_TOTAL_FIELDS):
            continue                            # unchanged -> write no change row
        base = prev or {}
        changes.append({
            "date": report.get("date"), "pulled_at": at, **cur,
            "d_episodes": (cur.get("total_episodes") or 0) - (base.get("total_episodes") or 0),
            "d_frames": (cur.get("total_frames") or 0) - (base.get("total_frames") or 0),
            "d_hours": round((cur.get("duration_hours") or 0) - (base.get("duration_hours") or 0), 3),
        })

    row = {
        "pulled_at": at,
        "date": report.get("date"),
        "org": report.get("org"),
        "total_datasets": report.get("total_datasets"),
        "total_episodes": report.get("total_episodes"),
        "total_frames": report.get("total_frames"),
        "total_hours": report.get("total_hours"),
        "present": sorted(present),
    }
    if report.get("source"):
        row["source"] = report["source"]
    totals.append(row)
    totals.sort(key=lambda t: t.get("pulled_at") or "")
    return log


def load_hf_change_history(path=CHANGE_HISTORY_FILE):
    """Read local Hugging Face commit-diff history cache."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict):
        data.setdefault("version", 1)
        data.setdefault("repos", {})
        return data
    return {"version": 1, "repos": {}}


def save_hf_change_history(data, path=CHANGE_HISTORY_FILE):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return path


def _legacy_history_rows(path=LEGACY_HISTORY_FILE):
    """Load the old local pull history file when it still exists."""
    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("pull_history", []) or []
    return []


def append_pull(report, path=DATASET_LOG_FILE):
    """Fold `report` into the git-committed dataset change-log at `path`.

    Unchanged datasets are not re-stored on every pull, so the file stays small.
    Re-reads first; safe to re-run. Returns the path written.
    """
    log = load_dataset_log(path)
    # A real stats/pull on the same day supersedes an aggregate-only manual
    # placeholder.  This prevents the synthetic 23:59:59 manual timestamp from
    # continuing to win after fresh detailed data becomes available.
    totals = log.setdefault("daily_totals", [])
    totals[:] = [t for t in totals if not (
        t.get("source") == "manual"
        and t.get("date") == report.get("date")
        and t.get("org") == report.get("org")
    )]
    _fold_report_into_log(log, report)
    return write_dataset_log(log, path)


def upsert_manual_totals(date, org, total_datasets, total_episodes,
                         total_frames, total_hours, path=DATASET_LOG_FILE,
                         today=None):
    """Insert or replace one aggregate-only manual snapshot.

    ``date`` uses the same YYMMDD format as regular reports.  The synthetic
    end-of-day timestamp makes the manual snapshot authoritative for that day;
    :func:`append_pull` removes it if real detailed statistics are later pulled
    for the same organisation and date.
    """
    try:
        day = dt.datetime.strptime(date, "%y%m%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("date must use YYMMDD format") from exc
    today = today or dt.date.today()
    if day > today:
        raise ValueError("manual snapshot date cannot be in the future")
    if not isinstance(org, str) or not org.strip():
        raise ValueError("org cannot be empty")

    integer_fields = {
        "total_datasets": total_datasets,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    }
    normalized = {}
    for key, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        normalized[key] = value
    if isinstance(total_hours, bool) or not isinstance(total_hours, (int, float)) \
            or total_hours < 0:
        raise ValueError("total_hours must be a non-negative number")

    report = {
        "pulled_at": day.strftime("%Y-%m-%dT23:59:59"),
        "date": date,
        "org": org.strip(),
        **normalized,
        "total_hours": round(float(total_hours), 3),
        "datasets": [],
        "source": "manual",
    }
    log = load_dataset_log(path)
    totals = log.setdefault("daily_totals", [])
    totals[:] = [t for t in totals if not (
        t.get("source") == "manual"
        and t.get("date") == date
        and t.get("org") == report["org"]
    )]
    _fold_report_into_log(log, report)
    write_dataset_log(log, path)
    return report


def _state_as_of(log, pulled_at, names):
    """Per-dataset snapshot rows for `names` as they stood at `pulled_at`.

    For each name, take the last change entry with pulled_at <= the target and
    merge in the dataset's metadata, rebuilding a row shaped like the old
    pull_history datasets[] entries (_HISTORY_DS_FIELDS)."""
    dsets = log.get("datasets", {})
    target = pulled_at or ""
    rows = []
    for name in names:
        sect = dsets.get(name)
        if not sect:
            continue
        latest = None
        for c in sect.get("changes", []):
            if (c.get("pulled_at") or "") <= target:
                latest = c
        if latest is None:
            continue
        row = {"dataset_name": name}
        for k in _DS_TOTAL_FIELDS:
            row[k] = latest.get(k)
        for k in _DS_META_FIELDS:
            row[k] = sect.get(k)
        rows.append(row)
    return rows


def _reconstruct_history(log):
    """Rebuild the oldest-first list of full pull snapshots from the change-log,
    equivalent to the old config['pull_history'] shape so downstream analytics
    (daily_series / find_baseline / compute_deltas) need no changes."""
    index = log.get("dataset_index", [])
    snaps = []
    for row in sorted(log.get("daily_totals", []),
                      key=lambda t: t.get("pulled_at") or ""):
        names = [index[i] for i in row.get("present", []) if 0 <= i < len(index)]
        snap = {k: row.get(k) for k in
                ("pulled_at", "date", "org", "total_datasets",
                 "total_episodes", "total_frames", "total_hours")}
        if row.get("source"):
            snap["source"] = row["source"]
        snap["datasets"] = _state_as_of(log, row.get("pulled_at"), names)
        snaps.append(snap)
    return snaps


def compact_dataset_log(path=DATASET_LOG_FILE):
    """Rewrite an existing log with only its latest rows per calendar day.

    The cleanup is deliberately in-place: dataset metadata and orphaned index
    entries are retained, while aggregate rows and each dataset's change rows
    are collapsed independently.  Deltas are then recomputed against the prior
    retained day.
    """
    log = load_dataset_log(path)
    before = len(log.get("daily_totals", []))
    latest = {}
    for row in log.get("daily_totals", []):
        key = (row.get("date") or row.get("pulled_at"), row.get("org"))
        previous = latest.get(key)
        if previous is None or (row.get("pulled_at") or "") \
                >= (previous.get("pulled_at") or ""):
            latest[key] = row
    log["daily_totals"] = sorted(
        latest.values(), key=lambda r: r.get("pulled_at") or "")

    for sect in log.get("datasets", {}).values():
        by_day = {}
        for change in sect.get("changes", []):
            key = change.get("date") or change.get("pulled_at")
            previous = by_day.get(key)
            if previous is None or (change.get("pulled_at") or "") \
                    >= (previous.get("pulled_at") or ""):
                by_day[key] = change
        changes = sorted(by_day.values(), key=lambda c: c.get("pulled_at") or "")
        previous = None
        for change in changes:
            base = previous or {}
            change["d_episodes"] = ((change.get("total_episodes") or 0)
                                     - (base.get("total_episodes") or 0))
            change["d_frames"] = ((change.get("total_frames") or 0)
                                   - (base.get("total_frames") or 0))
            change["d_hours"] = round(
                (change.get("duration_hours") or 0)
                - (base.get("duration_hours") or 0), 3)
            previous = change
        sect["changes"] = changes

    write_dataset_log(log, path)
    return before, len(log["daily_totals"])


def load_history(out_dir, log_file=DATASET_LOG_FILE, org=None):
    """Load pull snapshots oldest-first for trends / deltas.

    Reconstructs the committed snapshots from the dataset change-log and merges
    them with any local pull_result_*.json still on disk, deduping by
    pulled_at so both sources contribute but neither double-counts.
    """
    by_at = {}
    for r in _reconstruct_history(load_dataset_log(log_file)):
        if org is not None and r.get("org") != org:
            continue
        key = (r.get("pulled_at") or id(r), r.get("org"))
        by_at[key] = r
    files = sorted(Path(out_dir).glob("pull_result_*.json"))
    if not files:
        files = sorted(Path(out_dir).glob("*/pull_result_*.json"))
    for f in files:
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        key = (r.get("pulled_at") or str(f), r.get("org"))
        by_at.setdefault(key, r)  # log wins on ties
    history = list(by_at.values())
    history.sort(key=lambda r: r.get("pulled_at", ""))
    return history


def load_latest_local_report(out_dir, org=ORG):
    """Return the newest locally available report without network access.

    Priority: explicit pull_result JSON, then the committed dataset log, then a
    best-effort scan of downloaded <dataset>/meta/info.json directories.
    """
    latest = find_latest_report(out_dir)
    if latest:
        data = _load_json(latest)
        if isinstance(data, dict) and data.get("datasets"):
            return data, str(latest)

    history = load_history(out_dir, org=org)
    if history:
        report = history[-1]
        if isinstance(report, dict) and report.get("datasets"):
            return report, DATASET_LOG_FILE

    summaries = []
    for info in sorted(Path(out_dir).glob("*/meta/info.json")):
        dataset_dir = info.parent.parent
        summaries.append(build_summary(f"{org}/{dataset_dir.name}", str(dataset_dir)))
    if summaries:
        latest_time = max((Path(s["local_dir"]).stat().st_mtime for s in summaries), default=None)
        now = dt.datetime.fromtimestamp(latest_time) if latest_time else dt.datetime.now()
        return build_report(summaries, [], now, org, len(summaries)), str(Path(out_dir))
    return None, None


def migrate_pull_history_to_log(config_path=CONFIG_FILE, log_path=DATASET_LOG_FILE,
                                legacy_history_path=LEGACY_HISTORY_FILE):
    """Merge legacy pull-history sources into the committed dataset log.

    Supports the old config["pull_history"] field and the old local
    pull_history.local.json file. New pulls only write dataset_log.json.
    """
    cfg = load_config(config_path)
    legacy = list(cfg.get("pull_history", []) or [])
    legacy.extend(_legacy_history_rows(legacy_history_path))
    existing_log = load_dataset_log(log_path)
    existing = _reconstruct_history(existing_log)

    if not legacy and not existing_log.get("daily_totals"):
        cfg.pop("pull_history", None)
        Path(config_path).write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return log_path

    latest_log_rows = {}
    for row in existing_log.get("daily_totals", []):
        key = (row.get("date"), row.get("org"))
        previous = latest_log_rows.get(key)
        if previous is None or (row.get("pulled_at") or "") \
                >= (previous.get("pulled_at") or ""):
            latest_log_rows[key] = row
    latest_legacy_rows = {}
    for row in legacy:
        key = (row.get("date"), row.get("org"))
        previous = latest_legacy_rows.get(key)
        row_rank = (
            row.get("source") != "manual" and bool(row.get("datasets")),
            row.get("pulled_at") or "",
        )
        previous_rank = (
            previous.get("source") != "manual" and bool(previous.get("datasets")),
            previous.get("pulled_at") or "",
        ) if previous else (False, "")
        if previous is None or row_rank >= previous_rank:
            latest_legacy_rows[key] = row

    covered = True
    for key, legacy_row in latest_legacy_rows.items():
        log_row = latest_log_rows.get(key)
        legacy_is_detailed = (
            legacy_row.get("source") != "manual"
            and bool(legacy_row.get("datasets"))
        )
        if log_row is None or (legacy_is_detailed and log_row.get("source") == "manual") \
                or (log_row.get("pulled_at") or "") < (legacy_row.get("pulled_at") or ""):
            covered = False
            break

    if covered:
        if "pull_history" in cfg:
            cfg.pop("pull_history", None)
            Path(config_path).write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
        return log_path

    by_key = {}
    for snap in legacy + existing:
        by_key[(snap.get("pulled_at") or id(snap), snap.get("org"))] = snap

    snapshots = list(by_key.values())
    detailed_days = {
        (snap.get("date"), snap.get("org"))
        for snap in snapshots
        if snap.get("source") != "manual" and snap.get("datasets")
    }
    snapshots = [
        snap for snap in snapshots
        if not (snap.get("source") == "manual"
                and (snap.get("date"), snap.get("org")) in detailed_days)
    ]

    log = {"dataset_index": [], "daily_totals": [], "datasets": {}}
    for snap in sorted(snapshots, key=lambda r: r.get("pulled_at") or ""):
        _fold_report_into_log(log, snap)
    write_dataset_log(log, log_path)

    if "pull_history" in cfg:
        cfg.pop("pull_history", None)
        Path(config_path).write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return log_path


def daily_series(history):
    """Collapse history to one snapshot per day (the day's last pull).

    `total_hours` is the absolute library total at that snapshot (already
    cumulative). `new_hours` is the day-over-day increase (this day's total
    minus the previous pulled day's total); the first day's `new_hours` equals
    its total. Returns a date-sorted list of {date, total_hours, new_hours,
    total_episodes, total_frames, total_datasets} for trend charts.
    """
    by_day = {}
    for r in history:  # history is oldest-first, so later pulls overwrite
        by_day[r.get("date", "")] = r
    series = []
    prev_total = None
    for date in sorted(k for k in by_day if k):
        r = by_day[date]
        total = r.get("total_hours", 0) or 0
        new_hours = total if prev_total is None else round(total - prev_total, 3)
        prev_total = total
        series.append({
            "date": date,
            "total_hours": total,
            "new_hours": new_hours,
            "total_episodes": r.get("total_episodes", 0) or 0,
            "total_frames": r.get("total_frames", 0) or 0,
            "total_datasets": r.get("total_datasets", 0) or 0,
        })
    return series


def daily_group_series(history, key_fn):
    """Per-group daily positive growth from the last snapshot of each day.

    Returns rows sorted by date oldest-first and hours descending within each day:
    {date, group, hours, episodes, datasets}. The first detailed day counts
    each dataset's full duration as that day's contribution. If the previous day
    has only aggregate totals and no dataset details, attribution for the next
    day is skipped because per-group growth cannot be derived safely.
    """
    by_day = {}
    for r in history:
        by_day[r.get("date", "")] = r
    rows = []
    prev_report = None
    prev = {}
    for date in sorted(k for k in by_day if k):
        report = by_day[date]
        datasets = report.get("datasets", []) or []
        aggregate_only_prior = bool(prev_report) and not prev
        if not aggregate_only_prior:
            groups = {}
            for dataset in datasets:
                name = dataset.get("dataset_name")
                if not name:
                    continue
                prior = prev.get(name)
                d_hours = round((dataset.get("duration_hours") or 0)
                                - (prior.get("duration_hours") or 0 if prior else 0), 3)
                d_episodes = (dataset.get("total_episodes") or 0) \
                    - (prior.get("total_episodes") or 0 if prior else 0)
                hours = max(0, d_hours)
                episodes = max(0, d_episodes)
                if hours <= 0 and episodes <= 0:
                    continue
                key = key_fn(dataset) or "—"
                group = groups.setdefault(
                    key, {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
                group["hours"] += hours
                group["episodes"] += episodes
                group["datasets"] += 1
            day_rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
            for row in day_rows:
                row["hours"] = round(row["hours"], 3)
            rows.extend(day_rows)
        prev_report = report
        prev = {d.get("dataset_name"): d for d in datasets if d.get("dataset_name")}
    return rows


def daily_uploader_series(history):
    """Backward-compatible per-uploader daily growth helper."""
    return daily_group_series(history, lambda d: d.get("uploader") or "")


def _hf_update_date(dataset):
    """YYMMDD from a dataset's Hugging Face last_modified timestamp."""
    value = dataset.get("last_modified")
    if not value:
        return ""
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%y%m%d")
    except (TypeError, ValueError):
        return str(value)[:10].replace("-", "")[2:]


def hf_daily_group_series(datasets, key_fn):
    """Group current datasets by Hugging Face update day and dimension.

    Uses each dataset's `last_modified` from the Hub rather than local pull
    snapshots. Returns {date, group, hours, episodes, datasets} sorted by date
    oldest-first and hours descending within each date.
    """
    groups = {}
    for dataset in datasets or []:
        date = _hf_update_date(dataset)
        if not date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
        group["hours"] += dataset.get("duration_hours") or 0
        group["episodes"] += dataset.get("total_episodes") or 0
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 3)
    rows.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return rows


def hf_latest_update_date(datasets):
    """Newest Hugging Face last_modified date (YYMMDD) in current datasets."""
    dates = [_hf_update_date(dataset) for dataset in datasets or []]
    return max((date for date in dates if date), default="")


def hf_update_totals(datasets, date=None):
    """Totals for datasets whose Hugging Face update day equals `date`.

    If date is omitted, uses the newest HF update day in the dataset list.
    """
    date = date or hf_latest_update_date(datasets)
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    if not date:
        return totals
    for dataset in datasets or []:
        if _hf_update_date(dataset) != date:
            continue
        totals["hours"] += dataset.get("duration_hours") or 0
        totals["episodes"] += dataset.get("total_episodes") or 0
        totals["datasets"] += 1
    totals["hours"] = round(totals["hours"], 2)
    return totals


def hf_update_group_totals(datasets, key_fn, date=None):
    """Group totals for one Hugging Face update day."""
    date = date or hf_latest_update_date(datasets)
    groups = {}
    if not date:
        return []
    for dataset in datasets or []:
        if _hf_update_date(dataset) != date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            key, {"date": date, "group": key, "hours": 0.0,
                  "episodes": 0, "datasets": 0})
        group["hours"] += dataset.get("duration_hours") or 0
        group["episodes"] += dataset.get("total_episodes") or 0
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 2)
    rows.sort(key=lambda row: row["hours"], reverse=True)
    return rows


def hf_daily_group_delta_series(current_report, history, key_fn):
    """Incremental HF update rows grouped by last_modified day and dimension.

    Existing datasets contribute only their positive growth versus the previous
    detailed baseline snapshot. New datasets contribute their full current size.
    If no detailed baseline exists, returns [] rather than treating old data as
    new, because the true increment cannot be known safely.
    """
    prior = find_baseline(current_report, history)
    if not prior or not prior.get("datasets"):
        return []
    deltas = compute_deltas(current_report, history)
    groups = {}
    for dataset in current_report.get("datasets", []) if current_report else []:
        name = dataset.get("dataset_name")
        if not name:
            continue
        delta = deltas.get(name, {})
        hours = max(0, delta.get("d_hours", 0) or 0)
        episodes = max(0, delta.get("d_episodes", 0) or 0)
        if hours <= 0 and episodes <= 0:
            continue
        date = _hf_update_date(dataset)
        if not date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
        group["hours"] += hours
        group["episodes"] += episodes
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 3)
    rows.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return rows


def hf_update_delta_totals(current_report, history, date=None):
    """Incremental totals for one HF update day; defaults to newest delta day."""
    rows = hf_daily_group_delta_series(current_report, history, lambda dataset: "__all__")
    date = date or max((row["date"] for row in rows), default="")
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    for row in rows:
        if row["date"] != date:
            continue
        totals["hours"] += row["hours"]
        totals["episodes"] += row["episodes"]
        totals["datasets"] += row["datasets"]
    totals["hours"] = round(totals["hours"], 2)
    return totals


def hf_update_delta_group_totals(current_report, history, key_fn, date=None):
    """Incremental group totals for one HF update day."""
    rows = hf_daily_group_delta_series(current_report, history, key_fn)
    date = date or max((row["date"] for row in rows), default="")
    out = [row for row in rows if row["date"] == date]
    out.sort(key=lambda row: row["hours"], reverse=True)
    return out


def _commit_value(commit, *names):
    for name in names:
        if isinstance(commit, dict) and commit.get(name) is not None:
            return commit.get(name)
        if hasattr(commit, name):
            value = getattr(commit, name)
            if value is not None:
                return value
    return None


def _commit_created_at(commit):
    value = _commit_value(commit, "created_at", "createdAt", "date")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if value else ""


def _yymmdd_from_iso(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%y%m%d")
    except (TypeError, ValueError):
        text = str(value or "")[:10]
        return text.replace("-", "")[2:] if len(text) >= 10 else ""


def _commit_id(commit):
    return _commit_value(commit, "commit_id", "commitId", "oid", "id") or ""


def _summary_from_info(repo_id, info):
    summary = {"dataset_name": repo_id, "link": HF_DATASET_URL.format(repo_id=repo_id)}
    _apply_info(summary, info or {})
    return summary


def _fetch_info_at_revision(repo_id, revision, token=None):
    from huggingface_hub import hf_hub_download

    path = _call_hub_with_fallback(
        lambda endpoint: hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
            revision=revision,
            token=token,
            **({"endpoint": endpoint} if endpoint else {}),
        ),
        label=f"读取 {repo_id}@{revision} info",
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_meta_for_change(dataset):
    return {
        "dataset_name": dataset.get("dataset_name"),
        "uploader": dataset.get("uploader"),
        "robot_type": dataset.get("robot_type"),
        "total_tasks": dataset.get("total_tasks"),
        "last_modified": dataset.get("last_modified"),
    }


def build_hf_change_rows(repo_id, dataset, token=None):
    """Build true HF data-growth rows from commit history and meta/info.json diffs."""
    from huggingface_hub import HfApi

    def list_commits(endpoint):
        api = HfApi(**({"endpoint": endpoint} if endpoint else {}))
        try:
            return api.list_repo_commits(
                repo_id=repo_id, repo_type="dataset", token=token)
        except TypeError:
            # Keep compatibility with older huggingface_hub releases that did
            # not expose `token` on this method.
            return api.list_repo_commits(
                repo_id=repo_id, repo_type="dataset")

    commits = _call_hub_with_fallback(
        list_commits,
        label=f"读取 {repo_id} commits",
    )
    commits = sorted(commits, key=lambda commit: _commit_created_at(commit) or "")
    prev = None
    rows = []
    meta = _dataset_meta_for_change(dataset)
    for commit in commits:
        commit_id = _commit_id(commit)
        created_at = _commit_created_at(commit)
        if not commit_id or not created_at:
            continue
        try:
            summary = _summary_from_info(repo_id, _fetch_info_at_revision(repo_id, commit_id, token))
        except Exception:
            continue
        frames = summary.get("total_frames") or 0
        episodes = summary.get("total_episodes") or 0
        hours = summary.get("duration_hours") or 0
        prev_frames = prev.get("total_frames") or 0 if prev else 0
        prev_episodes = prev.get("total_episodes") or 0 if prev else 0
        prev_hours = prev.get("duration_hours") or 0 if prev else 0
        d_frames = frames - prev_frames
        d_episodes = episodes - prev_episodes
        d_hours = round(hours - prev_hours, 3)
        prev = summary
        if d_frames <= 0 and d_episodes <= 0 and d_hours <= 0:
            continue
        date = _yymmdd_from_iso(created_at)
        if not date:
            continue
        row = {
            **meta,
            "date": date,
            "commit_id": commit_id,
            "created_at": created_at,
            "hours": max(0, d_hours),
            "episodes": max(0, d_episodes),
            "frames": max(0, d_frames),
        }
        rows.append(row)
    return rows, commits[-1] if commits else None


def update_hf_change_history(report, token=None, path=CHANGE_HISTORY_FILE, log=None, progress=None):
    """Update local HF commit-diff cache for datasets in `report`."""
    cache = load_hf_change_history(path)
    repos = cache.setdefault("repos", {})
    datasets = report.get("datasets", []) if report else []
    total = len(datasets)
    for index, dataset in enumerate(datasets, 1):
        repo_id = dataset.get("dataset_name")
        if not repo_id:
            continue
        cached = repos.get(repo_id, {})
        if cached.get("last_modified") == dataset.get("last_modified") and cached.get("changes"):
            if progress:
                progress(index, total)
            continue
        if log:
            log(f"HF 变更历史 [{index}/{total}] {repo_id}")
        try:
            changes, head = build_hf_change_rows(repo_id, dataset, token)
        except Exception as exc:
            cached["error"] = str(exc)
            repos[repo_id] = cached
            if progress:
                progress(index, total)
            continue
        repos[repo_id] = {
            "dataset_name": repo_id,
            "last_modified": dataset.get("last_modified"),
            "head_commit": _commit_id(head) if head else "",
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "changes": changes,
        }
        if progress:
            progress(index, total)
    save_hf_change_history(cache, path)
    return cache


def hf_change_rows(change_history):
    rows = []
    for repo in (change_history or {}).get("repos", {}).values():
        rows.extend(repo.get("changes", []) or [])
    rows.sort(key=lambda row: (row.get("date") or "", row.get("created_at") or ""))
    return rows


def hf_change_daily_group_series(change_history, key_fn):
    groups = {}
    for row in hf_change_rows(change_history):
        date = row.get("date") or ""
        if not date:
            continue
        key = key_fn(row) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0, "_dataset_names": set()})
        group["hours"] += row.get("hours") or 0
        group["episodes"] += row.get("episodes") or 0
        group["_dataset_names"].add(row.get("dataset_name") or row.get("commit_id") or id(row))
    out = list(groups.values())
    for row in out:
        row["hours"] = round(row["hours"], 3)
        row["datasets"] = len(row.pop("_dataset_names"))
    out.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return out


def hf_change_latest_date(change_history):
    return max((row.get("date") for row in hf_change_rows(change_history) if row.get("date")), default="")


def hf_change_totals(change_history, date=None):
    date = date or hf_change_latest_date(change_history)
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    dataset_names = set()
    for row in hf_change_rows(change_history):
        if row.get("date") != date:
            continue
        totals["hours"] += row.get("hours") or 0
        totals["episodes"] += row.get("episodes") or 0
        dataset_names.add(row.get("dataset_name") or row.get("commit_id") or id(row))
    totals["hours"] = round(totals["hours"], 2)
    totals["datasets"] = len(dataset_names)
    return totals


def hf_change_group_totals(change_history, key_fn, date=None):
    date = date or hf_change_latest_date(change_history)
    rows = [row for row in hf_change_daily_group_series(change_history, key_fn)
            if row.get("date") == date]
    rows.sort(key=lambda row: row["hours"], reverse=True)
    return rows


def _repo_change_rows(change_history, repo_id):
    repo = (change_history or {}).get("repos", {}).get(repo_id, {})
    return repo.get("changes", []) or []


def _repo_cache_matches(change_history, dataset):
    repo_id = dataset.get("dataset_name")
    repo = (change_history or {}).get("repos", {}).get(repo_id, {})
    return bool(repo.get("changes")) and repo.get("last_modified") == dataset.get("last_modified")


def hf_report_has_matching_change_cache(current_report, change_history):
    """Whether a current dataset has a matching cache row for its update day."""
    for dataset in (current_report or {}).get("datasets", []):
        if not _repo_cache_matches(change_history, dataset):
            continue
        date = _hf_update_date(dataset) or (current_report or {}).get("date")
        if any(row.get("date") == date
               for row in _repo_change_rows(
                   change_history, dataset.get("dataset_name"))):
            return True
    return False


def hf_last_modified_dataset_deltas(current_report, history, change_history):
    """Per-dataset additions with HF last_modified as the date authority.

    Priority for numeric deltas:
    1. Matching HF commit-diff cache for this dataset and last_modified date.
    2. Local snapshot delta for datasets not covered by the HF cache.

    Local snapshots are therefore only a fallback, never the primary source when
    a matching HF commit cache exists.
    """
    if not current_report:
        return {}
    local_deltas = compute_deltas(current_report, history)
    out = {}
    for dataset in current_report.get("datasets", []) or []:
        name = dataset.get("dataset_name")
        if not name:
            continue
        date = _hf_update_date(dataset) or current_report.get("date") or ""
        if _repo_cache_matches(change_history, dataset):
            rows = [row for row in _repo_change_rows(change_history, name)
                    if row.get("date") == date]
        else:
            rows = []
        if rows:
            hours = round(sum(row.get("hours") or 0 for row in rows), 3)
            episodes = sum(row.get("episodes") or 0 for row in rows)
            frames = sum(row.get("frames") or 0 for row in rows)
        else:
            delta = local_deltas.get(name, {})
            hours = max(0, delta.get("d_hours", 0) or 0)
            episodes = max(0, delta.get("d_episodes", 0) or 0)
            frames = max(0, delta.get("d_frames", 0) or 0)
        out[name] = {
            "date": date,
            "d_hours": round(hours, 3),
            "d_episodes": episodes,
            "d_frames": frames,
            "is_new": bool(local_deltas.get(name, {}).get("is_new")),
        }
    return out


def hf_last_modified_daily_group_series(current_report, history, change_history, key_fn):
    """Daily group additions using HF last_modified for date attribution."""
    deltas = hf_last_modified_dataset_deltas(current_report, history, change_history)
    by_name = {dataset.get("dataset_name"): dataset
               for dataset in (current_report or {}).get("datasets", [])}
    groups = {}
    for name, delta in deltas.items():
        hours = max(0, delta.get("d_hours", 0) or 0)
        episodes = max(0, delta.get("d_episodes", 0) or 0)
        frames = max(0, delta.get("d_frames", 0) or 0)
        if hours <= 0 and episodes <= 0 and frames <= 0:
            continue
        dataset = by_name.get(name, {})
        date = delta.get("date") or ""
        if not date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
        group["hours"] += hours
        group["episodes"] += episodes
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 3)
    rows.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return rows


def hf_last_modified_latest_date(current_report):
    datasets = (current_report or {}).get("datasets", [])
    hf_date = max(
        (_hf_update_date(dataset) for dataset in datasets if _hf_update_date(dataset)),
        default="")
    return hf_date or (current_report or {}).get("date") or ""


def hf_last_modified_totals(current_report, history, change_history, date=None):
    date = date or hf_last_modified_latest_date(current_report)
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    if not current_report or not date:
        return totals
    if current_report.get("source") == "manual":
        totals["hours"], totals["episodes"] = aggregate_deltas(
            current_report, history)
        base = find_baseline(current_report, history)
        totals["datasets"] = max(
            0,
            (current_report.get("total_datasets") or 0)
            - ((base.get("total_datasets") or 0) if base else 0),
        )
        return totals
    rows = hf_last_modified_daily_group_series(
        current_report, history, change_history, lambda dataset: "__all__")
    for row in rows:
        if row.get("date") != date:
            continue
        totals["hours"] += row.get("hours") or 0
        totals["episodes"] += row.get("episodes") or 0
        totals["datasets"] += row.get("datasets") or 0
    totals["hours"] = round(totals["hours"], 2)
    return totals


def hf_last_modified_group_totals(current_report, history, change_history, key_fn, date=None):
    date = date or hf_last_modified_latest_date(current_report)
    rows = [row for row in hf_last_modified_daily_group_series(
        current_report, history, change_history, key_fn) if row.get("date") == date]
    rows.sort(key=lambda row: row["hours"], reverse=True)
    return rows


def _date_in_range(date, date_from=None, date_to=None):
    if date_from and date < date_from:
        return False
    if date_to and date > date_to:
        return False
    return True


def hf_last_modified_group_range_totals(
        current_report, history, change_history, key_fn,
        date_from=None, date_to=None):
    rows = [
        row for row in hf_last_modified_daily_group_series(
            current_report, history, change_history, key_fn)
        if _date_in_range(row.get("date") or "", date_from, date_to)
    ]
    groups = {}
    for row in rows:
        key = row.get("group") or "—"
        group = groups.setdefault(
            key, {"group": key, "hours": 0.0, "episodes": 0, "datasets": 0})
        group["hours"] += row.get("hours") or 0
        group["episodes"] += row.get("episodes") or 0
        group["datasets"] += row.get("datasets") or 0
    out = list(groups.values())
    for row in out:
        row["hours"] = round(row["hours"], 3)
    out.sort(key=lambda row: row["hours"], reverse=True)
    return out


def hf_last_modified_range_totals(
        current_report, history, change_history, date_from=None, date_to=None):
    rows = hf_last_modified_daily_group_series(
        current_report, history, change_history, lambda dataset: "__all__")
    totals = {"date_from": date_from or "", "date_to": date_to or "",
              "hours": 0.0, "episodes": 0, "datasets": 0}
    for row in rows:
        if not _date_in_range(row.get("date") or "", date_from, date_to):
            continue
        totals["hours"] += row.get("hours") or 0
        totals["episodes"] += row.get("episodes") or 0
        totals["datasets"] += row.get("datasets") or 0
    totals["hours"] = round(totals["hours"], 2)
    return totals


def find_baseline(current_report, history):
    """Return the snapshot to diff `current_report` against: the last pull of the
    most recent *earlier day*.

    "今日新增" is measured against the previous pull DAY, not merely the previous
    pull — so multiple pulls on the same day all compare back to that earlier day.
    `history` is oldest-first, so the last entry whose date precedes the current
    report's date is that day's final pull. Returns None when no earlier day
    exists (the current report is the first ever).
    """
    cur_date = current_report.get("date") or ""
    cur_org = current_report.get("org")
    prior = None
    for r in history:  # oldest-first; last match = newest earlier-day pull
        rd = r.get("date") or ""
        if cur_org is not None and r.get("org") not in (None, cur_org):
            continue
        if rd and rd < cur_date:
            prior = r
    return prior


def aggregate_deltas(current_report, history):
    """Return aggregate (new_hours, new_episodes) vs the prior recorded day."""
    base = find_baseline(current_report, history)
    base_hours = (base.get("total_hours") or 0) if base else 0
    base_eps = (base.get("total_episodes") or 0) if base else 0
    new_hours = round((current_report.get("total_hours") or 0) - base_hours, 2)
    new_episodes = (current_report.get("total_episodes") or 0) - base_eps
    return new_hours, new_episodes


def compute_deltas(current_report, history):
    """Per-dataset growth of current_report vs the previous pull day's snapshot.

    Baseline = find_baseline(current_report, history). Returns
    {dataset_name: {d_episodes, d_frames, d_hours, is_new}}. With no earlier-day
    snapshot every dataset is marked is_new with its full totals as the delta.
    """
    prior = find_baseline(current_report, history)
    prev = {d["dataset_name"]: d for d in (prior.get("datasets", []) if prior else [])}
    # An aggregate-only baseline (backfilled history that has totals but no
    # per-dataset detail) can't attribute growth to individual datasets. Report
    # zero per-dataset deltas there instead of pretending everything is new; the
    # KPI falls back to the aggregate total difference (see main_app._new_totals).
    aggregate_only = bool(prior) and not prev
    deltas = {}
    for d in current_report.get("datasets", []):
        name = d["dataset_name"]
        if aggregate_only:
            deltas[name] = {"d_episodes": 0, "d_frames": 0, "d_hours": 0,
                            "is_new": False}
            continue
        p = prev.get(name)
        deltas[name] = {
            "d_episodes": (d.get("total_episodes") or 0) - (p.get("total_episodes") or 0 if p else 0),
            "d_frames": (d.get("total_frames") or 0) - (p.get("total_frames") or 0 if p else 0),
            "d_hours": round((d.get("duration_hours") or 0) - (p.get("duration_hours") or 0 if p else 0), 3),
            "is_new": p is None,
        }
    return deltas


def task_prefix(dataset_name):
    """Derive a task label: drop the owner and a trailing -MMDD date suffix.

    e.g. 'TacVerse/taccap-g1-pepper-0703' -> 'taccap-g1-pepper'.
    """
    name = dataset_name.split("/")[-1]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return name


def rollup(datasets, key_fn):
    """Group datasets by key_fn(d) and sum count/episodes/frames/hours.

    Returns a list of {group, count, episodes, frames, hours, pct_hours} sorted
    by hours descending. pct_hours is each group's share of total hours.
    """
    groups = {}
    for d in datasets:
        key = key_fn(d) or "—"
        g = groups.setdefault(
            key, {"group": key, "count": 0, "episodes": 0, "frames": 0, "hours": 0.0})
        g["count"] += 1
        g["episodes"] += d.get("total_episodes") or 0
        g["frames"] += d.get("total_frames") or 0
        g["hours"] += d.get("duration_hours") or 0
    total_hours = sum(g["hours"] for g in groups.values()) or 1
    rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
    for g in rows:
        g["hours"] = round(g["hours"], 3)
        g["pct_hours"] = round(100 * g["hours"] / total_hours, 1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull HF datasets into one organization dataset folder."
    )
    parser.add_argument(
        "--org",
        default=ORG,
        help=f"Pull every dataset under this org/user (default: {ORG})",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        metavar="OWNER/NAME",
        help="Pull only these datasets (repeat); overrides --org discovery",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/TacVerse",
        help="Organization dataset directory; datasets are stored directly inside",
    )
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token (defaults to $HF_TOKEN or the cached login token)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Download only; skip writing the summary file",
    )
    parser.add_argument(
        "--migrate-log",
        action="store_true",
        help="One-time: convert config.json's legacy pull_history into "
             "dataset_log.json and drop pull_history from config.json, then exit",
    )
    args = parser.parse_args()

    if args.migrate_log:
        path = migrate_pull_history_to_log()
        print(f"Migrated pull_history -> {path}; removed pull_history from {CONFIG_FILE}")
        return 0

    normalize_proxy_env()

    try:
        import huggingface_hub  # noqa: F401  (imported for the clear error below)
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    meta_map = None
    if args.repo_ids:
        repo_ids = args.repo_ids
    else:
        print(f"Discovering datasets under '{args.org}' ...")
        meta = discover_datasets_meta(args.org, args.token)
        repo_ids = [m["id"] for m in meta]
        meta_map = {m["id"]: m["last_modified"] for m in meta}
        print(f"Found {len(repo_ids)} datasets.")
    if not repo_ids:
        sys.exit(f"No datasets to pull (org '{args.org}' returned nothing).")

    report, _ = run_pull(
        repo_ids,
        out_dir=args.out_dir,
        org=args.org,
        revision=args.revision,
        token=args.token,
        write_summary=not args.no_summary,
        meta_map=meta_map,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
