"""Manage the vendored xense_lerobot_viewer as a black-box web service.

Workbench talks to the viewer ONLY through its three stable contracts, so the
submodule stays read-only and only acts as a source reference:

  ① LOCAL_DATASET_ROOT env  → the shared dataset root the viewer scans
  ② HTTP on PORT            → home `/` and `/_local/<encodedPath>` deep links
  ③ side-effect files       → meta/xense_tags.json, meta/lerobot_annotations.json

This module owns the viewer runtime copy, subprocess lifecycle (start / health /
stop) and deep-link URLs. It is Qt-free so it can be unit-tested and reused.
"""

import base64
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER_SOURCE_DIR = ROOT / "third_party" / "lerobot_viewer"
VIEWER_RUNTIME_DIR = ROOT / "vendor" / "lerobot_viewer_runtime"
DEFAULT_PORT = 3000


def _path_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def encode_dataset_path(rel_path: str) -> str:
    """base64url(rel_path) without padding.

    Mirrors the viewer's encodeLocalDatasetPath so `/_local/<enc>` resolves to
    the same dataset the viewer discovered.
    """
    rel = rel_path.replace("\\", "/").strip("/")
    raw = base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def find_bun():
    """Locate the bun executable (PATH or the default ~/.bun install)."""
    found = shutil.which("bun")
    if found:
        return found
    bin_dir = Path.home() / ".bun" / "bin"
    candidates = [bin_dir / "bun.exe", bin_dir / "bun"] if os.name == "nt" \
        else [bin_dir / "bun"]
    return next((str(path) for path in candidates if path.is_file()), None)


def find_js_runner():
    """Return a package runner for the viewer, preferring Bun when installed."""
    bun = find_bun()
    if bun:
        return [bun, "run", "dev"], "bun"
    npm = shutil.which("npm")
    node_major, _ = node_version()
    if npm and node_major is not None and node_major >= 20:
        return [npm, "run", "dev"], "npm"
    return None, None


def node_version():
    node = shutil.which("node")
    if not node:
        return None, None
    try:
        out = subprocess.check_output(
            [node, "--version"], text=True, timeout=3).strip()
    except Exception:
        return None, None
    version = out.lstrip("v")
    major = version.split(".", 1)[0]
    return (int(major), version) if major.isdigit() else (None, version)


def install_hint(viewer_dir):
    viewer_dir = Path(viewer_dir)
    if _path_within(viewer_dir, VIEWER_SOURCE_DIR):
        return (
            "请执行 python scripts/install_viewer.py 使用 "
            f"{VIEWER_RUNTIME_DIR}；submodule 仅保留为源码引用。")
    if not (viewer_dir / "package.json").is_file():
        return (
            "请先执行 python scripts/install_viewer.py "
            "生成运行时 viewer；submodule 仅保留为源码引用。")
    if find_bun():
        return f"请在 {viewer_dir} 执行 bun install"
    node_major, node_text = node_version()
    if shutil.which("npm") and node_major is not None and node_major >= 20:
        return f"请在 {viewer_dir} 执行 npm install --no-package-lock"
    if shutil.which("npm") and node_text:
        return (
            f"当前 Node.js {node_text} 过旧；请安装 Bun，或升级 Node.js >=20 "
            f"后在 {viewer_dir} 执行 npm install --no-package-lock")
    return "请先安装 Bun，或安装 Node.js >=20/npm 后再安装 viewer 依赖"


def prepare_viewer_runtime(source_dir=VIEWER_SOURCE_DIR,
                           runtime_dir=VIEWER_RUNTIME_DIR):
    """Materialize a writable viewer runtime copy from the read-only submodule."""
    source_dir = Path(source_dir)
    runtime_dir = Path(runtime_dir)
    if _path_within(runtime_dir, source_dir):
        raise ValueError(
            "viewer runtime directory must be outside the viewer submodule")
    if not (source_dir / "package.json").is_file():
        raise FileNotFoundError(
            f"viewer source not found: {source_dir} "
            "(run git submodule update --init --recursive)")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_dir, runtime_dir,
        ignore=shutil.ignore_patterns(
            ".git", ".gitmodules", "node_modules", ".next", "out", "dist",
            "*.log", "*.tsbuildinfo",
        ),
    )
    return runtime_dir


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _http_ok(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


_TRANSIENT_HTTP_MARKERS = (
    "incomplete chunked read",
    "peer closed connection without sending complete message body",
    "incomplete read",
    "response ended prematurely",
    "remote end closed connection without response",
    "connection reset by peer",
    "connection aborted",
)


def _related_exceptions(exc):
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


def _is_transient_http_error(exc):
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
        if any(marker in text for marker in _TRANSIENT_HTTP_MARKERS):
            return True
    return False


def _retry_transient_http(call, *, attempts=3, delay=0.4):
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_http_error(exc):
                raise
            time.sleep(delay * attempt)


def _same_path(left, right):
    """Compare resolved paths with the host platform's case rules."""
    if not left or not right:
        return False
    return os.path.normcase(str(Path(left).resolve())) == \
        os.path.normcase(str(Path(right).resolve()))


def _listener_pids(port):
    """Return PIDs with a TCP LISTEN socket on `port` (best effort)."""
    port = int(port)
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
                creationflags=flags, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []

        pids = set()
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0].upper() != "TCP":
                continue
            if fields[-2].upper() != "LISTENING":
                continue
            if fields[1].rsplit(":", 1)[-1] != str(port):
                continue
            try:
                pids.add(int(fields[-1]))
            except ValueError:
                pass
        return sorted(pids)

    pids = set()
    try:
        import psutil
    except Exception:
        psutil = None

    if psutil is not None:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (conn.status == psutil.CONN_LISTEN and conn.laddr
                        and conn.laddr.port == port and conn.pid):
                    pids.add(conn.pid)
        except Exception:
            pass
        if pids:
            return sorted(pids)

    for args in (
        ("lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"),
        ("fuser", "-n", "tcp", str(port)),
    ):
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=2)
        except Exception:
            continue
        for token in out.stdout.split():
            candidate = token.split("/")[-1]
            if candidate.isdigit():
                pids.add(int(candidate))
    return sorted(pids)


def _process_environ(pid):
    """Return a process environment dict, or {} when it cannot be read."""
    try:
        import psutil
        return dict(psutil.Process(int(pid)).environ())
    except Exception:
        pass
    env = {}
    try:
        raw = Path(f"/proc/{int(pid)}/environ").read_bytes()
    except OSError:
        return env
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _listener_root(port):
    """LOCAL_DATASET_ROOT of the process listening on `port`, or None."""
    for pid in _listener_pids(port):
        root = _process_environ(pid).get("LOCAL_DATASET_ROOT")
        if root:
            try:
                return str(Path(root).resolve())
            except OSError:
                return str(root)
    return None


def _kill_pid(pid, sig):
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _windows_process_table():
    """Return a minimal Windows process table keyed by PID."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=10,
            creationflags=flags, check=False,
        )
        rows = json.loads(result.stdout.lstrip("\ufeff") or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    table = {}
    for row in rows:
        try:
            table[int(row["ProcessId"])] = row
        except (KeyError, TypeError, ValueError):
            pass
    return table


class ViewerService:
    """Supervises one viewer dev server process bound to a dataset root."""

    def __init__(self, viewer_dir=VIEWER_RUNTIME_DIR, source_dir=VIEWER_SOURCE_DIR,
                 port=DEFAULT_PORT):
        self.viewer_dir = Path(viewer_dir)
        self.source_dir = Path(source_dir)
        self.port = int(port)
        self.root = None
        self.proc = None
        self._log_path = None
        self._attached = False
        self._last_error = None
        self._info_cache = None
        self._info_cache_at = 0.0
        state_name = f"tacverse-viewer-{self.port}.json"
        self._state_path = Path(tempfile.gettempdir()) / state_name

    # --- URLs (contract ②) ------------------------------------------------
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def home_url(self):
        return self.base_url + "/"

    def dataset_url(self, rel_path, episode=None):
        url = f"{self.base_url}/_local/{encode_dataset_path(rel_path)}"
        if episode is not None:
            url += f"/{episode}"
        return url

    def dataset_rel_path(self, dataset, root=None):
        """Path of `dataset` relative to the dataset root, or None if not local.

        Uses the record's local_dir when present, else globs the root for a
        matching leaf directory (the stats-only case). Returns None when the
        dataset isn't under the root (so the caller can disable the open link).
        """
        root = Path(root or self.root or "").resolve()
        if not root:
            return None
        local_dir = (dataset or {}).get("local_dir")
        if local_dir:
            p = Path(local_dir).resolve()
            try:
                rel = p.relative_to(root)
                return str(rel)
            except ValueError:
                pass
        leaf = ((dataset or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        matches = [d for d in root.glob(f"*/{leaf}") if (d / "meta").is_dir()]
        matches += [d for d in root.glob(leaf) if (d / "meta").is_dir()]
        if not matches:
            return None
        best = max(matches, key=lambda d: d.stat().st_mtime)
        return str(best.relative_to(root))

    # --- lifecycle (contract ①②) -----------------------------------------
    def available(self):
        """viewer directory, installed deps, and a usable JS runner present?"""
        if _path_within(self.viewer_dir, self.source_dir):
            return False
        cmd, _ = find_js_runner()
        return bool(cmd) and (self.viewer_dir / "package.json").is_file() and \
               (self.viewer_dir / "node_modules").is_dir()

    def _invalidate_info(self):
        self._info_cache = None
        self._info_cache_at = 0.0

    def service_info(self, timeout=1.5, max_age=1.0):
        """Return the Viewer discovery response, or None for another service."""
        now = time.monotonic()
        if self._info_cache is not None and now - self._info_cache_at <= max_age:
            return self._info_cache
        try:
            def read_info():
                with urllib.request.urlopen(
                        self.base_url + "/api/local-datasets", timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            data = _retry_transient_http(read_info)
            if not isinstance(data, dict) or not isinstance(data.get("root"), str):
                return None
            if not isinstance(data.get("datasets"), list):
                return None
        except Exception:
            return None
        self._info_cache = data
        self._info_cache_at = now
        return data

    def _root_matches(self, info=None):
        info = info if info is not None else self.service_info()
        return bool(info and _same_path(info.get("root"), self.root))

    def is_running(self):
        return self._root_matches()

    def is_ready(self):
        # The discovery endpoint proves both that this is our Viewer and that
        # it has completed startup with the expected dataset root.
        return self._root_matches()

    def _write_state(self):
        if not self.proc:
            return
        try:
            self._state_path.write_text(json.dumps({
                "pid": self.proc.pid,
                "root": self.root,
                "viewer_dir": str(self.viewer_dir.resolve()),
            }), encoding="utf-8")
        except OSError:
            pass

    def _state_pid(self, expected_root=None):
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            root = self.root if expected_root is None else expected_root
            if _same_path(state.get("root"), root) and \
                    _same_path(state.get("viewer_dir"), self.viewer_dir):
                return int(state["pid"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return None

    @staticmethod
    def _kill_windows_tree(pid):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, creationflags=flags, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _windows_viewer_tree_roots(self, listener_pids):
        """Find the highest vendored Viewer ancestor for each listener PID."""
        table = _windows_process_table()
        viewer_marker = os.path.normcase(str(self.viewer_dir.resolve()))
        roots = set()
        for listener_pid in listener_pids:
            current = listener_pid
            highest = listener_pid
            visited = set()
            while current in table and current not in visited:
                visited.add(current)
                row = table[current]
                name = str(row.get("Name") or "").lower()
                command = os.path.normcase(str(row.get("CommandLine") or ""))
                if name not in {"node.exe", "node", "bun.exe", "bun"} or \
                        viewer_marker not in command:
                    break
                highest = current
                try:
                    current = int(row.get("ParentProcessId") or 0)
                except (TypeError, ValueError):
                    break
            roots.add(highest)
        return sorted(roots)

    def _stop_processes(self, actual_root=None):
        """Stop the launch process and any server still listening on our port."""
        parent_pid = self.proc.pid if self.proc else (
            self._state_pid(actual_root) if _port_in_use(self.port) else None)
        if os.name == "nt":
            if parent_pid:
                self._kill_windows_tree(parent_pid)
            listener_pids = _listener_pids(self.port)
            for pid in self._windows_viewer_tree_roots(listener_pids):
                self._kill_windows_tree(pid)
        elif self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    pass
        elif parent_pid:
            try:
                os.kill(parent_pid, signal.SIGTERM)
            except OSError:
                pass
        else:
            for pid in _listener_pids(self.port):
                _kill_pid(pid, signal.SIGTERM)

        deadline = time.monotonic() + 5
        while _port_in_use(self.port) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _port_in_use(self.port) and os.name != "nt":
            for pid in _listener_pids(self.port):
                _kill_pid(pid, signal.SIGKILL)
            time.sleep(0.3)

    def start(self, root, wait=False, timeout=60, log_path=None):
        """Launch (or reuse) the viewer bound to `root`. Returns (ok, message).

        `wait=False` returns as soon as the process is spawned; poll status()
        for readiness. `wait=True` blocks until the discovery API answers or
        times out. A compatible Viewer on this dedicated port is reused when it
        has the requested root, or replaced when it has a stale root.
        """
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = str(root_path)
        self._last_error = None
        self._invalidate_info()

        if self.proc and self.proc.poll() is None:
            info = self.service_info(max_age=0)
            if not info or self._root_matches(info):
                if wait:
                    return self._wait_until_ready(timeout)
                return True, "启动中…" if not info else \
                    f"已在运行 · {self.base_url}"
            self._stop_processes(info.get("root"))
            self.proc = None
            self._invalidate_info()

        if _port_in_use(self.port):
            info = self.service_info(max_age=0)
            if not info:
                self._last_error = f"端口 {self.port} 已被其他程序占用"
                return False, self._last_error
            self._attached = True
            if self._root_matches(info):
                return True, f"已连接现有 Viewer · {self.base_url}"
            # Port 3001 is reserved for Workbench's Viewer. Replace a verified
            # Viewer left behind with an obsolete dataset root.
            old_root = info.get("root") or "未知"
            self._stop_processes(info.get("root"))
            self._attached = False
            self._invalidate_info()
            if _port_in_use(self.port):
                self._last_error = f"无法关闭旧 Viewer（数据根: {old_root}）"
                return False, self._last_error
        if not self.available():
            self._last_error = f"Viewer 未就绪：{install_hint(self.viewer_dir)}"
            return False, self._last_error
        cmd, runner = find_js_runner()
        if not cmd:
            self._last_error = "未找到 Bun 或 npm，无法启动 Viewer"
            return False, self._last_error

        env = dict(os.environ)
        env["LOCAL_DATASET_ROOT"] = self.root
        env["PORT"] = str(self.port)
        self._log_path = str(log_path or (
            Path(tempfile.gettempdir()) / f"tacverse-viewer-{self.port}.log"))
        out = open(self._log_path, "w", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.viewer_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
        except Exception as exc:
            self.proc = None
            self._last_error = f"启动失败: {exc}"
            return False, self._last_error
        finally:
            out.close()

        self._attached = False
        self._write_state()

        if not wait:
            return True, f"启动中…（{runner}）"
        return self._wait_until_ready(timeout)

    def _wait_until_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                self._last_error = f"Viewer 进程意外退出（日志: {self._log_path}）"
                return False, self._last_error
            self._invalidate_info()
            if self.is_ready():
                return True, f"运行中 · {self.base_url}"
            time.sleep(0.5)
        self._last_error = f"启动超时（{timeout}s，日志: {self._log_path}）"
        return False, self._last_error

    def stop(self):
        """Terminate this Workbench Viewer and its full process tree."""
        port_in_use = _port_in_use(self.port)
        info = self.service_info(max_age=0) if port_in_use else None
        if port_in_use and not info:
            return False, f"端口 {self.port} 不是可识别的 Viewer，未关闭外部服务"
        owns_process = bool(self.proc or self._attached or
                            (port_in_use and self._state_pid(info.get("root"))))
        if info and not self._root_matches(info):
            return False, "端口上的 Viewer 数据根不匹配，未关闭外部服务"
        if not owns_process:
            try:
                self._state_path.unlink(missing_ok=True)
            except OSError:
                pass
            return True, "Viewer 已停止"

        self._stop_processes(info.get("root") if info else self.root)
        self.proc = None
        self._attached = False
        self._invalidate_info()
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError:
            pass
        if _port_in_use(self.port):
            self._last_error = f"Viewer 未能完全停止，端口 {self.port} 仍被占用"
            return False, self._last_error
        self._last_error = None
        return True, "Viewer 已停止"

    # --- introspection (contract ②) --------------------------------------
    def dataset_count(self):
        """How many datasets the viewer currently sees (JSON API), or None."""
        data = self.service_info(timeout=3)
        return len(data["datasets"]) if data else None

    def report(self, rel_path, include=None, timeout=180):
        """Fetch the viewer's /report analysis JSON for a dataset.

        Returns (report_dict, None) on success, or (None, error_message). The
        analysis is computed server-side and can take tens of seconds — call
        this off the UI thread.
        """
        url = f"{self.base_url}/api/local-datasets/{encode_dataset_path(rel_path)}/report"
        if include:
            url += "?include=" + ",".join(include)
        try:
            def read_report():
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            data = _retry_transient_http(read_report)
        except Exception as exc:
            fallback = self._local_report_fallback(rel_path)
            if fallback is not None:
                return fallback, None
            return None, str(exc)
        if isinstance(data, dict) and data.get("ok") is False:
            fallback = self._local_report_fallback(rel_path)
            if fallback is not None:
                return fallback, None
            return None, str(data.get("error") or "analysis failed")
        return data, None

    def doctor(self, rel_path, max_episodes=25, episode_range=None,
               checks=None, timeout=300, on_progress=None):
        """Run the viewer's TypeScript Doctor endpoint.

        The endpoint streams newline-delimited JSON progress events followed by
        one result event.  This method stays Qt-free and reports progress via
        ``on_progress(progress_dict)`` so the GUI can run it in a worker thread.
        Returns ``(result_dict, None)`` on success or ``(None, error)``.
        """
        url = (f"{self.base_url}/api/local-datasets/"
               f"{encode_dataset_path(rel_path)}/doctor?stream=1")
        payload = {
            "maxEpisodes": max_episodes,
            "episodeRange": episode_range,
        }
        if checks is not None:
            payload["checks"] = checks
        try:
            def read_doctor():
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=timeout) as resp:
                    return self._read_doctor_stream(resp, on_progress)
            return _retry_transient_http(read_doctor)
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
                message = body.get("error") if isinstance(body, dict) else None
            except Exception:
                message = None
            return None, str(message or f"HTTP {exc.code}")
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _read_doctor_stream(resp, on_progress=None):
        result = None
        for raw_line in resp:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            kind = event.get("type") if isinstance(event, dict) else None
            if kind == "progress":
                if on_progress:
                    on_progress(event.get("progress") or {})
            elif kind == "result":
                result = event.get("result")
            elif kind == "error":
                return None, str(event.get("error") or "Doctor failed")
        if result is None:
            return None, "Doctor stream ended without a result"
        return result, None

    def _local_report_fallback(self, rel_path):
        """Build the minimal report shape the Qt panel expects from metadata.

        The vendored viewer may not expose a `/report` route in every upstream
        revision. Keep Workbench usable by falling back to local LeRobot metadata
        without modifying the viewer submodule.
        """
        if not self.root or not rel_path:
            return None
        root = Path(self.root).resolve()
        ds_dir = (root / rel_path).resolve()
        try:
            ds_dir.relative_to(root)
        except ValueError:
            return None
        info_path = ds_dir / "meta" / "info.json"
        if not info_path.is_file():
            return None
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        features = info.get("features") or {}
        cameras = [
            key for key, val in features.items()
            if (val or {}).get("dtype") == "video"
            or "image" in key
            or "camera" in key
        ]
        fps = info.get("fps") or 0
        total_episodes = info.get("total_episodes") or 0
        total_frames = info.get("total_frames") or 0
        mean_len = None
        if fps and total_episodes and total_frames:
            mean_len = round(total_frames / fps / total_episodes, 2)

        issues = []
        if not total_episodes:
            issues.append("meta/info.json 缺少 total_episodes")
        if not total_frames:
            issues.append("meta/info.json 缺少 total_frames")
        if not fps:
            issues.append("meta/info.json 缺少 fps")
        if not (ds_dir / "meta" / "stats.json").is_file():
            issues.append("meta/stats.json 不存在，深度质量分析不可用")

        quality = {
            "jerkyEpisodes": [],
            "lowMovementEpisodes": [],
            "smoothness": {
                "verdict": {"label": "N/A"},
                "lines": ["当前 viewer 版本未提供 /report 接口，已使用本地元数据摘要。"],
                "tip": "打开浏览器 Viewer 可查看该子模块内置的完整交互分析。",
            },
        }
        if mean_len is not None:
            quality["episodeLength"] = {
                "shortest": mean_len,
                "longest": mean_len,
                "mean": mean_len,
                "median": mean_len,
                "std": 0,
            }

        return {
            "ok": True,
            "dataset": {
                "name": rel_path,
                "total_episodes": total_episodes or None,
                "total_frames": total_frames or None,
                "fps": fps or None,
                "cameras": cameras,
                "robot_type": info.get("robot_type"),
            },
            "integrity": {
                "status": "warning" if issues else "ok",
                "issues": issues,
            },
            "quality": quality,
            "training": {},
            "meta": {
                "sampledEpisodes": total_episodes or None,
            },
        }

    def status(self):
        port_in_use = _port_in_use(self.port)
        info = self.service_info() if port_in_use else None
        ready = bool(info and self._root_matches(info))
        process_alive = bool(self.proc and self.proc.poll() is None)
        if self.proc and not process_alive and not port_in_use and not self._last_error:
            self._last_error = f"Viewer 进程已退出（日志: {self._log_path}）"
        if ready:
            state = "ready"
        elif process_alive:
            state = "starting"
        elif port_in_use:
            state = "conflict"
        elif self._last_error:
            state = "error"
        else:
            state = "stopped"
        return {
            "state": state,
            "running": ready,
            "ready": ready,
            "managed": bool(process_alive or self._attached),
            "port_in_use": port_in_use,
            "port": self.port,
            "root": self.root,
            "actual_root": info.get("root") if info else None,
            "url": self.base_url,
            "error": self._last_error,
            "log_path": self._log_path,
        }

    # --- convenience openers ----------------------------------------------
    def open_home(self):
        webbrowser.open(self.home_url())

    def open_dataset(self, rel_path, episode=None):
        webbrowser.open(self.dataset_url(rel_path, episode))
