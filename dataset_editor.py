"""In-place-semantics editing for pulled LeRobot datasets — the write side.

`tasks_reader.py` only *reads* a dataset's prompt; this module *edits* it and
writes a new copy. Two operations are supported today, both chosen because they
are pure file-tree work that never touches the heavy `data/`+`videos/` payload:

- **Edit prompt** — the natural-language task string lives only in
  `meta/tasks.parquet` (column `task`); every frame references it by an integer
  `task_index`. So changing the wording is: rewrite the `task` column in
  `meta/tasks.parquet` (keeping `task_index`) and the matching entries in the
  per-episode `tasks` list column of `meta/episodes/**/*.parquet`. No `data/`,
  no `videos/`, no `info.json` change (`total_tasks` is unchanged).
- **Rename / save-as** — a dataset's `repo_id` is not stored in any metadata; it
  is just the folder name. So renaming == writing the new copy under a new leaf.

Everything is done on a **new copy** (`generate new copy` mode): the heavy
`data/`+`videos/` trees are hard-linked (no disk doubling, and they are never
written to), while `meta/` and the small top-level files are real copies so the
prompt rewrite cannot touch the original.

Qt-free and pandas-free (pyarrow only, imported lazily), mirroring
`tasks_reader.py` so the app stays light and this stays reusable from a CLI.
"""

import datetime as _dt
import json
import os
import re
import shutil
from pathlib import Path

META_REL = Path("meta")
TASKS_REL = META_REL / "tasks.parquet"
EPISODES_GLOB = "meta/episodes/**/*.parquet"

# Heavy payload dirs: hard-linked into the copy, never edited.
_HEAVY_DIRS = {"data", "videos"}
# Never carried into a copy (HF download bookkeeping).
_SKIP_TOP = {".cache"}

# A leaf (folder) name: no path separators / whitespace, filesystem-safe.
_LEAF_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# --- naming ----------------------------------------------------------------
def validate_leaf(leaf):
    """Return a cleaned leaf name or raise ValueError. `leaf` is the last path
    segment (dataset folder name), not the full `org/name` repo id."""
    leaf = (leaf or "").strip()
    if not leaf:
        raise ValueError("数据集名不能为空")
    if "/" in leaf or "\\" in leaf:
        raise ValueError("数据集名不能包含路径分隔符（只填最后一段名字）")
    if not _LEAF_RE.match(leaf):
        raise ValueError("数据集名只能包含字母、数字、点、下划线、连字符")
    return leaf


def default_copy_dir(new_leaf, out_dir="datasets/TacVerse", today=None):
    """Compute an organization-rooted output path with no date subdirectory."""
    new_leaf = validate_leaf(new_leaf)
    return Path(out_dir) / new_leaf


# --- copy ------------------------------------------------------------------
def _link_or_copy(src, dst):
    """Hard-link a file, falling back to a real copy across devices."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copytree_linked(src, dst):
    """copytree that hard-links regular files (used for data/ & videos/)."""
    shutil.copytree(src, dst, copy_function=_link_or_copy)


def copy_dataset(src_dir, dst_dir, *, hardlink_heavy=True):
    """Copy a pulled dataset directory to `dst_dir` (must not exist).

    `meta/` and top-level files are real copies (so the prompt rewrite is
    isolated from the source). `data/` and `videos/` are hard-linked when
    `hardlink_heavy` is True to avoid duplicating gigabytes of video. `.cache/`
    is skipped. Returns the destination Path.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if not (src_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"源不是有效数据集（缺 meta/info.json）: {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(f"目标已存在: {dst_dir}")
    src_resolved = src_dir.resolve()
    dst_resolved = dst_dir.resolve()
    if dst_resolved == src_resolved or src_resolved in dst_resolved.parents:
        raise ValueError(f"目标不能是源目录或其子目录: {dst_dir}")

    dst_dir.mkdir(parents=True)
    for child in sorted(src_dir.iterdir()):
        if child.name in _SKIP_TOP:
            continue
        target = dst_dir / child.name
        if child.is_dir():
            if hardlink_heavy and child.name in _HEAVY_DIRS:
                _copytree_linked(child, target)
            else:
                shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    return dst_dir


# --- prompt edit -----------------------------------------------------------
def _rewrite_column(path, column, transform):
    """Read a parquet file, replace one column via `transform(pylist)->pylist`,
    write it back preserving the original schema (type + pandas metadata)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    idx = table.schema.get_field_index(column)
    if idx < 0:
        return False
    field = table.schema.field(idx)
    new_values = transform(table.column(column).to_pylist())
    new_col = pa.array(new_values, type=field.type)
    new_table = table.set_column(idx, field, new_col)
    pq.write_table(new_table, path)
    return True


def set_prompt(dst_dir, replacements):
    """Apply `{old_task: new_task}` across a dataset copy's metadata.

    Rewrites `meta/tasks.parquet` (the `task` column) and the per-episode
    `tasks` list column in `meta/episodes/**/*.parquet`. `task_index` values and
    `info.json` are untouched. Returns the number of task strings changed.
    """
    dst_dir = Path(dst_dir)
    replacements = {k: v for k, v in (replacements or {}).items() if k != v and v is not None}
    if not replacements:
        return 0

    tasks_path = dst_dir / TASKS_REL
    if not tasks_path.is_file():
        raise FileNotFoundError(f"缺少 meta/tasks.parquet: {tasks_path}")

    changed = {"n": 0}

    def _map_scalar(values):
        out = []
        for v in values:
            if v in replacements:
                out.append(replacements[v])
                changed["n"] += 1
            else:
                out.append(v)
        return out

    _rewrite_column(tasks_path, "task", _map_scalar)

    def _map_list(values):
        # each row is a list[str] of the tasks used by that episode
        return [
            [replacements.get(x, x) for x in (row or [])]
            for row in values
        ]

    for ep_path in sorted(dst_dir.glob(EPISODES_GLOB)):
        _rewrite_column(ep_path, "tasks", _map_list)

    return changed["n"]


# --- upload ----------------------------------------------------------------
def push_to_hub(dst_dir, repo_id, token, *, private=True, commit_message=None):
    """Upload a dataset copy to the HuggingFace Hub as `repo_id` (org/name).

    Creates the dataset repo if needed and uploads everything except `.cache/`.
    Returns the commit/repo URL string. Requires a valid write token.
    """
    from huggingface_hub import HfApi

    dst_dir = Path(dst_dir)
    if not (dst_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"不是有效数据集: {dst_dir}")
    if not token:
        raise ValueError("缺少 HuggingFace token，无法上传")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(dst_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".cache/*", "**/.cache/*"],
        commit_message=commit_message or "Edit dataset via tacverse-workbench",
    )


# --- convenience: read info for display ------------------------------------
def read_info(dataset_dir):
    """Return meta/info.json as a dict (or {} on failure)."""
    try:
        return json.loads((Path(dataset_dir) / META_REL / "info.json").read_text())
    except Exception:
        return {}
