#!/usr/bin/env python3
"""TacVerse 多模态物理具身数据集工作台 — PySide6 dashboard over Hugging Face.

Wraps the logic in download_dataset.py. Top bar (org combo + actions + progress
+ speed) is shared; below it a tabbed dashboard:

  * 看板   -> KPI cards (+ today's MVP) + filterable, sortable dataset table.
  * 趋势   -> daily new-hours bar + cumulative-hours line (pyqtgraph).
  * 分组统计 -> rollup by uploader / task / robot_type, plus daily group growth.
  * 数据集编辑 -> 左侧同看板的数据集详情表；右侧两组操作：① 改名 / 改 prompt
    (本地 pyarrow，data+videos 硬链接生成新副本，可推 Hub)；② 删除/拆分/合并/
    增删特征 (子进程调用 lerobot 官方 dataset_tools，见 lerobot_ops_runner.py)。

Buttons: 仅拉取统计信息 (stats only, no download) / 下载当前选中数据集 (one dataset)
/ 拉取组织及其下所有数据集 (download all) / 检查新增数据集 (name diff) /
打开本地目录 / 切换账号 (swap HF token).

Run in the lerobot-xense env:  python main_app.py
"""

import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path


def _configure_qt_plugin_path():
    """Prefer the PySide6 plugin directory over conda's qt6-main plugins."""
    plugin_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "PySide6"
        / "Qt"
        / "plugins"
    )
    platform_root = plugin_root / "platforms"
    if platform_root.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_root)


_configure_qt_plugin_path()

import pyqtgraph as pg
from shiboken6 import isValid as qt_is_valid
from PySide6.QtCore import QDate, Qt, QPoint, QRect, QSize, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QFontDatabase, QIcon, QPalette, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QFileDialog, QProgressBar, QPushButton, QScrollArea,
    QDateEdit, QSizePolicy, QSpinBox, QStackedWidget,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)


import annotations_reader as ann
import tasks_reader as tsk
import checks as chk_mod
import viewer_service as vsvc 
import download_dataset as dd
import dataset_editor as de
import episode_lengths as ep_len
import pico_motracker as pico
import lerobot_ops as lops

DATASETS_ROOT = "datasets"
DEFAULT_DATASET_ORG = "TacVerse"
OUT_DIR = str(Path(DATASETS_ROOT) / DEFAULT_DATASET_ORG)
ASSETS_DIR = Path(__file__).resolve().parent / "assets"  # logos / image assets
LOGO_PATH = ASSETS_DIR / "logo.png"
RECENT_ORGS = ["TacVerse", "Xense"]  # seeds the editable org combo
DOWNLOAD_WORKER_RELEASE_DELAY_MS = 1000

# Keep the widget palette independent from the host desktop theme.  Mixing
# Windows' native dark/light palette with light, per-widget QSS made some text
# unreadable, while Ubuntu and Windows also gave the same controls different
# padding and heights.  Fusion plus this palette is intentionally shared by
# every platform; platform-specific font fallback is handled below.
UI_COLORS = {
    "window": "#F3F5F7",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F9FC",
    "border": "#D0D5DD",
    "border_strong": "#B8C0CC",
    "text": "#1D2939",
    "text_muted": "#667085",
    "text_disabled": "#98A2B3",
    "blue": "#2563EB",
    "blue_hover": "#1D4ED8",
    "green": "#238636",
    "green_hover": "#1A7F37",
    "amber": "#D97706",
    "amber_hover": "#B45309",
    "red": "#C62828",
}

MUTED_TEXT_STYLE = f"color:{UI_COLORS['text_muted']}; font-size:9pt;"
BLUE_PANEL_STYLE = (
    "QGroupBox{font-weight:bold; border:1px solid #9EC5FE;"
    " border-radius:6px; margin-top:7px; padding-top:2px; background:#F6F9FF;}"
    "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#175CD3;}"
)
GREEN_PANEL_STYLE = (
    "QGroupBox{font-weight:bold; border:1px solid #A3D9A5;"
    " border-radius:6px; margin-top:7px; padding-top:2px; background:#F4FAF4;}"
    "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#237A36;}"
)


def configure_application_ui(app):
    """Apply a predictable light theme and a CJK-capable platform font."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(UI_COLORS["window"]))
    palette.setColor(QPalette.WindowText, QColor(UI_COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(UI_COLORS["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(UI_COLORS["surface_alt"]))
    palette.setColor(QPalette.ToolTipBase, QColor("#101828"))
    palette.setColor(QPalette.ToolTipText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Text, QColor(UI_COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(UI_COLORS["surface"]))
    palette.setColor(QPalette.ButtonText, QColor(UI_COLORS["text"]))
    palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, QColor(UI_COLORS["blue"]))
    palette.setColor(QPalette.Highlight, QColor(UI_COLORS["blue"]))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.PlaceholderText, QColor(UI_COLORS["text_disabled"]))
    palette.setColor(
        QPalette.Disabled, QPalette.Text, QColor(UI_COLORS["text_disabled"]))
    palette.setColor(
        QPalette.Disabled, QPalette.ButtonText, QColor(UI_COLORS["text_disabled"]))
    app.setPalette(palette)

    if sys.platform == "win32":
        family = "Microsoft YaHei UI"
    elif sys.platform == "darwin":
        family = "PingFang SC"
    else:
        family = "Noto Sans CJK SC"
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    # Qt will gracefully fall back when a distribution does not ship the
    # preferred CJK family; avoiding a font-database scan also keeps headless
    # Windows/offscreen runs free of a misleading bundled-font warning.
    font.setFamily(family)
    font.setPointSize(10)
    app.setFont(font)

    # Pixel dimensions here are logical Qt pixels and therefore follow the OS
    # scale factor.  A single stylesheet also prevents native Windows metrics
    # from making controls taller than their Ubuntu counterparts.
    app.setStyleSheet(f"""
        QWidget {{ color: {UI_COLORS['text']}; }}
        QToolTip {{
            color: #FFFFFF; background: #101828; border: 1px solid #344054;
            padding: 4px 6px;
        }}
        QPushButton {{
            min-height: 26px; padding: 2px 9px;
            background: {UI_COLORS['surface']};
            border: 1px solid {UI_COLORS['border_strong']}; border-radius: 5px;
        }}
        QPushButton:hover {{ background: #F2F4F7; border-color: #98A2B3; }}
        QPushButton:pressed {{ background: #EAECF0; }}
        QPushButton:disabled {{
            color: {UI_COLORS['text_disabled']}; background: #F2F4F7;
            border-color: {UI_COLORS['border']};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{
            min-height: 26px; padding: 1px 7px;
            color: {UI_COLORS['text']}; background: {UI_COLORS['surface']};
            border: 1px solid {UI_COLORS['border_strong']}; border-radius: 4px;
            selection-background-color: {UI_COLORS['blue']};
            selection-color: #FFFFFF;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
        QDoubleSpinBox:focus, QDateEdit:focus {{ border-color: {UI_COLORS['blue']}; }}
        QComboBox QAbstractItemView {{
            color: {UI_COLORS['text']}; background: {UI_COLORS['surface']};
            border: 1px solid {UI_COLORS['border_strong']};
            selection-background-color: {UI_COLORS['blue']};
            selection-color: #FFFFFF;
        }}
        QTabWidget::pane {{
            background: {UI_COLORS['surface']};
            border: 1px solid {UI_COLORS['border']}; border-radius: 4px;
        }}
        QTabBar::tab {{
            min-height: 28px; padding: 3px 13px;
            color: {UI_COLORS['text_muted']}; background: #EAECF0;
            border: 1px solid {UI_COLORS['border']};
            border-bottom: none; border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            color: {UI_COLORS['blue']}; background: {UI_COLORS['surface']};
            font-weight: bold;
        }}
        QTableWidget, QTreeWidget, QListWidget {{
            color: {UI_COLORS['text']}; background: {UI_COLORS['surface']};
            alternate-background-color: {UI_COLORS['surface_alt']};
            border: 1px solid {UI_COLORS['border']}; border-radius: 3px;
            selection-background-color: #D9E8FF; selection-color: #102A56;
        }}
        QHeaderView::section {{
            min-height: 26px; padding: 2px 7px;
            color: #344054; background: #EAECF0;
            border: none; border-right: 1px solid {UI_COLORS['border']};
            border-bottom: 1px solid {UI_COLORS['border']}; font-weight: bold;
        }}
        QGroupBox {{
            border: 1px solid {UI_COLORS['border']}; border-radius: 5px;
            margin-top: 7px; padding-top: 2px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 9px; padding: 0 4px; }}
        QScrollArea {{ border: none; background: transparent; }}
        QProgressBar {{
            min-height: 16px; color: #344054; background: #EAECF0;
            border: 1px solid {UI_COLORS['border']}; border-radius: 4px;
            text-align: center;
        }}
        QProgressBar::chunk {{ background: {UI_COLORS['blue']}; border-radius: 3px; }}
        QSplitter::handle {{ background: #D8DEE8; }}
        QSplitter::handle:hover {{ background: #98A2B3; }}
    """)

# HF uploader id -> Chinese name lives in config.json ("uploader_names"
# section — edit that to add people). Ids with no entry render as 未知. Loaded once
# at startup; edit the file then restart to pick up new names.
_UPLOADER_NAMES = dd.load_uploader_names()

# Thresholds for the custom quality checks (config.json "checks" section).
# Loaded once at startup; edit the file then restart to change standards.
_CHECKS_CFG = dd.load_config().get("checks", {})


def uploader_cn(hf_id):
    """Map an HF uploader id to its Chinese name, or 未知 if absent/unknown."""
    return _UPLOADER_NAMES.get(hf_id, "未知") if hf_id else "未知"


def _esc(s):
    """Minimal HTML escape for text placed into rich-text (QLabel) content."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Local, git-ignored file where the "切换账号" dialog persists its token so it
# survives restarts without being committed / shared with other users.
TOKEN_FILE = Path(__file__).resolve().parent / ".hf_token"


def load_saved_token():
    """Return the locally-persisted token (from the 切换账号 dialog), or None."""
    try:
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def save_token(tok):
    """Persist `tok` to the git-ignored .hf_token (0600), or clear it if empty."""
    try:
        if tok and tok.strip():
            TOKEN_FILE.write_text(tok.strip() + "\n", encoding="utf-8")
            os.chmod(TOKEN_FILE, 0o600)
        elif TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
    except OSError:
        pass


def resolve_token():
    """HF token to talk to the Hub with.

    Priority: the token saved by the "切换账号" dialog (so the account you pick
    in the UI sticks across restarts), then $HF_TOKEN, then the token cached by
    `huggingface-cli login`. Private datasets are only visible when this token
    belongs to an org member — e.g. a TacVerse member sees TacVerse's private
    repos.
    """
    saved = load_saved_token()
    if saved:
        return saved
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

pg.setConfigOptions(
    background=UI_COLORS["surface"], foreground=UI_COLORS["text"], antialias=True)

# Dashboard table columns: (header, dataset key, kind). "__delta__" is special.
TABLE_COLS = [
    ("数据集", "dataset_name", "str"),
    ("本地", "__local__", "num"),  # raw files under datasets/TacVerse/ → openable in viewer
    ("episodes", "total_episodes", "num"),
    ("frames", "total_frames", "num"),
    ("小时", "duration_hours", "num"),
    ("均时长(s)", "__avg_sec__", "num"),  # avg seconds/episode — quality signal
    ("检查", "__check__", "num"),  # custom quality-check badge (✅/⚠️N/❌N)
    ("fps", "fps", "num"),
    ("robot_type", "robot_type", "str"),
    ("任务数", "total_tasks", "num"),
    ("HF ID", "uploader", "str"),
    ("上传者", "__uploader_cn__", "str"),
    ("最后更新", "last_modified", "date"),
    ("今日新增ep", "__delta__", "num"),
    ("检查状态", "__quality_status__", "str"),
]

# Column that carries last_modified — the table's default sort key. Derived so it
# stays correct if columns are inserted/reordered above.
DATE_COL = next(i for i, (_, k, _) in enumerate(TABLE_COLS) if k == "last_modified")
LOCAL_COL = next(i for i, (_, k, _) in enumerate(TABLE_COLS) if k == "__local__")
QUALITY_STATUS_COL = next(
    i for i, (_, k, _) in enumerate(TABLE_COLS) if k == "__quality_status__")

# Order = dropdown order; first entry (上传者) is the default. robot_type last.
ROLLUP_DIMS = {
    "上传者": lambda d: uploader_cn(d.get("uploader")),
    "任务": lambda d: dd.task_prefix(d.get("dataset_name", "")),
    "robot_type": lambda d: d.get("robot_type"),
}


def fmt_day(yymmdd):
    """'260703' -> '2026-07-03'. Returns the input unchanged if unparseable."""
    try:
        return dt.datetime.strptime(yymmdd, "%y%m%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return yymmdd or "—"


def fmt_day_wd(yymmdd):
    """'260703' -> '07-03\\n周四' (MM-DD + Chinese weekday) for trend axis labels.
    Weekday helps tell workdays from weekends/holidays at a glance."""
    try:
        d = dt.datetime.strptime(yymmdd, "%y%m%d")
        return d.strftime("%m-%d") + "\n周" + "一二三四五六日"[d.weekday()]
    except (ValueError, TypeError):
        return yymmdd or "—"


def qdate_from_yymmdd(yymmdd):
    try:
        d = dt.datetime.strptime(yymmdd, "%y%m%d")
        return QDate(d.year, d.month, d.day)
    except (ValueError, TypeError):
        return QDate()


def days_between(yymmdd_from, yymmdd_to):
    """Whole days from one YYMMDD date to another, or None if either is unparseable."""
    try:
        a = dt.datetime.strptime(yymmdd_from, "%y%m%d")
        b = dt.datetime.strptime(yymmdd_to, "%y%m%d")
        return (b - a).days
    except (ValueError, TypeError):
        return None


def _is_float(s):
    """True if s parses as a float (used to tell split fractions from indices)."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def fmt_value(v):
    """Render a value: thousands separators for numbers, — for None/empty."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:,}"
    if v is None or v == "":
        return "—"
    return str(v)


def fmt_speed(bytes_per_sec):
    """Human-readable transfer rate, e.g. '12.3 MB/s'."""
    rate = max(float(bytes_per_sec), 0.0)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if rate < 1024 or unit == "GB/s":
            return f"{rate:.1f} {unit}"
        rate /= 1024


def dir_size(path):
    """Total bytes of materialized files under path (skips hf .cache blobs)."""
    if path is None:
        return 0
    try:
        p = Path(path)
        if not p.exists():
            return 0
        files = p.rglob("*")
    except OSError:
        return 0
    total = 0
    try:
        for f in files:
            if ".cache" in f.parts:
                continue
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


class NumericItem(QTableWidgetItem):
    """Table item that displays formatted text but sorts by a numeric key."""

    def __init__(self, text, sort_key):
        super().__init__(text)
        self.sort_key = sort_key

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self.sort_key < other.sort_key
        return super().__lt__(other)


class FrozenDatasetTable(QWidget):
    """Two synchronized tables: fixed dataset column + scrollable detail columns."""

    cellClicked = Signal(int, int)
    cellDoubleClicked = Signal(int, int)
    itemSelectionChanged = Signal()

    def __init__(self, rows=0, columns=0, parent=None, frozen_width=440):
        super().__init__(parent)
        self._columns = columns
        self._sort_column = 0
        self._sort_order = Qt.AscendingOrder
        self.fixed = QTableWidget(rows, 1)
        self.detail = QTableWidget(rows, max(columns - 1, 0))

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter)
        self.splitter.addWidget(self.fixed)
        self.splitter.addWidget(self.detail)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([frozen_width, max(900, frozen_width * 2)])

        self.fixed.setMinimumWidth(220)
        self.fixed.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.fixed.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.detail.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.detail.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.fixed.verticalHeader().setVisible(False)
        self.detail.verticalHeader().setVisible(False)
        self.fixed.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.fixed.setColumnWidth(0, frozen_width)
        self.splitter.splitterMoved.connect(
            lambda *args: self._safe_call(self._sync_fixed_column_width, *args))

        self.fixed.verticalScrollBar().valueChanged.connect(
            self.detail.verticalScrollBar().setValue)
        self.detail.verticalScrollBar().valueChanged.connect(
            self.fixed.verticalScrollBar().setValue)
        self.fixed.cellClicked.connect(
            lambda row, _col: self._safe_call(self._handle_cell_clicked, row, 0))
        self.detail.cellClicked.connect(
            lambda row, col: self._safe_call(self._handle_cell_clicked, row, col + 1))
        self.fixed.cellDoubleClicked.connect(
            lambda row, col: self.cellDoubleClicked.emit(row, col))
        self.detail.cellDoubleClicked.connect(
            lambda row, col: self.cellDoubleClicked.emit(row, col + 1))
        self.fixed.itemSelectionChanged.connect(
            lambda: self._safe_call(self._sync_selection, self.fixed))
        self.detail.itemSelectionChanged.connect(
            lambda: self._safe_call(self._sync_selection, self.detail))
        self.fixed.horizontalHeader().sectionClicked.connect(
            lambda _col: self._safe_call(self.sortItems, 0))
        self.detail.horizontalHeader().sectionClicked.connect(
            lambda col: self._safe_call(self.sortItems, col + 1))

    @staticmethod
    def _safe_call(func, *args):
        try:
            return func(*args)
        except Exception:
            traceback.print_exc()
            return None

    def setHorizontalHeaderLabels(self, labels):
        self._columns = len(labels)
        self.fixed.setColumnCount(1 if labels else 0)
        self.detail.setColumnCount(max(len(labels) - 1, 0))
        self.fixed.setHorizontalHeaderLabels(labels[:1])
        self.detail.setHorizontalHeaderLabels(labels[1:])

    def horizontalHeaderItem(self, column):
        return self.fixed.horizontalHeaderItem(0) if column == 0 else self.detail.horizontalHeaderItem(column - 1)

    def horizontalHeader(self):
        return self.detail.horizontalHeader()

    def verticalHeader(self):
        return self.detail.verticalHeader()

    def setSortingEnabled(self, enabled):
        self._sorting_enabled = enabled

    def setEditTriggers(self, triggers):
        self.fixed.setEditTriggers(triggers)
        self.detail.setEditTriggers(triggers)

    def setSelectionBehavior(self, behavior):
        self.fixed.setSelectionBehavior(behavior)
        self.detail.setSelectionBehavior(behavior)

    def setSelectionMode(self, mode):
        self.fixed.setSelectionMode(mode)
        self.detail.setSelectionMode(mode)

    def setRowCount(self, rows):
        self.fixed.setRowCount(rows)
        self.detail.setRowCount(rows)

    def rowCount(self):
        return self.fixed.rowCount()

    def setColumnWidth(self, column, width):
        if column == 0:
            self.fixed.setColumnWidth(0, width)
            sizes = self.splitter.sizes()
            detail_width = sizes[1] if len(sizes) > 1 else max(900, width * 2)
            self.splitter.setSizes([width, detail_width])
        else:
            self.detail.setColumnWidth(column - 1, width)

    def _sync_fixed_column_width(self, *args):
        self.fixed.setColumnWidth(0, max(120, self.fixed.viewport().width()))

    def setItem(self, row, column, item):
        if column == 0:
            self.fixed.setItem(row, 0, item)
        else:
            self.detail.setItem(row, column - 1, item)

    def item(self, row, column):
        return self.fixed.item(row, 0) if column == 0 else self.detail.item(row, column - 1)

    def setRowHidden(self, row, hide):
        self.fixed.setRowHidden(row, hide)
        self.detail.setRowHidden(row, hide)

    def currentRow(self):
        row = self.fixed.currentRow()
        return row if row >= 0 else self.detail.currentRow()

    def selectRow(self, row):
        self.fixed.blockSignals(True)
        self.detail.blockSignals(True)
        try:
            self.fixed.selectRow(row)
            self.detail.selectRow(row)
        finally:
            self.fixed.blockSignals(False)
            self.detail.blockSignals(False)
        self.itemSelectionChanged.emit()

    def selectedRows(self):
        rows = set()
        for table in (self.fixed, self.detail):
            rows.update(index.row() for index in table.selectionModel().selectedIndexes())
        if not rows:
            row = self.currentRow()
            if row >= 0:
                rows.add(row)
        return sorted(rows)

    def _handle_cell_clicked(self, row, column):
        modifiers = QApplication.keyboardModifiers()
        if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
            self.selectRow(row)
        self.cellClicked.emit(row, column)

    def _sync_selection(self, source):
        target = self.detail if source is self.fixed else self.fixed
        rows = sorted({
            index.row() for index in source.selectionModel().selectedIndexes()
            if 0 <= index.row() < target.rowCount()
        })
        target.blockSignals(True)
        try:
            target.clearSelection()
            for row in rows:
                target.selectRow(row)
        finally:
            target.blockSignals(False)
        if rows:
            self.itemSelectionChanged.emit()

    def sortItems(self, column, order=None):
        if order is None:
            order = (Qt.DescendingOrder if self._sort_column == column
                     and self._sort_order == Qt.AscendingOrder else Qt.AscendingOrder)
        self._sort_column = column
        self._sort_order = order
        rows = []
        for row in range(self.rowCount()):
            items = [self._clone_item(self.item(row, col)) for col in range(self._columns)]
            key_item = items[column] if 0 <= column < len(items) else None
            rows.append((key_item, items))
        reverse = order == Qt.DescendingOrder
        rows.sort(key=lambda row: self._item_sort_key(row[0]), reverse=reverse)
        self.setRowCount(len(rows))
        for row, (_, items) in enumerate(rows):
            for col, item in enumerate(items):
                self.setItem(row, col, item)

    @staticmethod
    def _item_sort_key(item):
        if isinstance(item, NumericItem):
            return item.sort_key
        return item.text() if item else ""

    @staticmethod
    def _clone_item(item):
        if item is None:
            return QTableWidgetItem("")
        clone = item.clone()
        if isinstance(item, NumericItem):
            clone = NumericItem(item.text(), item.sort_key)
            clone.setData(Qt.UserRole, item.data(Qt.UserRole))
            clone.setToolTip(item.toolTip())
            clone.setForeground(item.foreground())
        return clone


# --------------------------------------------------------------------------- #
# Worker threads (network + downloads run off the UI thread)
# --------------------------------------------------------------------------- #
class PullWorker(QThread):
    """Discover an org's datasets and pull them all, streaming progress."""

    log = Signal(str)
    progress = Signal(int, int)  # done, total
    done = Signal(dict, str)     # report, out_path
    error = Signal(str)

    def __init__(self, org, out_dir, token):
        super().__init__()
        self.org, self.out_dir, self.token = org, out_dir, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            self.log.emit(f"Discovering datasets under '{self.org}' ...")
            meta = dd.discover_datasets_meta(self.org, self.token)
            repo_ids = [m["id"] for m in meta]
            meta_map = {m["id"]: m["last_modified"] for m in meta}
            self.log.emit(f"Found {len(repo_ids)} datasets.")
            if not repo_ids:
                self.error.emit(f"No datasets found under '{self.org}'.")
                return
            report, out_path = dd.run_pull(
                repo_ids, out_dir=self.out_dir, org=self.org, token=self.token,
                meta_map=meta_map, with_uploader=True,
                log=self.log.emit, progress=lambda d, t: self.progress.emit(d, t),
            )
            self.log.emit("更新 Hugging Face 变更历史缓存 ...")
            dd.update_hf_change_history(report, token=self.token, log=self.log.emit)
            self.done.emit(report, str(out_path) if out_path else "")
        except Exception as exc:
            self.error.emit(str(exc))


class DownloadOneWorker(QThread):
    """Download a single selected dataset (not the whole org) to save time."""

    done = Signal(str)   # local_dir of the downloaded dataset
    log = Signal(str)
    error = Signal(str)

    def __init__(self, repo_id, out_dir, token):
        super().__init__()
        self.repo_id, self.out_dir, self.token = repo_id, out_dir, token
        self.local_dir = ""
        self.error_msg = ""

    def run(self):
        try:
            dd.normalize_proxy_env()
            dataset_dir = Path(self.out_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dd.pull_dataset(
                self.repo_id, dataset_dir, revision=None, token=self.token,
                log=self.log.emit,
            )
            self.local_dir = str(dataset_dir / self.repo_id.split("/")[-1])
        except Exception as exc:
            self.error_msg = str(exc)
        else:
            self.done.emit(self.local_dir)
        if self.error_msg:
            self.error.emit(self.error_msg)


class StatsWorker(QThread):
    """Fetch stats only (meta/info.json + commits) — no dataset files pulled."""

    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, org, token):
        super().__init__()
        self.org, self.token = org, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            self.log.emit(f"Discovering datasets under '{self.org}' ...")
            meta = dd.discover_datasets_meta(self.org, self.token)
            repo_ids = [m["id"] for m in meta]
            meta_map = {m["id"]: m["last_modified"] for m in meta}
            self.log.emit(f"Found {len(repo_ids)} datasets.")
            if not repo_ids:
                self.error.emit(f"No datasets found under '{self.org}'.")
                return
            report = dd.collect_stats(
                repo_ids, org=self.org, token=self.token,
                meta_map=meta_map, with_uploader=True,
                log=self.log.emit, progress=lambda d, t: self.progress.emit(d, t),
            )
            self.log.emit("更新 Hugging Face 变更历史缓存 ...")
            dd.update_hf_change_history(report, token=self.token, log=self.log.emit)
            self.done.emit(report)
        except Exception as exc:
            self.error.emit(str(exc))


class CheckWorker(QThread):
    """Compare Hub dataset names against the last pulled report (names only)."""

    result = Signal(list, list, int, int)  # new, removed, hub_count, local_count
    error = Signal(str)

    def __init__(self, org, out_dir, token):
        super().__init__()
        self.org, self.out_dir, self.token = org, out_dir, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            hub = set(dd.discover_datasets(self.org, self.token))
            local = set()
            latest_report, _ = dd.load_latest_local_report(
                self.out_dir, org=self.org)
            if latest_report:
                report = latest_report
                local = {d["dataset_name"] for d in report.get("datasets", [])}
            # Stats-only runs write dataset_log.json but not pull_result_*.json,
            # so also count any raw dirs that are present on disk.
            for info in Path(self.out_dir).glob("*/*/meta/info.json"):
                local.add(f"{self.org}/{info.parent.parent.name}")
            self.result.emit(sorted(hub - local), sorted(local - hub),
                             len(hub), len(local))
        except Exception as exc:
            self.error.emit(str(exc))


class IdentityWorker(QThread):
    """Resolve who the current token logs in as and how many org datasets it can
    see — so the status bar can flag token/permission problems at a glance."""

    done = Signal(str, bool, str, int)  # username, has_token, org, count(-1=err)

    def __init__(self, org, token):
        super().__init__()
        self.org, self.token = org, token

    def run(self):
        dd.normalize_proxy_env()
        name = ""
        if self.token:
            try:
                from huggingface_hub import HfApi
                name = HfApi().whoami(token=self.token).get("name", "") or ""
            except Exception:
                name = ""  # token present but invalid/expired
        try:
            count = len(dd.discover_datasets_meta(self.org, self.token))
        except Exception:
            count = -1
        self.done.emit(name, bool(self.token), self.org, count)


class ReportWorker(QThread):
    """Fetch the viewer's /report analysis off the UI thread (it can take tens
    of seconds). `seq` lets the UI ignore results from stale selections."""

    done = Signal(int, str, object, str)  # seq, rel_path, report|None, error

    def __init__(self, viewer, rel_path, seq):
        super().__init__()
        self.viewer, self.rel_path, self.seq = viewer, rel_path, seq

    def run(self):
        report, err = self.viewer.report(self.rel_path, timeout=180)
        self.done.emit(self.seq, self.rel_path, report, err or "")


class PicoMotrackerWorker(QThread):
    """Run the opt-in local PICO MoTracker scan away from the Qt UI thread."""

    done = Signal(int, str, object, str)  # seq, dataset key, result|None, error

    def __init__(self, dataset_dir, cfg, seq):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.cfg = cfg
        self.seq = seq

    def run(self):
        try:
            result = pico.detect(self.dataset_dir, self.cfg)
            self.done.emit(self.seq, str(self.dataset_dir.resolve()), result, "")
        except Exception as exc:
            self.done.emit(self.seq, str(self.dataset_dir.resolve()), None, str(exc))


class DoctorWorker(QThread):
    """Run the viewer Doctor stream away from the Qt UI thread."""

    progress = Signal(int, str)
    done = Signal(int, str, object, str)  # seq, dataset key, result|None, error

    def __init__(self, viewer, rel_path, options, seq):
        super().__init__()
        self.viewer, self.rel_path = viewer, rel_path
        self.options, self.seq = options, seq

    def run(self):
        def emit_progress(progress):
            percent = int(progress.get("overall_percent", 0) or 0)
            message = str(progress.get("message") or "Doctor running…")
            self.progress.emit(percent, message)

        result, error = self.viewer.doctor(
            self.rel_path,
            max_episodes=self.options.get("maxEpisodes"),
            episode_range=self.options.get("episodeRange"),
            checks=self.options.get("checks"),
            timeout=300,
            on_progress=emit_progress,
        )
        self.done.emit(self.seq, self.rel_path, result, error or "")


class QualityWorker(QThread):
    """Run local or selectively cached episode quality checks off the UI thread."""

    done = Signal(int, str, list, str)
    progress = Signal(str, int)

    def __init__(self, seq, dataset, token, cfg):
        super().__init__()
        self.seq = seq
        self.dataset = dict(dataset or {})
        self.dataset_name = self.dataset.get("dataset_name") or ""
        self.token = token
        self.cfg = dict(cfg or {})
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        cfg = dict(self.cfg.get("local_quality") or {})
        cfg["token"] = self.token
        report_dir = ""
        try:
            import dataset_quality

            self.progress.emit("准备深度检查...", 0)
            issues, report_dir = dataset_quality.scan_dataset_with_report(
                self.dataset,
                out_dir=OUT_DIR,
                cfg=cfg,
                progress=lambda message, pct=None: self.progress.emit(
                    message, int(pct or 0)),
                cancel=lambda: self.cancel_requested,
            )
            status, message, details = chk_mod.format_local_quality_issues(issues)
        except Exception as exc:
            status = chk_mod.SKIP
            message = "检查已取消" if self.cancel_requested else f"检查出错: {exc}"
            details = []
        result = chk_mod.CheckResult(
            "episode_local_quality", "Episode 级质量定位", "local_quality",
            status, message, details,
        )
        self.done.emit(self.seq, self.dataset_name, [result], report_dir)


class EditWorker(QThread):
    """Write an edited copy of a pulled dataset off the UI thread: hard-link the
    heavy payload into a new dir, then rewrite the prompt in its metadata."""

    done = Signal(str, int)  # dst_dir, n_prompts_changed
    error = Signal(str)

    def __init__(self, src_dir, dst_dir, replacements):
        super().__init__()
        self.src_dir, self.dst_dir, self.replacements = src_dir, dst_dir, replacements

    def run(self):
        try:
            de.copy_dataset(self.src_dir, self.dst_dir)
            n = de.set_prompt(self.dst_dir, self.replacements)
            self.done.emit(str(self.dst_dir), n)
        except Exception as exc:
            # Roll back a half-written copy so a retry starts clean.
            try:
                import shutil
                if Path(self.dst_dir).exists():
                    shutil.rmtree(self.dst_dir)
            except Exception:
                pass
            self.error.emit(str(exc))


class PushWorker(QThread):
    """Upload an edited dataset copy to the Hub off the UI thread."""

    done = Signal(str)  # commit / repo URL
    error = Signal(str)

    def __init__(self, dst_dir, repo_id, token, private=True):
        super().__init__()
        self.dst_dir, self.repo_id = dst_dir, repo_id
        self.token, self.private = token, private

    def run(self):
        try:
            dd.normalize_proxy_env()
            url = de.push_to_hub(self.dst_dir, self.repo_id, self.token,
                                 private=self.private)
            self.done.emit(str(url))
        except Exception as exc:
            self.error.emit(str(exc))


class LerobotOpWorker(QThread):
    """Run a lerobot dataset operation (delete/split/merge/add/remove) via the
    subprocess runner off the UI thread, streaming the child's log lines."""

    log = Signal(str)
    done = Signal(dict)   # the runner's result dict
    error = Signal(str)

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            result = lops.run_op(
                self.spec, log=self.log.emit,
                cancel=lambda: self.cancel_requested)
            if result.get("ok"):
                self.done.emit(result)
            else:
                self.error.emit(result.get("error") or "操作失败")
        except Exception as exc:
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class FlowLayout(QLayout):
    """Left-to-right layout that wraps to a new row when it runs out of width
    (like flowing text). Used for the top toolbar so its minimum width is just
    its widest single control — the window can shrink to fit small laptops and
    the controls wrap instead of forcing the window wider (which previously made
    it snap wider on the first relayout after opening)."""

    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=4):
        super().__init__(parent)
        self._items = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    # Qt plumbing --------------------------------------------------------- #
    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # Core wrapping pass -------------------------------------------------- #
    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        right = rect.right() - m.right()
        line_height = 0
        for it in self._items:
            hint = it.sizeHint()
            next_x = x + hint.width()
            if next_x - 1 > right and line_height > 0:  # wrap to next row
                x = rect.x() + m.left()
                y = y + line_height + self._vspace
                next_x = x + hint.width()
                line_height = 0
            if not test_only:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x + self._hspace
            line_height = max(line_height, hint.height())
        return y + line_height + m.bottom() - rect.y()


# --------------------------------------------------------------------------- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TacVerse 多模态物理具身数据集工作台")
        if LOGO_PATH.is_file():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))
        # ---- 主窗口默认分辨率（如需再调整，改这里） ----------------------
        # target_w / target_h 是首选像素尺寸；随后按屏幕「可用区域」(不含任务栏)
        # 的比例收窄并居中，避免在小屏笔记本上开得过大、预览不全。想改默认大小
        # 就改 target_w/target_h；想在小屏上留更多余量就调低 0.90 / 0.88 两个系数。
        target_w, target_h = 1440, 900
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(target_w, int(avail.width() * 0.90))
            target_h = min(target_h, int(avail.height() * 0.88))
        screen_width = screen.availableGeometry().width() if screen else target_w
        # Windows scaling reduces the logical screen width. Keep the dataset
        # name useful without letting its fixed width squeeze every detail field.
        self.dataset_column_width = max(280, min(440, int(screen_width * 0.23)))
        self.setWindowState(Qt.WindowNoState)
        self.resize(target_w, target_h)
        if screen:
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())
        self.token = resolve_token()
        self.worker = None
        self._pull_worker = None
        self._check_worker = None
        self._download_workers = []
        self._download_started = 0
        self._download_completed = 0
        self._download_successes = []
        self._download_failures = []
        self._retired_download_workers = []
        self._download_message_box = None
        self._stats_worker = None
        self._closing = False
        self._shutdown_done = False
        self.report = None
        self.history = []
        self._id_workers = []  # in-flight IdentityWorkers (kept alive until done)
        self._id_seq = 0       # monotonic id; only the latest check may update UI
        # Vendored viewer (xense_lerobot_viewer) managed as a black-box service.
        # Port 3001 keeps it separate from any viewer the user runs on 3000, so
        # Workbench always launches its own instance bound to datasets/TacVerse.
        self.viewer = vsvc.ViewerService(port=3001)
        self._report_workers = []   # in-flight ReportWorkers
        self._report_seq = 0        # only the latest selection's report renders
        self._pico_workers = []     # in-flight opt-in trajectory scans
        self._pico_seq = 0
        self._pico_cache = {}       # dataset path -> DetectionResult or error tuple
        self._doctor_workers = []
        self._doctor_seq = 0
        self._doctor_cache = {}     # (rel_path, scope) -> Doctor response
        self._quality_records = self._load_quality_records()
        self._quality_status = {
            name: record.get("status", "未检查")
            for name, record in self._quality_records.items()
        }
        self._quality_reports = {
            name: record.get("report_dir", "")
            for name, record in self._quality_records.items()
        }
        self._quality_seq = 0
        self._rollup_range_initialized = False
        self.quality_worker = None
        # 数据集编辑 state: the dataset being edited, its prompt editors, and the
        # last copy written (so 推送到 Hub knows what to upload). Workers held on
        # self so they are not GC'd mid-run.
        self._edit_src = None       # selected dataset dict for the edit tab
        self._prompt_edits = []     # [(task_index, old_task, QLineEdit)]
        self._last_copy_dir = None  # Path of the most recent edited copy
        self._last_copy_leaf = None # leaf name of that copy (for the repo id)
        self._edit_worker = None
        self._push_worker = None
        self._op_worker = None
        self.quality_worker = None

        self._build_ui()

        # Auto-start the viewer so the analysis panel works without a manual
        # step. Non-blocking; the Viewer tab's status shows progress.
        if self.viewer.available():
            self.viewer.start(self._viewer_root(), wait=False)

        self._watch_dir = None
        self._prev_bytes = 0
        self._prev_t = None
        self.speed_timer = QTimer(self)
        self.speed_timer.setInterval(1000)
        self.speed_timer.timeout.connect(
            self._guarded("测速刷新", self._tick_speed, stop_speed=False))
        self.quality_status_timer = QTimer(self)
        self.quality_status_timer.setInterval(3000)
        self.quality_status_timer.timeout.connect(
            self._guarded("同步检查状态", self._sync_quality_status_from_disk))
        self.quality_status_timer.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._prepare_shutdown)

        # 看板/表格 default to the LAST pull's results (from committed history) so
        # they aren't blank on open; a stale banner flags that it's not live, and
        # 统计/拉取 replaces it with fresh data.
        migration_note = ""
        try:
            dd.migrate_pull_history_to_log()
        except OSError as exc:
            migration_note = f"；旧历史迁移失败: {exc}"
        active_org = self.org_combo.currentText().strip() or dd.ORG
        try:
            self.history = dd.load_history(OUT_DIR, org=active_org)
            self.hf_changes = dd.load_hf_change_history()
        except Exception as exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self.history = []
            self.hf_changes = {"version": 1, "repos": {}}
            migration_note += f"；本地历史读取失败: {exc}"
        last = self.history[-1] if self.history else None
        if last:
            self.report = last
            try:
                self._refresh_all()
                self._show_stale_banner(last)
            except Exception as exc:
                self._handle_ui_exception("加载本地快照", exc, stop_speed=False)
                self.report = None
                self.history = []
                self._set_rollup_range_defaults()
                self._refresh_rollup()
        else:
            self._set_rollup_range_defaults()
            try:
                self._refresh_rollup()
            except Exception as exc:
                self._handle_ui_exception("初始化分组统计", exc, stop_speed=False)
        self.status.setText(
            "就绪：「仅拉取统计信息」(快) / 「下载当前选中数据集」/ "
            f"「拉取组织及其下所有数据集」{migration_note}。")
        self._refresh_identity()  # populate the login/visibility indicator

    def _load_quality_records(self):
        try:
            import dataset_quality
            cfg = (_CHECKS_CFG.get("local_quality") or {})
            records = dataset_quality.load_quality_status(
                path="quality_status.local.json",
                report_dir=cfg.get("report_dir", ".quality_reports"))
            dataset_quality.save_quality_status(records, path="quality_status.local.json")
            return records
        except Exception:
            return {}

    def _save_quality_records(self):
        try:
            import dataset_quality
            dataset_quality.save_quality_status(
                self._quality_records, path="quality_status.local.json")
        except Exception:
            pass

    def _mark_quality_unchecked(self, name):
        if not name:
            return
        self._quality_records.pop(name, None)
        self._quality_status[name] = "未检查"
        self._quality_reports.pop(name, None)
        self._save_quality_records()
        self._update_quality_status_cells(name)

    def _sync_quality_status_from_disk(self):
        changed = []
        for name, status in list(self._quality_status.items()):
            if status != "已检查":
                continue
            report_dir = self._quality_reports.get(name)
            if not report_dir or not Path(report_dir).is_dir():
                changed.append(name)
        for name in changed:
            self._mark_quality_unchecked(name)
        if changed:
            self.status.setText(f"检查报告已删除，已同步 {len(changed)} 个数据集为未检查。")

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 6)
        root.setSpacing(5)

        # The toolbar is intentionally split into two functional rows.  The
        # first row contains data acquisition/maintenance actions; the second
        # row contains account, Viewer and status controls.  Keeping status
        # widgets out of the action row prevents long identity text and the
        # clock from forcing the primary buttons into an unclear wrap order.
        toolbar = QVBoxLayout()
        toolbar.setSpacing(1)
        row1 = QHBoxLayout()
        row1.setSpacing(5)
        row2 = QHBoxLayout()
        row2.setSpacing(5)

        def separator(layout):
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)

        def section_label(text, layout):
            label = QLabel(text)
            label.setStyleSheet("color:#666; font-size:11px; font-weight:bold;")
            layout.addWidget(label)
        if LOGO_PATH.is_file():
            logo = QLabel()
            logo.setPixmap(QPixmap(str(LOGO_PATH)).scaledToHeight(
                30, Qt.SmoothTransformation))
            logo.setToolTip("TacVerse")
            row1.addWidget(logo)
        section_label("数据源", row1)
        row1.addWidget(QLabel("组织:"))
        self.org_combo = QComboBox()
        self.org_combo.setEditable(True)
        self.org_combo.addItems(RECENT_ORGS)
        self.org_combo.setMinimumWidth(160)
        self.org_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.org_combo.currentIndexChanged.connect(
            self._guarded("刷新登录状态", self._refresh_identity))
        self.org_combo.lineEdit().editingFinished.connect(
            self._guarded("刷新登录状态", self._refresh_identity))
        row1.addWidget(self.org_combo)

        separator(row1)
        section_label("数据获取", row1)
        self.btn_stats = QPushButton("刷新统计")
        self.btn_stats.setToolTip("仅获取数据集元信息，不下载 Parquet 和视频，速度最快。")
        self.btn_download = QPushButton("下载选中")
        self.btn_download.setToolTip(
            "下载当前表格中选中的一个或多个数据集；"
            "可 Ctrl/Shift 多选，下载任务并行执行。")
        self.btn_pull = QPushButton("同步全部")
        self.btn_pull.setToolTip("下载当前组织下全部数据集，速度较慢并占用磁盘空间。")
        for button in (self.btn_stats, self.btn_download, self.btn_pull):
            row1.addWidget(button)

        separator(row1)
        section_label("数据维护", row1)
        self.btn_check = QPushButton("检查新增")
        self.btn_check.setToolTip("检查 Hub 中新增或本地缺失的数据集。")
        self.btn_manual_stats = QPushButton("手动补录")
        self.btn_manual_stats.setToolTip("手动补录某一天的数据集统计快照。")
        self.btn_open = QPushButton("数据目录")
        self.btn_open.setToolTip("打开本地 datasets/TacVerse/ 目录。")
        self.btn_stats.clicked.connect(self._guarded("刷新统计", self.on_stats))
        self.btn_download.clicked.connect(
            self._guarded("下载选中数据集", self.on_download_selected))
        self.btn_pull.clicked.connect(self._guarded("同步全部数据集", self.on_pull))
        self.btn_check.clicked.connect(self._guarded("检查新增", self.on_check))
        self.btn_manual_stats.clicked.connect(
            self._guarded("手动补录统计", self.on_manual_stats))
        self.btn_open.clicked.connect(self._guarded("打开数据目录", self.on_open_dir))

        primary_css = (
            "QPushButton { font-weight: bold; padding: 6px 13px; border-radius: 6px;"
            " color: white; background: %s; }"
            "QPushButton:hover { background: %s; }"
            "QPushButton:disabled { color:#667085; background:#E4E7EC; }"
        )
        self.btn_stats.setStyleSheet(
            primary_css % (UI_COLORS["green"], UI_COLORS["green_hover"]))
        self.btn_download.setStyleSheet(
            primary_css % (UI_COLORS["amber"], UI_COLORS["amber_hover"]))
        self.btn_pull.setStyleSheet(
            primary_css % (UI_COLORS["blue"], UI_COLORS["blue_hover"]))
        secondary_css = (
            "QPushButton { padding: 6px 11px; border-radius: 6px; color: #444;"
            " border: 1px solid #C4C4C4; background: #F5F5F5; }"
            "QPushButton:hover { background: #ECECEC; }"
        )
        for b in (self.btn_stats, self.btn_download, self.btn_pull):
            b.setMinimumHeight(30)
            b.setStyleSheet(primary_css % (
                "#34A853" if b is self.btn_stats else
                "#F59E0B" if b is self.btn_download else "#4C8BF5",
                "#2E9247" if b is self.btn_stats else
                "#D98A00" if b is self.btn_download else "#3B7AE0"))
        for b in (self.btn_check, self.btn_manual_stats, self.btn_open):
            b.setStyleSheet(secondary_css)
            row1.addWidget(b)

        row1.addStretch(1)

        # Account and visibility status are kept together on the second row.
        section_label("账号", row2)
        self.btn_account = QPushButton("切换账号")
        self.btn_account.setStyleSheet(secondary_css)
        self.btn_account.setToolTip("切换 Hugging Face 账号或更新访问令牌。")
        self.btn_account.clicked.connect(
            self._guarded("切换账号", self.on_switch_account))
        row2.addWidget(self.btn_account)
        self.identity_label = QLabel("登录状态: 检测中…")
        self.identity_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.identity_label.setStyleSheet(f"color:{UI_COLORS['text_muted']};")
        self.identity_label.setMinimumWidth(245)
        row2.addWidget(self.identity_label)

        separator(row2)
        section_label("Viewer", row2)
        self.top_viewer_dot = QLabel("● Viewer")
        self.top_viewer_dot.setToolTip("Viewer 服务状态")
        self.top_viewer_dot.setFixedWidth(92)
        row2.addWidget(self.top_viewer_dot)
        self.top_viewer_start = QPushButton("启动")
        self.top_viewer_stop = QPushButton("停止")
        self.top_viewer_home = QPushButton("首页")
        self.open_viewer_btn = QPushButton("打开选中")
        self.open_viewer_btn.setToolTip("在浏览器的 Viewer 里打开选中的数据集")
        self.top_viewer_start.clicked.connect(
            self._guarded("启动 Viewer", self._viewer_start))
        self.top_viewer_stop.clicked.connect(
            self._guarded("停止 Viewer", self._viewer_stop))
        self.top_viewer_home.clicked.connect(
            self._guarded("打开 Viewer 首页", self._viewer_open_home))
        self.open_viewer_btn.clicked.connect(
            self._guarded("打开选中数据集 Viewer", self._open_selected_in_viewer))
        for b in (self.top_viewer_start, self.top_viewer_stop,
                  self.top_viewer_home, self.open_viewer_btn):
            b.setStyleSheet(secondary_css)
            row2.addWidget(b)

        separator(row2)
        section_label("目标", row2)
        row2.addWidget(QLabel("每日小时:"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(10)
        self.target_spin.valueChanged.connect(
            self._guarded("刷新 KPI", self._refresh_kpis))
        self.target_spin.setFixedWidth(72)
        self.target_spin.setToolTip("用于计算看板中的每日目标完成度。")
        row2.addWidget(self.target_spin)

        row2.addStretch(1)
        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet(
            f"color:{UI_COLORS['text']}; font-weight:bold;")
        self.clock_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row2.addWidget(self.clock_label)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(
            self._guarded("刷新时钟", self._tick_clock))
        self.clock_timer.start(1000)
        self._tick_clock()
        toolbar.addLayout(row1)
        toolbar.addLayout(row2)
        root.addLayout(toolbar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "看板")
        self.tabs.addTab(self._build_rollup_tab(), "分组统计")
        self.tabs.addTab(self._build_edit_tab(), "数据集编辑")
        self.tabs.addTab(self._build_viewer_tab(), "Viewer")
        root.addWidget(self.tabs, 1)

        # Progress: status line + (bar + speed)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        prog_row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setValue(0)
        prog_row.addWidget(self.bar, 1)
        self.speed_label = QLabel("—")
        self.speed_label.setMinimumWidth(90)
        self.speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(self.speed_label)
        root.addLayout(prog_row)

    # KPI cards: (report key, title, highlight?). 总小时数 is highlighted — it's
    # the key metric. Order per spec.
    KPI_CARDS = [
        ("total_datasets", "数据集总数", False),
        ("total_episodes", "总 episodes", False),
        ("new_episodes", "今日新增episodes", False),
        ("total_frames", "总 frames", False),
        ("total_hours", "总小时数", True),
        ("new_hours", "今日新增小时", False),
        ("completion", "目标完成度", False),
    ]

    def _build_dashboard_tab(self):
        """看板 = 左「数据集统计分区」(总览 + 详情表) | 右「数据集检查分区」(分析网格)."""
        w = QWidget()
        outer = QHBoxLayout(w)
        outer.setContentsMargins(3, 3, 3, 3)
        outer.setSpacing(4)
        split = QSplitter(Qt.Horizontal)

        # ===== LEFT: 数据集统计分区 =====
        left = QGroupBox("数据集统计分区")
        left.setStyleSheet(BLUE_PANEL_STYLE)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 6)
        lv.setSpacing(5)

        # Stale-data banner: on open we default to showing the LAST pull's results
        # (so the 看板/表格 aren't blank), clearly flagged as not live. Hidden once
        # a fresh 统计/拉取 replaces the data.
        self.stale_banner = QLabel("")
        self.stale_banner.setWordWrap(True)
        self.stale_banner.setVisible(False)
        self.stale_banner.setStyleSheet(
            "background:#fff3cd; color:#8a6d3b; border:1px solid #ffe69c;"
            " border-radius:6px; padding:5px 9px; font-weight:bold;")
        lv.addWidget(self.stale_banner)

        # 数据集总览 (KPI cards, 4 per row)
        self.kpi_labels = {}
        kpi_grid = QGridLayout()
        kpi_grid.setHorizontalSpacing(6)
        kpi_grid.setVerticalSpacing(6)
        for column in range(4):
            kpi_grid.setColumnStretch(column, 1)
        for i, (key, title, hl) in enumerate(self.KPI_CARDS):
            kpi_grid.addWidget(self._make_card(key, title, hl), i // 4, i % 4)
        n = len(self.KPI_CARDS)
        kpi_grid.addWidget(self._make_mvp_card(), n // 4, n % 4)
        lv.addLayout(kpi_grid)

        self.baseline_hint = QLabel("")
        self.baseline_hint.setStyleSheet(MUTED_TEXT_STYLE)
        self.baseline_hint.setWordWrap(True)
        lv.addWidget(self.baseline_hint)

        # Filter box
        filt = QHBoxLayout()
        filt.setSpacing(5)
        filt.addWidget(QLabel("筛选:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("按 名称 / robot_type / 上传者 过滤…")
        self.filter_edit.textChanged.connect(
            self._guarded("筛选看板", self._apply_filter))
        filt.addWidget(self.filter_edit)
        self.only_issues = QCheckBox("只看有问题的")
        self.only_issues.toggled.connect(
            self._guarded("筛选问题数据集", self._apply_filter))
        filt.addWidget(self.only_issues)
        lv.addLayout(filt)

        # 数据集详情 (table)
        self.table = FrozenDatasetTable(
            0, len(TABLE_COLS), frozen_width=self.dataset_column_width)
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.horizontalHeaderItem(LOCAL_COL).setToolTip(
            "本地文件表示原始数据是否已下载到 datasets/<组织名>/，已下载的数据集可在 Viewer 打开。")
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(
            self._guarded("打开检查报告入口", self._on_table_cell_clicked, pass_args=True))
        self.table.cellDoubleClicked.connect(
            self._guarded("打开 HF 页面", self._open_row_link, pass_args=True))
        self.table.itemSelectionChanged.connect(
            self._guarded("刷新数据集详情面板", self._on_dataset_selected))
        hdr = self.table.detail.horizontalHeader()
        for i in range(self.table.detail.columnCount()):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.table.setColumnWidth(0, self.dataset_column_width)
        lv.addWidget(self.table, 1)
        self.table_hint = QLabel("点「仅拉取统计信息」加载数据集列表(双击行打开 HF 页面)。")
        self.table_hint.setWordWrap(True)
        lv.addWidget(self.table_hint)

        # ===== RIGHT: 数据集检查分区 =====
        right = QGroupBox("数据集检查分区")
        right.setStyleSheet(GREEN_PANEL_STYLE)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 6)
        rv.setSpacing(5)
        rv.addWidget(self._build_prompt_panel())

        split.addWidget(left)
        split.addWidget(right)
        # 数据集统计分区与数据集检查分区默认等宽，便于同时查看列表和详情。
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setCollapsible(1, True)
        # QSplitter otherwise gives the wide table panel most of the initial
        # width based on sizeHint(). Apply an explicit 1:1 ratio after the
        # parent has been laid out so both main sections start at half width.
        def equalize_main_splitter():
            width = split.width()
            if width > 1:
                split.setSizes([width // 2, width - width // 2])
        QTimer.singleShot(
            0, self._guarded("初始化主分割栏", equalize_main_splitter))
        outer.addWidget(split)
        return w

    @staticmethod
    def _panel_tree():
        t = QTreeWidget()
        t.setHeaderHidden(True)
        t.setWordWrap(True)
        t.setRootIsDecorated(False)
        return t

    @staticmethod
    def _block_scroll(box):
        """Wrap a detail block in a frameless scroll area.

        A splitter never shrinks a child below its minimum size hint, so a
        block with wide content (e.g. the 检查规则 button row) pins its drag
        handles in place. Behind a scroll area the block can shrink to any
        size; overflowing content just scrolls."""
        sa = QScrollArea()
        sa.setWidget(box)
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setStyleSheet("QScrollArea{background:transparent;}")
        sa.viewport().setStyleSheet("background:transparent;")
        return sa

    def _build_prompt_panel(self):
        """Right-side detail panel, laid out as a grid that mirrors the viewer's
        tabs: ANNOTATIONS / STATISTICS / FILTERING / ACTION INSIGHTS.

        ANNOTATIONS shows the local task instruction + viewer annotations;
        STATISTICS / FILTERING / ACTION INSIGHTS are filled from the viewer
        /report analysis (fetched async), so the key info is visible without
        opening the viewer WebUI."""
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(4, 0, 0, 0)
        pv.setSpacing(5)

        self.prompt_meta = QLabel("")
        self.prompt_meta.setStyleSheet(MUTED_TEXT_STYLE)
        self.prompt_meta.setWordWrap(True)
        pv.addWidget(self.prompt_meta)

        # Indeterminate marquee shown while the /report analysis runs.
        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 0)  # 0..0 = animated indeterminate
        self.report_progress.setTextVisible(False)
        self.report_progress.setMaximumHeight(6)
        self.report_progress.setVisible(False)
        pv.addWidget(self.report_progress)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(5)
        mode_row.addWidget(QLabel("右侧视图:"))
        self.data_check_mode_button = QPushButton("数据检查")
        self.doctor_mode_button = QPushButton("Doctor")
        for button in (self.data_check_mode_button, self.doctor_mode_button):
            button.setCheckable(True)
            button.setAutoExclusive(True)
        self.data_check_mode_button.setChecked(True)
        self.data_check_mode_button.clicked.connect(
            self._guarded("切换数据检查视图", lambda: self._set_detail_mode("checks")))
        self.doctor_mode_button.clicked.connect(
            self._guarded("切换 Doctor 视图", lambda: self._set_detail_mode("doctor")))
        mode_row.addWidget(self.data_check_mode_button)
        mode_row.addWidget(self.doctor_mode_button)
        mode_row.addStretch(1)
        pv.addLayout(mode_row)

        # --- grid of viewer-mirroring panels --------------------------------
        self.detail_grid = QWidget()
        grid = QGridLayout(self.detail_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 2)  # ANNOTATIONS            | STATISTICS(rowspan2)
        grid.setRowStretch(1, 2)  # FILTERING              | STATISTICS cont
        grid.setRowStretch(2, 3)  # ACTION INSIGHTS         | 检查规则

        # 浅绿色分组样式 — 与右侧「数据集检查分区」保持一致
        green_box_css = GREEN_PANEL_STYLE

        # ANNOTATIONS 标注 — local task instruction + viewer language annotations
        ann_box = QGroupBox("ANNOTATIONS 标注")
        ann_box.setStyleSheet(green_box_css)
        al = QVBoxLayout(ann_box)
        al.addWidget(QLabel("Language Instruction"))
        self.task_list = QListWidget()
        self.task_list.setWordWrap(True)
        self.task_list.setMaximumHeight(84)
        al.addWidget(self.task_list)
        self.task_note = QLabel("")
        self.task_note.setStyleSheet(MUTED_TEXT_STYLE)
        self.task_note.setWordWrap(True)
        al.addWidget(self.task_note)
        anno_hd = QLabel("语言标注 (viewer)")
        anno_hd.setStyleSheet(f"color:{UI_COLORS['text']};")
        al.addWidget(anno_hd)
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("集:"))
        self.prompt_ep = QComboBox()
        self.prompt_ep.currentIndexChanged.connect(
            self._guarded("刷新语言标注", self._refresh_prompt_tree))
        ep_row.addWidget(self.prompt_ep, 1)
        self.prompt_ep_wrap = QWidget()
        self.prompt_ep_wrap.setLayout(ep_row)
        al.addWidget(self.prompt_ep_wrap)
        self.prompt_tree = self._panel_tree()
        self.prompt_tree.setRootIsDecorated(True)
        al.addWidget(self.prompt_tree, 1)
        self.anno_note = QLabel("")
        self.anno_note.setStyleSheet(MUTED_TEXT_STYLE)
        self.anno_note.setWordWrap(True)
        al.addWidget(self.anno_note)
        grid.addWidget(ann_box, 0, 0)

        # STATISTICS 统计信息 — dataset/episode stats (from report)
        stat_box = QGroupBox("STATISTICS 统计信息")
        sl = QVBoxLayout(stat_box)
        episode_title = QLabel("Episode 时长分布")
        episode_title.setStyleSheet("font-weight:bold; color:#555;")
        sl.addWidget(episode_title)
        self.episode_length_tree = QTreeWidget()
        self.episode_length_tree.setColumnCount(3)
        self.episode_length_tree.setHeaderLabels(
            ["时长区间 / Episode", "数量 / 时长", "Frames"])
        self.episode_length_tree.setRootIsDecorated(True)
        self.episode_length_tree.setAlternatingRowColors(True)
        self.episode_length_tree.setUniformRowHeights(True)
        episode_header = self.episode_length_tree.header()
        episode_header.setSectionResizeMode(0, QHeaderView.Stretch)
        episode_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        episode_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        sl.addWidget(self.episode_length_tree, 1)

        # 数据集概要放在时长明细下方，默认折叠，避免挤占 episode 分布空间。
        summary_box = QGroupBox("数据集概要")
        summary_box.setCheckable(True)
        summary_box.setChecked(False)
        summary_box.setStyleSheet(
            "QGroupBox{background:#ffffff; border:1px solid #d4e7d4;"
            " border-radius:4px; margin-top:8px; padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin; left:8px;"
            " color:#2e7d32; font-weight:bold;}")
        summary_layout = QVBoxLayout(summary_box)
        self.stat_view = QLabel("")
        self.stat_view.setWordWrap(True)
        self.stat_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stat_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.stat_view.setStyleSheet(
            "background:#ffffff; border:1px solid #d4e7d4; padding:2px;")
        summary_layout.addWidget(self.stat_view)
        summary_box.toggled.connect(self.stat_view.setVisible)
        self.stat_view.setVisible(False)
        sl.addWidget(summary_box)
        grid.addWidget(stat_box, 0, 1, 2, 1)

        # 检查规则 — custom quality-check results for the selected dataset
        rules_box = QGroupBox("检查规则")
        rules_box.setStyleSheet(green_box_css)
        rul = QVBoxLayout(rules_box)
        quality_actions = QGridLayout()
        self.btn_quality_check = QPushButton("执行深度检查")
        self.btn_quality_check.setToolTip(
            "检查本地数据；未下载时按配置只缓存检查所需的远程文件。")
        self.btn_quality_check.clicked.connect(
            self._guarded("执行深度检查", self.on_quality_check))
        self.btn_quality_cancel = QPushButton("取消")
        self.btn_quality_cancel.setEnabled(False)
        self.btn_quality_cancel.clicked.connect(
            self._guarded("取消深度检查", self.on_quality_cancel))
        self.btn_open_quality_report = QPushButton("打开报告")
        self.btn_open_quality_report.clicked.connect(
            self._guarded("打开检查报告", self.on_open_quality_report))
        self.btn_export_quality_report = QPushButton("导出 ZIP")
        self.btn_export_quality_report.clicked.connect(
            self._guarded("导出检查报告", self.on_export_quality_report))
        self.btn_clear_quality_reports = QPushButton("清理报告")
        self.btn_clear_quality_reports.clicked.connect(
            self._guarded("清理检查报告", self.on_clear_quality_reports))
        self.btn_clear_quality_cache = QPushButton("清理缓存")
        self.btn_clear_quality_cache.clicked.connect(
            self._guarded("清理检查缓存", self.on_clear_quality_cache))
        for index, button in enumerate((
            self.btn_quality_check, self.btn_quality_cancel,
            self.btn_open_quality_report, self.btn_export_quality_report,
            self.btn_clear_quality_reports, self.btn_clear_quality_cache,
        )):
            quality_actions.addWidget(button, index // 2, index % 2)
        quality_actions.setColumnStretch(0, 1)
        quality_actions.setColumnStretch(1, 1)
        rul.addLayout(quality_actions)
        self.quality_progress = QProgressBar()
        self.quality_progress.setRange(0, 100)
        self.quality_progress.setVisible(False)
        rul.addWidget(self.quality_progress)
        self.quality_note = QLabel("")
        self.quality_note.setWordWrap(True)
        self.quality_note.setStyleSheet("color:#777; font-size:11px;")
        rul.addWidget(self.quality_note)
        pico_row = QHBoxLayout()
        self.pico_check_button = QPushButton("检查 PICO MoTracker 轨迹")
        self.pico_check_button.clicked.connect(
            self._guarded("检查 PICO MoTracker 轨迹", self._start_pico_check))
        pico_row.addWidget(self.pico_check_button)
        self.pico_check_status = QLabel("未检测")
        self.pico_check_status.setStyleSheet("color:#888; font-size:11px;")
        self.pico_check_status.setWordWrap(True)
        pico_row.addWidget(self.pico_check_status, 1)
        rul.addLayout(pico_row)
        self.check_tree = self._panel_tree()
        self.check_tree.setRootIsDecorated(True)
        rul.addWidget(self.check_tree, 1)
        self.quality_overview = QLabel("")
        self.quality_overview.setWordWrap(True)
        self.quality_overview.setStyleSheet("color:#777; font-size:11px;")
        rul.addWidget(self.quality_overview)
        review_actions = QHBoxLayout()
        for label in ("确认问题", "误报", "已修复", "未确认"):
            button = QPushButton(label)
            button.clicked.connect(
                self._guarded(
                    f"标记问题为{label}",
                    lambda status=label: self.on_mark_quality_issue(status)))
            review_actions.addWidget(button)
        rul.addLayout(review_actions)
        self.quality_issue_tree = QTreeWidget()
        self.quality_issue_tree.setHeaderLabels(
            ["#", "episode", "问题", "字段", "时间", "确认状态"])
        self.quality_issue_tree.setRootIsDecorated(False)
        self.quality_issue_tree.setAlternatingRowColors(True)
        self.quality_issue_tree.setMinimumHeight(80)
        rul.addWidget(self.quality_issue_tree)
        grid.addWidget(rules_box, 2, 1)

        # FILTERING 过滤器 — smoothness "Overall" verdict + breakdown lines
        filt_box = QGroupBox("FILTERING 过滤器")
        fl = QVBoxLayout(filt_box)
        self.filter_view = QLabel("")
        self.filter_view.setWordWrap(True)
        self.filter_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.filter_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.filter_view.setStyleSheet(
            "background:#ffffff; border:1px solid #d4e7d4;"
            " border-radius:2px; padding:6px;")
        fl.addWidget(self.filter_view, 1)
        grid.addWidget(filt_box, 1, 0)

        # FRAMES 首位帧暂未实现，先停用。
        # frames_box = QGroupBox("FRAMES 首位帧")
        # frl = QVBoxLayout(frames_box)
        # ph = QLabel("占位，暂未实现")
        # frl.addWidget(ph, 1)
        # grid.addWidget(frames_box, 2, 1)

        # ACTION INSIGHTS 行动指导与训练配置 — training config (report)
        insight_box = QGroupBox("ACTION INSIGHTS 行动指导与训练配置")
        il = QVBoxLayout(insight_box)
        self.insight_view = QLabel("")
        self.insight_view.setWordWrap(True)
        self.insight_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.insight_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        il.addWidget(self.insight_view, 1)
        grid.addWidget(insight_box, 2, 0)

        self.detail_stack = QStackedWidget()
        self.detail_stack.addWidget(self.detail_grid)
        self.doctor_panel = self._build_doctor_panel()
        self.detail_stack.addWidget(self.doctor_panel)
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)
        self.detail_scroll.setMinimumHeight(180)
        self.detail_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.detail_scroll.setStyleSheet("QScrollArea{background:transparent;}")
        self.detail_scroll.viewport().setStyleSheet("background:transparent;")
        self.detail_scroll.setWidget(self.detail_stack)
        pv.addWidget(self.detail_scroll, 1)

        # --- Fallback: nothing selected -------------------------------------
        self.prompt_empty = QLabel("选择左侧数据集查看信息。")
        self.prompt_empty.setStyleSheet(MUTED_TEXT_STYLE)
        self.prompt_empty.setWordWrap(True)
        self.prompt_empty.setAlignment(Qt.AlignCenter)
        self.prompt_empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        pv.addWidget(self.prompt_empty, 1)

        self._prompt_doc = {"episodes": {}, "updated_at": None}
        self._show_prompt_empty("选择左侧数据集查看信息。")
        return panel

    def _set_detail_mode(self, mode):
        """Switch the right-side detail area between checks and Doctor."""
        if mode == "doctor":
            self.detail_stack.setCurrentWidget(self.doctor_panel)
        else:
            self.detail_stack.setCurrentWidget(self.detail_grid)

    def _build_doctor_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Doctor 数据集诊断")
        title.setStyleSheet("font-size:16px; font-weight:bold; color:#2e7d32;")
        layout.addWidget(title)
        self.doctor_dataset_label = QLabel("请选择已下载的数据集")
        self.doctor_dataset_label.setStyleSheet("color:#777;")
        self.doctor_dataset_label.setWordWrap(True)
        layout.addWidget(self.doctor_dataset_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("检查范围:"))
        self.doctor_scope = QComboBox()
        self.doctor_scope.addItem("前 25 个 Episode", {"maxEpisodes": 25})
        self.doctor_scope.addItem("前 50 个 Episode", {"maxEpisodes": 50})
        self.doctor_scope.addItem("前 100 个 Episode", {"maxEpisodes": 100})
        self.doctor_scope.addItem("全部 Episode", {"maxEpisodes": None})
        self.doctor_scope.addItem("自定义 Episode 范围", {"episodeRange": True})
        self.doctor_scope.currentIndexChanged.connect(
            self._guarded("切换 Doctor 范围", self._doctor_scope_changed))
        controls.addWidget(self.doctor_scope, 1)
        self.doctor_range_start = QSpinBox()
        self.doctor_range_start.setRange(0, 1_000_000)
        self.doctor_range_start.setValue(0)
        self.doctor_range_start.setPrefix("起始 ")
        self.doctor_range_start.setVisible(False)
        controls.addWidget(self.doctor_range_start)
        self.doctor_range_end = QSpinBox()
        self.doctor_range_end.setRange(0, 1_000_000)
        self.doctor_range_end.setValue(25)
        self.doctor_range_end.setPrefix("结束 ")
        self.doctor_range_end.setVisible(False)
        controls.addWidget(self.doctor_range_end)
        self.doctor_run_button = QPushButton("运行 Doctor")
        self.doctor_run_button.clicked.connect(
            self._guarded("运行 Doctor", self._start_doctor))
        controls.addWidget(self.doctor_run_button)
        self.doctor_export_button = QPushButton("导出 JSON")
        self.doctor_export_button.setEnabled(False)
        self.doctor_export_button.clicked.connect(
            self._guarded("导出 Doctor JSON", self._export_doctor))
        controls.addWidget(self.doctor_export_button)
        layout.addLayout(controls)

        self.doctor_progress = QProgressBar()
        self.doctor_progress.setRange(0, 100)
        self.doctor_progress.setValue(0)
        self.doctor_progress.setFormat("未运行")
        layout.addWidget(self.doctor_progress)
        self.doctor_status = QLabel("Doctor 只在点击按钮后运行。")
        self.doctor_status.setStyleSheet("color:#888; font-size:11px;")
        self.doctor_status.setWordWrap(True)
        layout.addWidget(self.doctor_status)

        summary = QHBoxLayout()
        self.doctor_pass = QLabel("PASS 0")
        self.doctor_warn = QLabel("WARN 0")
        self.doctor_fail = QLabel("FAIL 0")
        self.doctor_pass.setStyleSheet("color:#2e7d32; font-weight:bold;")
        self.doctor_warn.setStyleSheet("color:#ef8c00; font-weight:bold;")
        self.doctor_fail.setStyleSheet("color:#c62828; font-weight:bold;")
        summary.addWidget(self.doctor_pass)
        summary.addWidget(self.doctor_warn)
        summary.addWidget(self.doctor_fail)
        summary.addStretch(1)
        layout.addLayout(summary)

        self.doctor_tree = self._panel_tree()
        self.doctor_tree.setRootIsDecorated(True)
        self.doctor_tree.setWordWrap(True)
        layout.addWidget(self.doctor_tree, 1)
        return panel

    def _doctor_scope_changed(self, _index):
        custom = bool((self.doctor_scope.currentData() or {}).get("episodeRange"))
        self.doctor_range_start.setVisible(custom)
        self.doctor_range_end.setVisible(custom)

    def _doctor_scope_options(self):
        selected = dict(self.doctor_scope.currentData() or {"maxEpisodes": 25})
        if selected.pop("episodeRange", False):
            start = self.doctor_range_start.value()
            end = max(start, self.doctor_range_end.value())
            selected = {
                "maxEpisodes": None,
                "episodeRange": {"start": start, "end": end},
            }
        return selected

    def _make_card(self, key, title, highlight=False):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setMinimumHeight(66)
        if highlight:
            # 总小时数 — the key metric, visually distinct.
            card.setStyleSheet(
                "QFrame{background:#E8F5E9; border:1px solid #66BB6A;"
                " border-radius:6px;}")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(8, 6, 8, 6)
        cv.setSpacing(3)
        t = QLabel(title)
        t.setStyleSheet(
            "color:#1B5E20; font-size:9pt; font-weight:bold;" if highlight
            else f"color:{UI_COLORS['text_muted']}; font-size:9pt;")
        val = QLabel("—")
        val.setStyleSheet(
            "font-size:19pt; font-weight:bold; color:#237A36;" if highlight
            else "font-size:17pt; font-weight:bold;")
        val.setWordWrap(True)
        val.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cv.addWidget(t)
        cv.addWidget(val)
        self.kpi_labels[key] = val
        return card

    def _make_mvp_card(self):
        """Special card: today's top contributor (by new hours) + their tallies."""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setMinimumHeight(66)
        cv = QVBoxLayout(card)
        cv.setContentsMargins(8, 6, 8, 6)
        cv.setSpacing(3)
        t = QLabel("今日 MVP ⭐")
        t.setStyleSheet(f"color:{UI_COLORS['text_muted']}; font-size:9pt;")
        self.mvp_name_lbl = QLabel("—")
        self.mvp_name_lbl.setStyleSheet(
            "font-size:17pt; font-weight:bold; color:#B54708;")
        self.mvp_name_lbl.setWordWrap(True)
        self.mvp_name_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mvp_name_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mvp_sub_lbl = QLabel("")
        self.mvp_sub_lbl.setStyleSheet(
            f"color:{UI_COLORS['text_muted']}; font-size:8pt;")
        self.mvp_sub_lbl.setWordWrap(True)
        self.mvp_sub_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        cv.addWidget(t)
        cv.addWidget(self.mvp_name_lbl)
        cv.addWidget(self.mvp_sub_lbl)
        return card

    def _build_rollup_trends_box(self):
        w = QGroupBox("区间趋势")
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 6)
        self.trend_hint = QLabel("")
        self.trend_hint.setStyleSheet(MUTED_TEXT_STYLE)
        self.trend_hint.setWordWrap(True)
        v.addWidget(self.trend_hint)
        self.trend_plot = pg.PlotWidget(title="每日新增与累计小时数")
        self.trend_plot.showGrid(x=False, y=True, alpha=0.3)
        self.trend_plot.setMinimumHeight(160)
        self.trend_plot.getAxis("bottom").setHeight(32)
        self.trend_plot.getAxis("left").setLabel("小时")
        self.trend_cum_view = pg.ViewBox()
        self.trend_plot.scene().addItem(self.trend_cum_view)
        self.trend_plot.getAxis("right").linkToView(self.trend_cum_view)
        self.trend_cum_view.setXLink(self.trend_plot)
        self.trend_plot.hideAxis("right")
        self.trend_plot.getViewBox().sigResized.connect(
            self._guarded("同步趋势图坐标", self._sync_trend_cum_view))
        v.addWidget(self.trend_plot, 1)
        return w

    def _sync_trend_cum_view(self):
        if not hasattr(self, "trend_cum_view"):
            return
        main_vb = self.trend_plot.getViewBox()
        self.trend_cum_view.setGeometry(main_vb.sceneBoundingRect())
        self.trend_cum_view.linkedViewChanged(main_vb, self.trend_cum_view.XAxis)

    def _build_rollup_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("分组维度:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(list(ROLLUP_DIMS.keys()))
        self.dim_combo.currentTextChanged.connect(
            self._guarded("刷新分组统计", self._refresh_rollup))
        row.addWidget(self.dim_combo)
        row.addSpacing(12)
        row.addWidget(QLabel("时间范围:"))
        self.rollup_start_date = QDateEdit()
        self.rollup_start_date.setCalendarPopup(True)
        self.rollup_start_date.setDisplayFormat("yyyy-MM-dd")
        self.rollup_start_date.setToolTip("区间起始日期")
        row.addWidget(self.rollup_start_date)
        row.addWidget(QLabel("—"))
        self.rollup_end_date = QDateEdit()
        self.rollup_end_date.setCalendarPopup(True)
        self.rollup_end_date.setDisplayFormat("yyyy-MM-dd")
        self.rollup_end_date.setToolTip("区间结束日期")
        row.addWidget(self.rollup_end_date)
        self.rollup_apply_btn = QPushButton("应用")
        self.rollup_apply_btn.clicked.connect(
            self._guarded("应用分组统计范围", self._refresh_rollup))
        row.addWidget(self.rollup_apply_btn)
        self.rollup_reset_btn = QPushButton("全量")
        self.rollup_reset_btn.clicked.connect(
            self._guarded("重置分组统计范围", self._reset_rollup_range))
        row.addWidget(self.rollup_reset_btn)
        row.addStretch()
        v.addLayout(row)

        self.rollup_hint = QLabel("")
        self.rollup_hint.setStyleSheet("color:#888; font-size:12px;")
        self.rollup_hint.setWordWrap(True)
        v.addWidget(self.rollup_hint)

        self.rollup_splitter = QSplitter(Qt.Horizontal)
        self.rollup_splitter.setChildrenCollapsible(False)

        range_box = QGroupBox("区间内单组新增总时长")
        range_v = QVBoxLayout(range_box)
        self.range_group_hint = QLabel("请选择一个时间范围。")
        self.range_group_hint.setStyleSheet(MUTED_TEXT_STYLE)
        self.range_group_hint.setWordWrap(True)
        range_v.addWidget(self.range_group_hint)
        self.range_group_table = QTableWidget(0, 5)
        self.range_group_table.setHorizontalHeaderLabels(
            ["分组", "新增小时", "新增episodes", "数据集数", "占比%"])
        self.range_group_table.setSortingEnabled(True)
        self.range_group_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.range_group_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.range_group_table.verticalHeader().setVisible(False)
        range_hdr = self.range_group_table.horizontalHeader()
        range_hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            range_hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        range_v.addWidget(self.range_group_table, 1)
        self.rollup_plot = pg.PlotWidget(title="各分组小时数")
        self.rollup_plot.showGrid(x=False, y=True, alpha=0.3)
        range_v.addWidget(self.rollup_plot, 1)
        self.rollup_table = self.range_group_table
        self.rollup_splitter.addWidget(range_box)

        daily_group_box = QGroupBox("区间内按日明细")
        daily_group_v = QVBoxLayout(daily_group_box)
        self.daily_group_hint = QLabel("按 Hugging Face commit history 分日，展示所选时间范围内的每日新增小时。")
        self.daily_group_hint.setStyleSheet(MUTED_TEXT_STYLE)
        daily_group_v.addWidget(self.daily_group_hint)
        self.daily_group_table = QTableWidget(0, 5)
        self.daily_group_table.setHorizontalHeaderLabels(
            ["日期", "分组", "新增小时", "新增episodes", "数据集数"])
        self.daily_group_table.setSortingEnabled(True)
        self.daily_group_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.daily_group_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.daily_group_table.verticalHeader().setVisible(False)
        daily_hdr = self.daily_group_table.horizontalHeader()
        daily_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        daily_hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 5):
            daily_hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.rollup_daily_splitter = QSplitter(Qt.Vertical)
        self.rollup_daily_splitter.setChildrenCollapsible(False)
        self.rollup_daily_splitter.addWidget(self.daily_group_table)
        self.rollup_daily_splitter.addWidget(self._build_rollup_trends_box())
        self.rollup_daily_splitter.setStretchFactor(0, 5)
        self.rollup_daily_splitter.setStretchFactor(1, 2)
        self.rollup_daily_splitter.setSizes([500, 190])
        daily_group_v.addWidget(self.rollup_daily_splitter, 1)
        self.rollup_splitter.addWidget(daily_group_box)

        self.rollup_splitter.setStretchFactor(0, 3)
        self.rollup_splitter.setStretchFactor(1, 4)
        self.rollup_splitter.setSizes([620, 760])
        v.addWidget(self.rollup_splitter, 1)
        return w

    # ---- 数据集编辑 tab (edit prompt / rename → new copy, optional push) ---- #
    def _build_edit_tab(self):
        """数据集编辑 = 左「数据集详情表」(同看板) | 右「编辑 + lerobot 操作」.

        Left is the same dataset detail list as 看板 so a dataset can be picked
        here directly. Right has two families:
          * 改名 / 改 Prompt — workbench-native pyarrow edits (no lerobot); the
            heavy payload is hard-linked, output is a new datasets/ copy.
          * 数据集操作 — delete / split / merge / add-feature / remove-feature,
            delegated to lerobot's REAL dataset_tools via a subprocess runner.
        """
        w = QWidget()
        outer = QHBoxLayout(w)
        outer.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)

        # ===== LEFT: dataset detail table (mirrors 看板) =====
        left = QGroupBox("数据集详情（选中要编辑 / 操作的数据集）")
        left.setStyleSheet(BLUE_PANEL_STYLE)
        lv = QVBoxLayout(left)
        ef = QHBoxLayout()
        ef.addWidget(QLabel("筛选:"))
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("按 名称 / robot_type / 上传者 过滤…")
        self.edit_filter.textChanged.connect(
            self._guarded("筛选编辑表格", self._apply_edit_filter))
        ef.addWidget(self.edit_filter)
        self.edit_only_downloaded = QCheckBox("只看已下载")
        self.edit_only_downloaded.setChecked(True)
        self.edit_only_downloaded.toggled.connect(
            self._guarded("筛选已下载数据集", self._apply_edit_filter))
        ef.addWidget(self.edit_only_downloaded)
        lv.addLayout(ef)

        self.edit_table = QTableWidget(0, len(TABLE_COLS))
        self.edit_table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.edit_table.horizontalHeaderItem(LOCAL_COL).setToolTip(
            "本地文件表示原始数据是否已下载到 datasets/<组织名>/，已下载的数据集可编辑或在 Viewer 打开。")
        self.edit_table.setSortingEnabled(True)
        self.edit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.edit_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.edit_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.edit_table.verticalHeader().setVisible(False)
        self.edit_table.itemSelectionChanged.connect(
            self._guarded("刷新编辑面板", self._refresh_edit_tab))
        ehdr = self.edit_table.horizontalHeader()
        ehdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for i in range(1, len(TABLE_COLS)):
            ehdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        ehdr.setStretchLastSection(True)
        self.edit_table.setColumnWidth(0, max(300, self.dataset_column_width - 40))
        lv.addWidget(self.edit_table, 1)
        split.addWidget(left)

        # ===== RIGHT: editing controls (scrollable) =====
        rscroll = QScrollArea()
        rscroll.setWidgetResizable(True)
        right = QWidget()
        rv = QVBoxLayout(right)

        self.edit_src_lbl = QLabel("—")
        self.edit_src_lbl.setStyleSheet("font-weight:bold;")
        self.edit_src_lbl.setWordWrap(True)
        self.edit_src_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.edit_src_lbl)

        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("输出数据集名:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("输出文件夹名（最后一段，例如 taccap-g1-...-0713）")
        nrow.addWidget(self.edit_name, 1)
        rv.addLayout(nrow)
        rv.addWidget(self._hline())

        # ---- Group A: 改名 / 改 Prompt (workbench-native) ----
        boxA = QGroupBox("① 改名 / 改 Prompt（本地实现，生成新副本）")
        av = QVBoxLayout(boxA)
        self.edit_prompt_holder = QWidget()
        self.edit_prompt_box = QVBoxLayout(self.edit_prompt_holder)
        self.edit_prompt_box.setContentsMargins(0, 0, 0, 0)
        note = QLabel("（未选择数据集）")
        note.setStyleSheet(MUTED_TEXT_STYLE)
        self.edit_prompt_box.addWidget(note)
        av.addWidget(self.edit_prompt_holder)
        arow = QHBoxLayout()
        self.btn_make_copy = QPushButton("生成新副本")
        self.btn_make_copy.setMinimumHeight(32)
        self.btn_make_copy.setStyleSheet(
            "QPushButton { font-weight:bold; padding:6px 16px; border-radius:6px;"
            f" color:white; background:{UI_COLORS['green']}; }}"
            f"QPushButton:hover {{ background:{UI_COLORS['green_hover']}; }}"
            "QPushButton:disabled { background:#B0B0B0; }")
        self.btn_make_copy.clicked.connect(
            self._guarded("生成数据集副本", self.on_make_copy))
        self.btn_push_copy = QPushButton("推送到 Hub")
        self.btn_push_copy.setMinimumHeight(32)
        self.btn_push_copy.setStyleSheet(
            f"QPushButton {{ padding:6px 12px; border-radius:5px; color:{UI_COLORS['text']};"
            f" border:1px solid {UI_COLORS['border_strong']}; background:{UI_COLORS['surface']}; }}"
            "QPushButton:hover { background:#F2F4F7; border-color:#98A2B3; }"
            f"QPushButton:disabled {{ color:{UI_COLORS['text_disabled']}; }}")
        self.btn_push_copy.clicked.connect(
            self._guarded("推送数据集副本", self.on_push_copy))
        arow.addWidget(self.btn_make_copy)
        arow.addWidget(self.btn_push_copy)
        arow.addStretch()
        av.addLayout(arow)
        rv.addWidget(boxA)

        # ---- Group B: lerobot 数据集操作 ----
        boxB = QGroupBox("② 数据集操作（lerobot：生成新输出，不改源目录）")
        bv = QVBoxLayout(boxB)
        oprow = QHBoxLayout()
        oprow.addWidget(QLabel("操作:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(
            ["删除 episodes", "拆分数据集", "合并数据集", "增加特征", "删除特征"])
        self.op_combo.currentIndexChanged.connect(
            self._guarded("切换数据集操作", self._on_op_changed, pass_args=True))
        oprow.addWidget(self.op_combo, 1)
        bv.addLayout(oprow)
        self.op_stack = QStackedWidget()
        self.op_stack.addWidget(self._build_op_delete())
        self.op_stack.addWidget(self._build_op_split())
        self.op_stack.addWidget(self._build_op_merge())
        self.op_stack.addWidget(self._build_op_addfeat())
        self.op_stack.addWidget(self._build_op_rmfeat())
        bv.addWidget(self.op_stack)
        self.btn_run_op = QPushButton("执行操作（生成新数据集）")
        self.btn_run_op.setMinimumHeight(32)
        self.btn_run_op.setStyleSheet(
            "QPushButton { font-weight:bold; padding:6px 16px; border-radius:6px;"
            f" color:white; background:{UI_COLORS['blue']}; }}"
            f"QPushButton:hover {{ background:{UI_COLORS['blue_hover']}; }}"
            "QPushButton:disabled { background:#B0B0B0; }")
        self.btn_run_op.clicked.connect(
            self._guarded("执行数据集操作", self.on_run_op))
        bv.addWidget(self.btn_run_op)
        self.op_note = QLabel(
            "所有操作只写到新的 datasets/<组织名>/<输出名> 目录；若目标已存在会停止，源数据目录不覆盖。")
        self.op_note.setStyleSheet("color:#888; font-size:12px;")
        self.op_note.setWordWrap(True)
        bv.addWidget(self.op_note)
        rv.addWidget(boxB)

        self.edit_result = QLabel("")
        self.edit_result.setWordWrap(True)
        self.edit_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.edit_result.setStyleSheet(f"color:{UI_COLORS['green']};")
        rv.addWidget(self.edit_result)
        rv.addStretch()

        rscroll.setWidget(right)
        split.addWidget(rscroll)
        split.setSizes([1080, 640])
        outer.addWidget(split)

        self._set_edit_enabled(False)
        return w

    def _hline(self):
        ln = QFrame()
        ln.setFrameShape(QFrame.HLine)
        ln.setFrameShadow(QFrame.Sunken)
        return ln

    # ---- op sub-forms (fields stashed on self; rebuilt lists on selection) ----
    def _build_op_delete(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("要从新输出中排除的 episode 序号（源数据不改，逗号分隔，如 0,2,5）:"))
        self.op_del_indices = QLineEdit()
        self.op_del_indices.setPlaceholderText("0,2,5")
        v.addWidget(self.op_del_indices)
        return w

    def _build_op_split(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("拆分方式（生成 <输出名>_<split>，源数据不改；比例或序号区间）:"))
        self.op_split_spec = QLineEdit()
        self.op_split_spec.setPlaceholderText("train:0.8,val:0.2  或  train:0-4,val:5-6")
        v.addWidget(self.op_split_spec)
        tip = QLabel("输出为 <输出名>_train / <输出名>_val 等，写到 datasets/<组织名>/。")
        tip.setStyleSheet("color:#888; font-size:12px;")
        tip.setWordWrap(True)
        v.addWidget(tip)
        return w

    def _build_op_merge(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("勾选要合并的数据集（需已下载，输出名用上方「输出数据集名」）:"))
        self.op_merge_list = QListWidget()
        self.op_merge_list.setMaximumHeight(160)
        v.addWidget(self.op_merge_list)
        return w

    def _build_op_addfeat(self):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QLabel("特征名:"), 0, 0)
        self.op_add_name = QLineEdit()
        self.op_add_name.setPlaceholderText("reward")
        g.addWidget(self.op_add_name, 0, 1)
        g.addWidget(QLabel("dtype:"), 1, 0)
        self.op_add_dtype = QComboBox()
        self.op_add_dtype.addItems(["float32", "float64", "int64"])
        g.addWidget(self.op_add_dtype, 1, 1)
        g.addWidget(QLabel("shape:"), 2, 0)
        self.op_add_shape = QLineEdit("1")
        self.op_add_shape.setPlaceholderText("1  或  3（向量维度）")
        g.addWidget(self.op_add_shape, 2, 1)
        g.addWidget(QLabel("填充值:"), 3, 0)
        self.op_add_fill = QLineEdit("0")
        g.addWidget(self.op_add_fill, 3, 1)
        g.setColumnStretch(1, 1)
        return w

    def _build_op_rmfeat(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("要从新输出中移除的特征 / 相机（源数据不改）:"))
        self.op_rm_list = QListWidget()
        self.op_rm_list.setMaximumHeight(160)
        v.addWidget(self.op_rm_list)
        return w

    def _on_op_changed(self, idx):
        self.op_stack.setCurrentIndex(idx)

    def _apply_edit_filter(self):
        q = self.edit_filter.text().strip().lower()
        only_dl = self.edit_only_downloaded.isChecked()
        downloaded = self._downloaded_leaves()
        for row in range(self.edit_table.rowCount()):
            item = self.edit_table.item(row, 0)
            d = item.data(Qt.UserRole) if item else {}
            d = d or {}
            hay = " ".join(str(d.get(k, "")) for k in
                           ("dataset_name", "robot_type", "uploader")).lower()
            hay += " " + uploader_cn(d.get("uploader")).lower()
            hide = bool(q) and q not in hay
            if not hide and only_dl:
                leaf = (d.get("dataset_name") or "").split("/")[-1]
                hide = leaf not in downloaded
            self.edit_table.setRowHidden(row, hide)

    def _set_edit_enabled(self, on):
        """Enable/disable the edit inputs (off when nothing editable is selected)."""
        for wdg in (self.edit_name, self.btn_make_copy, self.op_combo,
                    self.op_stack, self.btn_run_op):
            wdg.setEnabled(on)
        # Push only makes sense once a copy exists.
        self.btn_push_copy.setEnabled(on and self._last_copy_dir is not None)

    def _dataset_dir(self, d):
        """On-disk directory of a selected dataset, or None if not downloaded.

        Prefers the record's local_dir; otherwise resolves the flat
        datasets/TacVerse/<leaf>/ layout (with legacy date-layout fallback)."""
        local = (d or {}).get("local_dir")
        if local and (Path(local) / "meta" / "info.json").is_file():
            return Path(local)
        leaf = ((d or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        cands = [p.parent.parent for p in
                 Path(OUT_DIR).glob(f"{leaf}/meta/info.json")]
        cands += [p.parent.parent for p in
                   Path(OUT_DIR).glob(f"*/{leaf}/meta/info.json")]
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

    def _clear_prompt_edits(self):
        """Tear down the per-selection prompt editor rows."""
        self._prompt_edits = []
        while self.edit_prompt_box.count():
            item = self.edit_prompt_box.takeAt(0)
            child = item.widget()
            if child:
                child.deleteLater()

    def _selected_edit_dataset(self):
        """The dataset dict selected in the 数据集编辑 tab's own table, or None."""
        row = self.edit_table.currentRow()
        if row < 0:
            return None
        item = self.edit_table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _downloaded_dataset_dirs(self):
        """{leaf: newest dir} for every dataset under datasets/TacVerse/."""
        out = {}
        for info in Path(OUT_DIR).glob("*/meta/info.json"):
            ddir = info.parent.parent
            leaf = ddir.name
            prev = out.get(leaf)
            if prev is None or ddir.stat().st_mtime > prev.stat().st_mtime:
                out[leaf] = ddir
        for info in Path(OUT_DIR).glob("*/*/meta/info.json"):
            ddir = info.parent.parent
            leaf = ddir.name
            prev = out.get(leaf)
            if prev is None or ddir.stat().st_mtime > prev.stat().st_mtime:
                out[leaf] = ddir
        return out

    def _refresh_edit_tab(self):
        """Sync the edit tab to its table selection (called on select)."""
        # A copy is tied to the previously-selected source; reset on change.
        self._last_copy_dir = None
        self._last_copy_leaf = None
        self.edit_result.setText("")
        self._clear_prompt_edits()
        self._refresh_merge_list()

        d = self._selected_edit_dataset()
        src_dir = self._dataset_dir(d) if d else None
        self._edit_src = d
        if not d or src_dir is None:
            self.edit_src_lbl.setText("—")
            self.edit_name.setText("")
            note = QLabel("请选择一个已下载(已拉取)的数据集（仅统计的行不能编辑）。")
            note.setStyleSheet(MUTED_TEXT_STYLE)
            self.edit_prompt_box.addWidget(note)
            self.op_rm_list.clear()
            self._set_edit_enabled(False)
            return

        name = d.get("dataset_name") or ""
        leaf = name.split("/")[-1]
        self.edit_src_lbl.setText(f"{name}  ·  {src_dir}")
        self.edit_name.setText(leaf)  # default: same name (user changes to rename)

        rows, err = tsk.load(src_dir / tsk.TASKS_REL)
        if err:
            note = QLabel(f"无法读取指令: {err}")
            note.setStyleSheet("color:#c62828;")
            self.edit_prompt_box.addWidget(note)
        elif not rows:
            note = QLabel("该数据集没有 tasks.parquet（无可编辑指令），仍可改名生成副本。")
            note.setStyleSheet(MUTED_TEXT_STYLE)
            self.edit_prompt_box.addWidget(note)
        else:
            for r in rows:
                holder = QWidget()
                line = QHBoxLayout(holder)
                line.setContentsMargins(0, 0, 0, 0)
                line.addWidget(QLabel(f"#{r['index']}"))
                edit = QLineEdit(r["task"])
                line.addWidget(edit, 1)
                self.edit_prompt_box.addWidget(holder)
                self._prompt_edits.append((r["index"], r["task"], edit))

        self._refresh_rmfeat_list(src_dir)
        self._set_edit_enabled(True)

    def _refresh_merge_list(self):
        """Populate the merge candidate list with all downloaded datasets."""
        if not hasattr(self, "op_merge_list"):
            return
        self.op_merge_list.clear()
        for leaf, ddir in sorted(self._downloaded_dataset_dirs().items()):
            it = QListWidgetItem(leaf)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, str(ddir))
            self.op_merge_list.addItem(it)

    def _refresh_rmfeat_list(self, src_dir):
        """Populate the remove-feature list from the selected dataset's schema."""
        self.op_rm_list.clear()
        info = de.read_info(src_dir)
        required = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        for key in (info.get("features") or {}):
            if key in required:
                continue  # lerobot forbids removing these
            it = QListWidgetItem(key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.op_rm_list.addItem(it)

    def on_make_copy(self):
        d, src = self._edit_src, self._dataset_dir(self._edit_src)
        if not d or src is None:
            QMessageBox.warning(self, "提示", "请先选择一个已下载的数据集。")
            return
        try:
            new_leaf = de.validate_leaf(self.edit_name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "名字不合法", str(exc))
            return

        # Prompt replacements: old string -> edited string (only real changes).
        replacements = {}
        for _idx, old, edit in self._prompt_edits:
            new = edit.text().strip()
            if new and new != old:
                replacements[old] = new

        dst = de.default_copy_dir(new_leaf, OUT_DIR)
        if dst.exists():
            QMessageBox.warning(
                self, "目标已存在",
                f"{dst} 已存在。请换一个新名字，或先删除该目录。")
            return
        if not replacements and new_leaf == src.name:
            QMessageBox.information(self, "无改动", "指令和名字都没有变化，未生成副本。")
            return

        self._set_busy(True)
        self.status.setText(f"生成副本 {dst} ...")
        self.edit_result.setText("")
        self._edit_worker = EditWorker(str(src), str(dst), replacements)
        self._edit_worker.done.connect(
            self._guarded("生成数据集副本", self._on_edit_done, pass_args=True))
        self._edit_worker.error.connect(
            self._guarded("生成数据集副本", self._on_worker_error, pass_args=True))
        self._edit_worker.finished.connect(
            lambda worker=self._edit_worker:
            self._on_one_shot_worker_finished("_edit_worker", worker))
        self._edit_worker.start()

    def _on_edit_done(self, dst_dir, n_changed):
        self._last_copy_dir = Path(dst_dir)
        self._last_copy_leaf = Path(dst_dir).name
        self.btn_push_copy.setEnabled(True)
        note = f"已生成副本: {dst_dir}（修改 {n_changed} 条指令）"
        self.edit_result.setText(note)
        self.status.setText(note)
        self._refresh_table()  # the new copy now counts as 已下载
        QMessageBox.information(
            self, "完成",
            f"{note}\n\n数据/视频为硬链接（未额外占用磁盘）。\n"
            f"如需上传到 HuggingFace，点「推送到 Hub」。")

    def on_push_copy(self):
        if not self._last_copy_dir or not self._last_copy_dir.exists():
            QMessageBox.warning(self, "提示", "请先「生成新副本」。")
            return
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        if not self.token:
            QMessageBox.warning(self, "未登录",
                                "当前没有 HuggingFace token，无法上传。请先「切换账号」。")
            return
        repo_id = f"{org}/{self._last_copy_leaf}"
        ok = QMessageBox.question(
            self, "确认上传",
            f"将把\n  {self._last_copy_dir}\n上传为 HuggingFace 数据集（私有）:\n"
            f"  {repo_id}\n\n确定继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        self._set_busy(True)
        self.status.setText(f"上传到 {repo_id} ...")
        self._push_worker = PushWorker(
            str(self._last_copy_dir), repo_id, self.token, private=True)
        self._push_worker.done.connect(
            self._guarded("推送数据集副本", self._on_push_done, pass_args=True))
        self._push_worker.error.connect(
            self._guarded("推送数据集副本", self._on_worker_error, pass_args=True))
        self._push_worker.finished.connect(
            lambda worker=self._push_worker:
            self._on_one_shot_worker_finished("_push_worker", worker))
        self._push_worker.start()

    def _on_push_done(self, url):
        msg = f"上传完成: {url}"
        self.edit_result.setText(msg)
        self.status.setText(msg)
        QMessageBox.information(self, "上传完成", msg)

    # ---- lerobot 操作 (delete / split / merge / add / remove) --------------- #
    @staticmethod
    def _parse_int_list(text):
        """'0,2,5' or '0-3,7' -> [0,2,5] / [0,1,2,3,7]. Raises ValueError."""
        out = []
        for part in text.replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    def _parse_splits(self, text):
        """'train:0.8,val:0.2' -> {train:0.8,val:0.2};
        'train:0-4,val:5-6' -> {train:[0..4], val:[5,6]}. Raises ValueError."""
        splits = {}
        for chunk in text.replace("，", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"格式应为 名字:值 —— '{chunk}'")
            name, val = chunk.split(":", 1)
            name, val = name.strip(), val.strip()
            if any(c in val for c in "-,") or (not _is_float(val)):
                # explicit index list (e.g. 0-4) grouped per split
                splits[name] = self._parse_int_list(val)
            else:
                splits[name] = float(val)
        kinds = {isinstance(v, float) for v in splits.values()}
        if len(kinds) > 1:
            raise ValueError("比例和序号区间不能混用")
        return splits

    def on_run_op(self):
        d, src = self._edit_src, self._dataset_dir(self._edit_src)
        op_idx = self.op_combo.currentIndex()
        org = self.org_combo.currentText().strip() or "TacVerse"

        # Merge does not require a selected row (it uses the checkbox list); the
        # others operate on the selected dataset.
        if op_idx != 2 and (not d or src is None):
            QMessageBox.warning(self, "提示", "请先在左表选择一个已下载的数据集。")
            return
        try:
            out_leaf = de.validate_leaf(self.edit_name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "输出名不合法", str(exc))
            return

        spec = {"vcodec": lops.DEFAULT_VCODEC, "params": {}}
        out_dir = lops.default_out_dir(out_leaf, OUT_DIR)
        src_repo = d.get("dataset_name") if d else f"{org}/{out_leaf}"

        try:
            if op_idx == 0:  # delete episodes
                idx = self._parse_int_list(self.op_del_indices.text())
                if not idx:
                    raise ValueError("请填写要删除的 episode 序号。")
                spec.update(op="delete", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["episode_indices"] = idx
            elif op_idx == 1:  # split
                splits = self._parse_splits(self.op_split_spec.text())
                if not splits:
                    raise ValueError("请填写拆分方式。")
                for split_name in splits:
                    try:
                        de.validate_leaf(split_name)
                    except ValueError as exc:
                        raise ValueError(
                            f"拆分名不合法: {split_name}（{exc}）") from exc
                spec.update(op="split", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_parent=str(out_dir.parent), out_leaf=out_leaf,
                            out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["splits"] = splits
            elif op_idx == 2:  # merge
                sources = []
                for i in range(self.op_merge_list.count()):
                    it = self.op_merge_list.item(i)
                    if it.checkState() == Qt.Checked:
                        sources.append({"repo_id": f"{org}/{it.text()}",
                                        "root": it.data(Qt.UserRole)})
                if len(sources) < 2:
                    raise ValueError("请至少勾选 2 个数据集进行合并。")
                spec.update(op="merge", sources=sources,
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
            elif op_idx == 3:  # add feature
                name = self.op_add_name.text().strip()
                if not name:
                    raise ValueError("请填写特征名。")
                shape = self._parse_int_list(self.op_add_shape.text() or "1")
                if not shape:
                    raise ValueError("shape 至少要有一个维度，如 1 或 3。")
                fill_txt = (self.op_add_fill.text() or "0").strip()
                dtype = self.op_add_dtype.currentText()
                fill = int(fill_txt) if dtype == "int64" else float(fill_txt)
                spec.update(op="add_feature", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"].update(name=name, dtype=dtype, shape=shape, fill=fill)
            elif op_idx == 4:  # remove feature
                names = [self.op_rm_list.item(i).text()
                         for i in range(self.op_rm_list.count())
                         if self.op_rm_list.item(i).checkState() == Qt.Checked]
                if not names:
                    raise ValueError("请勾选要删除的特征。")
                spec.update(op="remove_feature", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["feature_names"] = names
        except ValueError as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            return

        # Guard against clobbering existing output dirs (split writes siblings).
        if spec["op"] == "split":
            clashes = [p for name in spec["params"]["splits"]
                       if (p := out_dir.parent / f"{out_leaf}_{name}").exists()]
            if clashes:
                QMessageBox.warning(self, "目标已存在",
                                    "以下输出目录已存在，请换名：\n" +
                                    "\n".join(str(c) for c in clashes))
                return
        elif out_dir.exists():
            QMessageBox.warning(self, "目标已存在",
                                f"{out_dir} 已存在。请换一个输出名。")
            return

        self._set_busy(True)
        self.edit_result.setText("")
        self.status.setText(f"执行 {self.op_combo.currentText()} ...（视频操作较慢）")
        self._op_worker = LerobotOpWorker(spec)
        self._op_worker.log.connect(self.status.setText)
        self._op_worker.done.connect(
            self._guarded("执行数据集操作", self._on_op_done, pass_args=True))
        self._op_worker.error.connect(
            self._guarded("执行数据集操作", self._on_worker_error, pass_args=True))
        self._op_worker.finished.connect(
            lambda worker=self._op_worker:
            self._on_one_shot_worker_finished("_op_worker", worker))
        self._op_worker.start()

    def _on_op_done(self, result):
        outs = result.get("outputs", [])
        lines = [f"{o['repo_id']}  ({o['episodes']} eps / {o['frames']} frames)\n  → {o['root']}"
                 for o in outs]
        msg = f"{self.op_combo.currentText()} 完成，生成 {len(outs)} 个数据集:"
        self.edit_result.setText(msg + "\n" + "\n".join(lines))
        self.status.setText(msg)
        self._refresh_table()  # new outputs now count as 已下载
        QMessageBox.information(self, "完成", msg + "\n\n" + "\n".join(lines))

    # ---- Viewer tab (vendored xense_lerobot_viewer, black-box service) ---- #
    def _viewer_root(self):
        """The stable, flat organization dataset root scanned by Viewer."""
        return str(Path(OUT_DIR).resolve())

    def _build_viewer_tab(self):
        """Reserved space for the viewer: service status + controls.

        The viewer serves ALL its features over the web; this tab drives its
        lifecycle and opens it in the browser. The placeholder area is kept so
        a future phase can drop an embedded web view in without restructuring.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        self.viewer_status = QLabel("")
        self.viewer_status.setStyleSheet("font-size:11pt; font-weight:bold;")
        v.addWidget(self.viewer_status)
        self.viewer_detail = QLabel("")
        self.viewer_detail.setStyleSheet(MUTED_TEXT_STYLE)
        self.viewer_detail.setWordWrap(True)
        v.addWidget(self.viewer_detail)

        row = QHBoxLayout()
        self.viewer_start_btn = QPushButton("启动 Viewer")
        self.viewer_start_btn.clicked.connect(
            self._guarded("启动 Viewer", self._viewer_start))
        self.viewer_stop_btn = QPushButton("停止")
        self.viewer_stop_btn.clicked.connect(
            self._guarded("停止 Viewer", self._viewer_stop))
        self.viewer_home_btn = QPushButton("打开首页")
        self.viewer_home_btn.clicked.connect(
            self._guarded("打开 Viewer 首页", self._viewer_open_home))
        for b in (self.viewer_start_btn, self.viewer_stop_btn, self.viewer_home_btn):
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)

        self.viewer_placeholder = QLabel(
            "Viewer 以网页形式提供全部功能（数据集预览 / 健康检查 / 3D 回放 / 标注）。\n"
            "点「启动 Viewer」后，用「打开首页」，或在「看板」选中数据集点「🔍 在 Viewer 打开」。\n\n"
            "（此区域为预留：后续可在此内嵌网页视图）")
        self.viewer_placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_placeholder.setWordWrap(True)
        self.viewer_placeholder.setStyleSheet(
            f"color:{UI_COLORS['text_muted']}; border:1px dashed "
            f"{UI_COLORS['border']}; padding:24px;")
        v.addWidget(self.viewer_placeholder, 1)

        self._viewer_tick = 0
        self._viewer_count = None
        self.viewer_timer = QTimer(self)
        self.viewer_timer.timeout.connect(
            self._guarded("刷新 Viewer 状态", self._refresh_viewer_status))
        self.viewer_timer.start(2000)
        self._refresh_viewer_status()
        return w

    def _viewer_start(self):
        if not self.viewer.available():
            msg = f"Viewer 未就绪：{vsvc.install_hint(self.viewer.viewer_dir)}"
            self.viewer_detail.setText(msg)
            self.status.setText(msg)
            return
        ok, msg = self.viewer.start(self._viewer_root(), wait=False)
        self.status.setText(f"Viewer: {msg}")
        self._viewer_count = None
        self._refresh_viewer_status()

    def _viewer_stop(self):
        ok, msg = self.viewer.stop()
        self._viewer_count = None
        self.status.setText(msg)
        if not ok:
            QMessageBox.warning(self, "Viewer 关闭失败", msg)
        self._refresh_viewer_status()

    def _viewer_open_home(self):
        if not self.viewer.is_ready():
            self.status.setText("Viewer 尚未就绪：请先启动并等待状态变为运行中")
            return
        self.viewer.open_home()
        self.status.setText(f"已打开首页: {self.viewer.home_url()}")

    def _refresh_viewer_status(self):
        st = self.viewer.status()
        state = st.get("state", "stopped")
        if state == "ready":
            color, text = "#2e7d32", "运行中"
        elif state == "starting":
            color, text = "#F9A825", "启动中…"
        elif state == "conflict":
            color, text = "#c62828", "端口冲突"
        elif state == "error":
            color, text = "#c62828", "启动失败"
        else:
            color, text = "#667085", "已停止"

        # Refresh the dataset count occasionally (every ~6s) to avoid hammering
        # the discovery API on every tick.
        self._viewer_tick += 1
        if st["ready"] and self._viewer_tick % 3 == 0:
            self._viewer_count = self.viewer.dataset_count()
        elif not st["ready"]:
            self._viewer_count = None
        extra = f" · 可见数据集 {self._viewer_count}" if self._viewer_count is not None else ""

        # Toolbar controls (canonical).
        self.top_viewer_dot.setText(
            f'<span style="color:{color}">●</span> Viewer: {text} · {st["port"]}')
        detail = st.get("error") or (
            f'数据根: {st.get("actual_root") or st["root"] or self._viewer_root()}')
        self.top_viewer_dot.setToolTip(
            f"Viewer: {text} · 端口 {st['port']}\n{detail}")
        self.top_viewer_start.setEnabled(state in {"stopped", "error", "conflict"})
        self.top_viewer_stop.setEnabled(st["managed"])
        self.top_viewer_home.setEnabled(st["ready"])
        self.open_viewer_btn.setEnabled(st["ready"])

        # Keep the (soon-to-be-optional) Viewer tab in sync if it still exists.
        if hasattr(self, "viewer_status"):
            self.viewer_status.setText(
                f'<span style="color:{color}">●</span> Viewer: {text} · 端口 {st["port"]}')
            self.viewer_detail.setText(
                f'{detail}{extra}   ({st["url"]})')
            self.viewer_start_btn.setEnabled(
                state in {"stopped", "error", "conflict"})
            self.viewer_stop_btn.setEnabled(st["managed"])
            self.viewer_home_btn.setEnabled(st["ready"])

    def _open_selected_in_viewer(self):
        d = self._selected_dataset()
        if not d:
            self.status.setText("请先在左侧选中一个数据集")
            return
        if not self.viewer.is_ready():
            self.status.setText("Viewer 尚未就绪：请先启动并等待状态变为运行中")
            return
        rel = self.viewer.dataset_rel_path(d, root=self._viewer_root())
        if not rel:
            self.status.setText(
                f"该数据集不在数据根下（未拉取到 {OUT_DIR}/），无法在 Viewer 打开")
            return
        self.viewer.open_dataset(rel)
        self.status.setText(f"已在浏览器打开: {self.viewer.dataset_url(rel)}")

    # ---- Rendering -------------------------------------------------------- #
    def _refresh_all(self):
        self._refresh_kpis()
        self._refresh_table()
        self._set_rollup_range_defaults()
        self._refresh_rollup()

    def _show_stale_banner(self, snap):
        """Flag that the 看板/表格 currently show a past pull, not live data."""
        if snap.get("source") == "manual":
            self.stale_banner.setText(
                f"ℹ️ 当前显示 {fmt_day(snap.get('date'))} 手动补录的总统计。"
                "该快照不含逐数据集明细；今日新增按前一个有记录日期的累计值推导。")
            self.stale_banner.setVisible(True)
            return
        at = (snap.get("pulled_at") or "").replace("T", " ")
        when = f"（{at}）" if at else ""
        self.stale_banner.setText(
            f"⚠️ 当前显示的是上一次拉取结果{when}，并非实时最新。"
            "点「仅拉取统计信息」刷新为最新数据。")
        self.stale_banner.setVisible(True)

    def _hide_stale_banner(self):
        """Data is now live (a fresh 统计/拉取 just finished) — drop the flag."""
        self.stale_banner.setVisible(False)

    def _current_deltas(self):
        return dd.hf_last_modified_dataset_deltas(self.report, self.history, self.hf_changes)

    def _rollup_date_bounds(self):
        dates = set()
        for row in self.history or []:
            if row.get("date"):
                dates.add(row["date"])
        if self.report and self.report.get("date"):
            dates.add(self.report["date"])
        try:
            for row in dd.hf_change_rows(self.hf_changes):
                if row.get("date"):
                    dates.add(row["date"])
        except Exception:
            pass
        if not dates:
            return "", ""
        return min(dates), max(dates)

    def _set_rollup_range_defaults(self, force=False):
        if not hasattr(self, "rollup_start_date") or not hasattr(self, "rollup_end_date"):
            return
        if self._rollup_range_initialized and not force:
            return
        start, end = self._rollup_date_bounds()
        if start and end:
            start_q = qdate_from_yymmdd(start)
            end_q = qdate_from_yymmdd(end)
            if start_q.isValid() and end_q.isValid():
                self.rollup_start_date.setDate(start_q)
                self.rollup_end_date.setDate(end_q)
        self._rollup_range_initialized = True

    def _selected_rollup_range(self):
        if not hasattr(self, "rollup_start_date") or not hasattr(self, "rollup_end_date"):
            return "", ""
        start = self.rollup_start_date.date().toString("yyMMdd")
        end = self.rollup_end_date.date().toString("yyMMdd")
        if start and end and start > end:
            start, end = end, start
            self.rollup_start_date.setDate(qdate_from_yymmdd(start))
            self.rollup_end_date.setDate(qdate_from_yymmdd(end))
        return start, end

    def _reset_rollup_range(self):
        self._rollup_range_initialized = False
        self._set_rollup_range_defaults(force=True)
        self._refresh_rollup()

    def _refresh_baseline_hint(self):
        """Spell out the Hugging Face commit-diff basis for dashboard highlights."""
        totals = self._hf_update_totals()
        if not totals.get("date"):
            self.baseline_hint.setText(
                "暂无 HF last_modified 数据；执行「仅拉取统计信息」后按 HF 变更统计今日新增。")
            return
        if self.report.get("source") == "manual":
            self.baseline_hint.setText(
                f"「今日新增」= {fmt_day(totals['date'])} 的 dataset log 累计总量"
                "减去前一个有记录日期；手动快照不计算 MVP 和分组贡献。")
            return
        source = (
            "HF commit history 差分，未覆盖的数据集由 dataset log 兜底"
            if self._has_hf_change_rows() else
            "dataset log 快照兜底（刷新统计后会尝试生成 HF commit 差分缓存）"
        )
        self.baseline_hint.setText(
            f"「今日新增 / MVP」= HF last_modified 属于 {fmt_day(totals['date'])} 的增量；数值来源：{source}。")

    def _refresh_kpis(self):
        r = self.report
        if not r:
            for lbl in self.kpi_labels.values():
                lbl.setText("—")
            self.baseline_hint.setText("")
            self.mvp_name_lbl.setText("—")
            self.mvp_sub_lbl.setText("")
            return
        self._refresh_baseline_hint()
        hf_totals = self._hf_update_totals()
        self._refresh_mvp(hf_totals["date"])
        self.kpi_labels["total_datasets"].setText(fmt_value(r.get("total_datasets")))
        self.kpi_labels["total_hours"].setText(fmt_value(r.get("total_hours")))
        self.kpi_labels["total_episodes"].setText(fmt_value(r.get("total_episodes")))
        self.kpi_labels["total_frames"].setText(fmt_value(r.get("total_frames")))
        if not hf_totals.get("date"):
            self.kpi_labels["new_hours"].setText("—")
            self.kpi_labels["new_episodes"].setText("—")
            self.kpi_labels["completion"].setText("—")
            return
        new_hours, new_eps = hf_totals["hours"], hf_totals["episodes"]
        target = self.target_spin.value()
        pct = f"{round(100 * new_hours / target)}%" if target else "—"
        self.kpi_labels["new_hours"].setText(f"+{new_hours}")
        self.kpi_labels["new_episodes"].setText(f"+{fmt_value(new_eps)}")
        self.kpi_labels["completion"].setText(pct)

    def _hf_update_totals(self):
        return dd.hf_last_modified_totals(self.report, self.history, self.hf_changes)

    def _has_hf_change_rows(self):
        return dd.hf_report_has_matching_change_cache(
            self.report, self.hf_changes)

    def _refresh_mvp(self, date):
        """MVP by HF update day: top uploader among datasets updated that day."""
        if self.report and self.report.get("source") == "manual":
            self.mvp_name_lbl.setText("—")
            self.mvp_sub_lbl.setText("无逐数据集贡献信息，无法计算")
            return
        rows = dd.hf_last_modified_group_totals(
            self.report, self.history, self.hf_changes,
            lambda dataset: uploader_cn(dataset.get("uploader")), date)
        top = rows[0] if rows else None
        if not top or top["hours"] <= 0:
            self.mvp_name_lbl.setText("—")
            self.mvp_sub_lbl.setText("今日暂无新增贡献")
            return
        self.mvp_name_lbl.setText(top["group"])
        self.mvp_sub_lbl.setText(
            f"{fmt_day(top['date'])} · {top['hours']} 小时 · {fmt_value(top['episodes'])} episodes")

    def _downloaded_leaves(self):
        """Leaf names of datasets whose raw files are under datasets/<org>/.

        A dataset counts as downloaded when datasets/<org>/<leaf>/meta/info.json
        exists (a full 拉取 writes it; 统计-only never touches datasets/). Only these
        can be opened in the viewer. Scanned once per table refresh."""
        return set(self._downloaded_dataset_dirs())

    def _fill_dataset_table(self, table, datasets, deltas, downloaded):
        """Populate a QTableWidget with the dataset detail rows (shared by the
        看板 table and the 数据集编辑 table so both show the same list)."""
        table.setSortingEnabled(False)
        table.setRowCount(len(datasets))
        for row, d in enumerate(datasets):
            for col, (_, key, kind) in enumerate(TABLE_COLS):
                if key == "__local__":
                    leaf = (d.get("dataset_name") or "").split("/")[-1]
                    dl = leaf in downloaded
                    item = NumericItem("✅ 已下载" if dl else "—", 1 if dl else 0)
                    item.setToolTip(
                        "原始文件已下载到本地 datasets/<组织名>/，可在 Viewer 打开" if dl else
                        "未下载（仅统计信息）；先「拉取」才能在 Viewer 打开")
                    if dl:
                        item.setForeground(QBrush(QColor("#2e7d32")))
                elif key == "__delta__":
                    if deltas is None:
                        item = NumericItem("—", -1)
                        item.setToolTip("暂无 HF commit 变更历史缓存；执行「仅拉取统计信息」后显示真实提交差分。")
                        item.setForeground(QBrush(QColor("#9e9e9e")))
                        table.setItem(row, col, item)
                        continue
                    dv = deltas.get(d["dataset_name"], {})
                    n = dv.get("d_episodes", 0)
                    if n > 0:
                        txt, color = f"⬆ +{n}", "#2e7d32"
                    else:
                        txt, color = "➖ 0", "#9e9e9e"
                    item = NumericItem(txt, n)
                    item.setForeground(QBrush(QColor(color)))
                elif key == "__quality_status__":
                    name = d.get("dataset_name") or ""
                    item = self._quality_status_item(name)
                elif key == "__avg_sec__":
                    eps = d.get("total_episodes") or 0
                    hrs = d.get("duration_hours") or 0
                    v = round(hrs * 3600 / eps, 1) if eps else 0
                    item = NumericItem(fmt_value(v), v)
                elif key == "__check__":
                    results, agg = chk_mod.run_checks(d, cfg=_CHECKS_CFG)
                    txt, sort_key = chk_mod.badge(agg)
                    item = NumericItem(txt, sort_key)
                    item.setToolTip("\n".join(
                        f"{chk_mod.icon(x.status)} {x.title}: {x.message}"
                        for x in results))
                elif key == "__uploader_cn__":
                    item = QTableWidgetItem(uploader_cn(d.get("uploader")))
                elif kind == "num":
                    v = d.get(key)
                    item = NumericItem(fmt_value(v), v if isinstance(v, (int, float)) else -1)
                elif kind == "date":
                    v = d.get(key) or ""
                    # Show day granularity but sort by the full ISO timestamp
                    # (ISO strings sort chronologically), so the default 最后更新↓
                    # order reproduces HF's "Recently updated" ranking — same-day
                    # datasets keep their real order instead of shuffling.
                    item = NumericItem(v[:10] if v else "—", v or "")
                else:
                    item = QTableWidgetItem(fmt_value(d.get(key)))
                if col == 0:
                    item.setData(Qt.UserRole, d)  # stash the row's dataset dict
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        # Default order: most-recently-updated first (matches org page / 发现顺序).
        table.sortItems(DATE_COL, Qt.DescendingOrder)

    def _refresh_table(self):
        r = self.report
        datasets = r.get("datasets", []) if r else []
        deltas = self._current_deltas()
        downloaded = self._downloaded_leaves()  # dataset leaf names present in datasets/
        self._fill_dataset_table(self.table, datasets, deltas, downloaded)
        # Mirror the same list into the 数据集编辑 tab's table (if built).
        if hasattr(self, "edit_table"):
            self._fill_dataset_table(self.edit_table, datasets, deltas, downloaded)
            self._apply_edit_filter()
            self._refresh_merge_list()
        self.table_hint.setText(
            f"共 {len(datasets)} 个数据集，双击行打开 HF 页面；点表头排序。"
            if datasets else (
                "该快照仅包含手动总量，不含逐数据集明细。"
                if r and r.get("source") == "manual" else
                "点「仅拉取统计信息」加载数据集列表。"))
        self._apply_filter()

    def _quality_status_item(self, name):
        status = self._quality_status.get(name, "未检查")
        item = QTableWidgetItem(status)
        if status == "已检查":
            item.setForeground(QBrush(QColor("#2e7d32")))
            item.setToolTip(f"点击打开检查报告目录:\n{self._quality_reports.get(name, '')}")
        elif status == "检查中":
            item.setForeground(QBrush(QColor("#ef8c00")))
            item.setToolTip("深度检查正在后台执行")
        else:
            item.setForeground(QBrush(QColor("#777777")))
            item.setToolTip("尚未执行深度检查")
        return item

    def _update_quality_status_cells(self, name):
        for table in (getattr(self, "table", None), getattr(self, "edit_table", None)):
            if table is None:
                continue
            for row in range(table.rowCount()):
                first = table.item(row, 0)
                data = first.data(Qt.UserRole) if first else {}
                if (data or {}).get("dataset_name") == name:
                    table.setItem(row, QUALITY_STATUS_COL, self._quality_status_item(name))

    def _apply_filter(self):
        q = self.filter_edit.text().strip().lower()
        only_issues = self.only_issues.isChecked()
        for row in range(self.table.rowCount()):
            data_item = self.table.item(row, 0)
            d = data_item.data(Qt.UserRole) if data_item else {}
            hay = " ".join(str(d.get(k, "")) for k in
                           ("dataset_name", "robot_type", "uploader")).lower()
            hay += " " + uploader_cn(d.get("uploader")).lower()
            hide = bool(q) and q not in hay
            if not hide and only_issues:
                _, agg = chk_mod.run_checks(d, cfg=_CHECKS_CFG)
                hide = agg["worst"] == chk_mod.OK
            self.table.setRowHidden(row, hide)

    def _open_row_link(self, row, _col):
        d = self.table.item(row, 0).data(Qt.UserRole) or {}
        link = d.get("link")
        if link:
            QDesktopServices.openUrl(QUrl(link))
            self.status.setText(f"已打开: {link}")

    def _on_table_cell_clicked(self, row, col):
        if col != QUALITY_STATUS_COL:
            return
        item = self.table.item(row, 0)
        d = item.data(Qt.UserRole) if item else {}
        name = (d or {}).get("dataset_name")
        if self._quality_status.get(name) != "已检查":
            return
        report_dir = self._quality_reports.get(name)
        if report_dir and Path(report_dir).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(report_dir).resolve())))
            self.status.setText(f"已打开检查报告: {report_dir}")
        else:
            self._mark_quality_unchecked(name)
            QMessageBox.warning(self, "提示", "检查报告目录不存在或已被删除。")

    # ---- Language-annotation Prompt panel (read-only, 方式1 读文件) ----------
    def _selected_dataset(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _selected_datasets(self):
        """Return unique selected datasets in table order, skipping hidden rows."""
        datasets = []
        seen = set()
        for row in self.table.selectedRows():
            if self.table.fixed.isRowHidden(row):
                continue
            item = self.table.item(row, 0)
            d = item.data(Qt.UserRole) if item else None
            name = (d or {}).get("dataset_name")
            if name and name not in seen:
                seen.add(name)
                datasets.append(d)
        return datasets

    def _show_prompt_empty(self, msg):
        """Show only the centered fallback label (nothing selected)."""
        self.prompt_empty.setText(msg)
        self.prompt_empty.setVisible(True)
        self.detail_scroll.setVisible(False)
        self.report_progress.setVisible(False)

    def _on_dataset_selected(self):
        d = self._selected_dataset()
        if not d:
            self.prompt_meta.setText("")
            self._show_prompt_empty("选择左侧数据集查看信息。")
            if hasattr(self, "btn_quality_check"):
                self.btn_quality_check.setEnabled(False)
            return

        name = (d.get("dataset_name") or "").split("/")[-1]
        # Task text carried inline in the record (fetched during 统计/拉取) is
        # preferred — it means the prompt shows without any local file.
        inline_tasks = d.get("tasks") if isinstance(d.get("tasks"), list) else None
        task_path = tsk.resolve_path(d, OUT_DIR)
        anno_path = ann.resolve_path(d, OUT_DIR)

        # Checks run off the record itself (name / duration / prompt), so the
        # panel is useful for any selected row even before a full pull.
        self.prompt_empty.setVisible(False)
        self.detail_scroll.setVisible(True)

        self._pico_seq += 1  # invalidate a scan belonging to a previous row

        n_tasks = self._refresh_tasks(inline_tasks, task_path)
        n_anno_eps, total_eps = self._refresh_annotations(anno_path)
        agg = self._refresh_checks(d)
        self._refresh_episode_lengths(d)
        self._refresh_quality_report_panel(d)
        self._refresh_report(d)
        self._refresh_doctor_selection(d)

        bits = [f"数据集: {name}", f"{n_tasks} 条指令"]
        if anno_path:
            bits.append(f"{n_anno_eps}/{total_eps} 集有标注")
        if agg["n_fail"] or agg["n_warn"]:
            bits.append(f"检查 {chk_mod.badge(agg)[0]}")
        self.prompt_meta.setText(" · ".join(bits))

    def _doctor_dataset_key(self, d):
        """Return the viewer-relative path used by the Doctor API."""
        return self.viewer.dataset_rel_path(d, root=self._viewer_root())

    def _refresh_doctor_selection(self, d):
        """Reset or restore the opt-in Doctor result for the selected dataset."""
        self._doctor_seq += 1
        name = (d.get("dataset_name") or "").split("/")[-1]
        rel = self._doctor_dataset_key(d)
        self.doctor_dataset_label.setText(
            f"数据集: {name}" if rel else "该数据集不在 Viewer 数据根，无法运行 Doctor")
        self.doctor_run_button.setEnabled(bool(rel and self.viewer.is_running()))
        self.doctor_export_button.setEnabled(False)
        self.doctor_progress.setValue(0)
        self.doctor_progress.setFormat("未运行")
        self.doctor_tree.clear()
        self.doctor_status.setText(
            "Doctor 只在点击按钮后运行。" if rel
            else "请先下载数据集并确保 Viewer 数据根可访问。")
        self.doctor_current_result = None
        if not rel:
            return
        scope = self._doctor_scope_options()
        cache_key = (rel, json.dumps(scope, sort_keys=True))
        cached = self._doctor_cache.get(cache_key)
        if cached is not None:
            self._render_doctor(cached)

    def _start_doctor(self):
        """Start an explicit Doctor run for the selected local dataset."""
        dataset = self._selected_dataset()
        if not dataset:
            self.doctor_status.setText("请先选择数据集")
            return
        rel = self._doctor_dataset_key(dataset)
        if not rel:
            self.doctor_status.setText("该数据集不在 Viewer 数据根")
            return
        if not self.viewer.is_running():
            self.doctor_status.setText("Viewer 未运行，请先启动 Viewer")
            return
        scope = self._doctor_scope_options()
        cache_key = (rel, json.dumps(scope, sort_keys=True))
        self._doctor_seq += 1
        seq = self._doctor_seq
        self.doctor_run_button.setEnabled(False)
        self.doctor_export_button.setEnabled(False)
        self.doctor_progress.setValue(0)
        self.doctor_progress.setFormat("%p%")
        self.doctor_status.setText("正在启动 Doctor…")
        worker = DoctorWorker(self.viewer, rel, scope, seq)
        worker.progress.connect(
            self._guarded("刷新 Doctor 进度", self._on_doctor_progress, pass_args=True))
        worker.done.connect(
            self._guarded(
                "渲染 Doctor 结果",
                lambda done_seq, key, result, error:
                self._on_doctor_done(done_seq, key, result, error, cache_key),
                pass_args=True))
        worker.finished.connect(
            lambda worker=worker: self._forget_worker("_doctor_workers", worker))
        self._doctor_workers.append(worker)
        worker.start()

    def _on_doctor_progress(self, percent, message):
        self.doctor_progress.setValue(max(0, min(100, percent)))
        self.doctor_progress.setFormat(f"{percent}%")
        self.doctor_status.setText(message)

    def _on_doctor_done(self, seq, rel, result, error, cache_key):
        self._doctor_workers = [worker for worker in self._doctor_workers
                                if worker.isRunning()]
        if result is not None:
            self._doctor_cache[cache_key] = result
        if seq != self._doctor_seq:
            return
        current = self._selected_dataset()
        current_rel = self._doctor_dataset_key(current) if current else None
        if current_rel != rel:
            return
        self.doctor_run_button.setEnabled(bool(current_rel))
        if error:
            self.doctor_progress.setFormat("失败")
            self.doctor_status.setText(f"Doctor 失败: {error}")
            self.doctor_current_result = None
            return
        self._render_doctor(result)

    @staticmethod
    def _doctor_icon(severity):
        return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(severity, "—")

    def _render_doctor(self, result):
        """Render a Doctor response as an expandable tree."""
        self.doctor_current_result = result
        self.doctor_export_button.setEnabled(True)
        self.doctor_progress.setValue(100)
        self.doctor_progress.setFormat("完成")
        report = (result or {}).get("report") or {}
        execution = (result or {}).get("execution") or {}
        summary = report.get("summary") or {}
        self.doctor_pass.setText(f"PASS {summary.get('PASS', 0)}")
        self.doctor_warn.setText(f"WARN {summary.get('WARN', 0)}")
        self.doctor_fail.setText(f"FAIL {summary.get('FAIL', 0)}")
        loaded = execution.get("loaded_episode_count")
        duration = execution.get("duration_ms")
        self.doctor_status.setText(
            f"诊断完成：加载 {loaded} 个 Episode"
            + (f"，耗时 {duration / 1000:.1f}s" if isinstance(duration, (int, float)) else ""))
        self.doctor_tree.clear()
        for check in report.get("checks") or []:
            severity = check.get("severity", "WARN")
            messages = check.get("messages") or []
            item = QTreeWidgetItem([
                f"{self._doctor_icon(severity)} {check.get('name', 'Doctor check')}"
                f" · {severity} · {len(messages)} 条信息"
            ])
            self.doctor_tree.addTopLevelItem(item)
            for message in messages:
                child_severity = message.get("severity", severity)
                child = QTreeWidgetItem([
                    f"{self._doctor_icon(child_severity)} {message.get('message', '')}"
                ])
                child.setToolTip(0, child.text(0))
                item.addChild(child)
            item.setExpanded(severity != "PASS")

    def _export_doctor(self):
        if not self.doctor_current_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Doctor 报告", "doctor-report.json", "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self.doctor_current_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.doctor_status.setText(f"报告已导出: {path}")
        except OSError as exc:
            self.doctor_status.setText(f"导出失败: {exc}")

    def _refresh_checks(self, d):
        """Populate the 检查 tree (grouped by provider). Returns the aggregate."""
        self.check_tree.clear()
        results, agg = chk_mod.run_checks(
            d, providers=("custom", "viewer"), cfg=_CHECKS_CFG)
        provider_cn = {
            "custom": "自定义检查",
            "viewer": "Viewer 检查",
        }
        by_provider = {}
        for r in results:
            by_provider.setdefault(r.provider, []).append(r)
        for provider in ("custom", "viewer"):
            group = by_provider.get(provider)
            if not group:
                continue
            parent = QTreeWidgetItem([provider_cn.get(provider, provider)])
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            self.check_tree.addTopLevelItem(parent)
            for r in group:
                line = f"{chk_mod.icon(r.status)} {r.title}: {r.message}"
                node = QTreeWidgetItem([line])
                node.setToolTip(0, line)
                parent.addChild(node)
                for det in r.details:
                    node.addChild(QTreeWidgetItem([det]))
                node.setExpanded(True)
            parent.setExpanded(True)

        dataset_dir = self._dataset_dir(d)
        self.pico_check_button.setEnabled(dataset_dir is not None)
        dataset_key = str(dataset_dir.resolve()) if dataset_dir else None
        cached = self._pico_cache.get(dataset_key) if dataset_key else None
        if cached:
            result, error = cached
            self._render_pico_result(result, error)
        else:
            self.pico_check_status.setText(
                "未检测（点击按钮开始）" if dataset_dir else "未下载本地数据，无法检测")
        self._quality_group = self._add_quality_check_group(
            [chk_mod.CheckResult(
                "episode_local_quality", "Episode 级质量定位", "local_quality",
                chk_mod.SKIP, "等待手动执行。点击「执行深度检查」开始。", [])])
        return agg

    def _add_quality_check_group(self, results):
        parent = QTreeWidgetItem(["本地/远程深度检查"])
        font = parent.font(0)
        font.setBold(True)
        parent.setFont(0, font)
        self.check_tree.addTopLevelItem(parent)
        for result in results:
            line = f"{chk_mod.icon(result.status)} {result.title}: {result.message}"
            node = QTreeWidgetItem([line])
            node.setToolTip(0, line)
            parent.addChild(node)
            for detail in result.details:
                node.addChild(QTreeWidgetItem([detail]))
            node.setExpanded(True)
        parent.setExpanded(True)
        return parent

    def on_quality_check(self):
        dataset = self._selected_dataset()
        if not dataset:
            QMessageBox.warning(self, "提示", "请先在左侧表格选中一个数据集。")
            return
        if self.quality_worker and self.quality_worker.isRunning():
            QMessageBox.information(self, "提示", "已有深度检查正在运行。")
            return
        self._quality_seq += 1
        seq = self._quality_seq
        self.btn_quality_check.setEnabled(False)
        self.btn_quality_cancel.setEnabled(True)
        self.quality_progress.setVisible(True)
        self.quality_progress.setValue(0)
        self.quality_note.setText("检查中，界面可以继续操作。")
        index = self.check_tree.indexOfTopLevelItem(self._quality_group)
        if index >= 0:
            self.check_tree.takeTopLevelItem(index)
        self._quality_group = self._add_quality_check_group(
            [chk_mod.CheckResult(
                "episode_local_quality", "Episode 级质量定位", "local_quality",
                chk_mod.SKIP, "检查中...", [])])
        self.quality_worker = QualityWorker(seq, dataset, self.token, _CHECKS_CFG)
        self.quality_worker.progress.connect(
            self._guarded("刷新深度检查进度", self._on_quality_progress, pass_args=True))
        self.quality_worker.done.connect(
            self._guarded("渲染深度检查结果", self._on_quality_done, pass_args=True))
        self.quality_worker.finished.connect(
            self._guarded("结束深度检查", self._on_quality_worker_finished))
        self.quality_worker.start()

    def on_quality_cancel(self):
        if self.quality_worker and self.quality_worker.isRunning():
            self.quality_worker.cancel()
            self.quality_note.setText("已请求取消，任务将在当前处理阶段结束后退出。")
            self.btn_quality_cancel.setEnabled(False)

    def _on_quality_progress(self, message, pct):
        self.quality_note.setText(message)
        self.quality_progress.setValue(max(0, min(100, int(pct or 0))))

    def _on_quality_done(self, seq, name, results, report_dir):
        if seq != self._quality_seq:
            return
        index = self.check_tree.indexOfTopLevelItem(self._quality_group)
        if index >= 0:
            self.check_tree.takeTopLevelItem(index)
        self._quality_group = self._add_quality_check_group(results)
        if report_dir and Path(report_dir).is_dir():
            self._quality_status[name] = "已检查"
            self._quality_reports[name] = report_dir
            self._quality_records[name] = {
                "status": "已检查", "report_dir": report_dir,
                "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            self._save_quality_records()
            self.quality_note.setText(f"报告目录: {report_dir}")
            self.status.setText(f"深度检查完成，报告: {report_dir}")
        else:
            self.quality_note.setText(results[0].message if results else "未生成报告。")
        selected = self._selected_dataset() or {}
        if selected.get("dataset_name") == name:
            self._refresh_quality_report_panel(selected)

    def _on_quality_worker_finished(self):
        worker = self.sender()
        if worker is self.quality_worker:
            self.quality_worker = None
        self.btn_quality_check.setEnabled(True)
        self.btn_quality_cancel.setEnabled(False)
        self.quality_progress.setVisible(False)
        if worker is not None:
            worker.deleteLater()

    def _current_quality_report(self):
        dataset = self._selected_dataset() or {}
        name = dataset.get("dataset_name") or ""
        report_dir = self._quality_reports.get(name)
        if report_dir and Path(report_dir).is_dir():
            return name, Path(report_dir)
        return name, None

    def _refresh_quality_report_panel(self, dataset=None):
        self.quality_issue_tree.clear()
        self.quality_overview.setText("")
        dataset = dataset or self._selected_dataset() or {}
        report_dir = self._quality_reports.get(dataset.get("dataset_name") or "")
        if not report_dir or not Path(report_dir).is_dir():
            return
        try:
            import dataset_quality

            records = dataset_quality.load_report_records(report_dir)
            summary = dataset_quality.summarize_records(records)
        except Exception as exc:
            self.quality_overview.setText(f"报告读取失败: {exc}")
            return
        rules = ", ".join(
            f"{key}:{value}" for key, value in summary["by_rule"].items()) or "无"
        self.quality_overview.setText(
            f"质量总览：问题 {summary['total_issues']} 项，涉及 episode "
            f"{summary['episode_count']} 条，未确认 {summary['unconfirmed']} 项；"
            f"问题类型 {rules}")
        for index, record in enumerate(records, 1):
            start, end = record.get("start_sec"), record.get("end_sec")
            if start is not None and end is not None:
                time_text = f"{float(start):.3f}s-{float(end):.3f}s"
            elif record.get("frame") is not None:
                time_text = f"frame {record['frame']}"
            else:
                time_text = "-"
            item = QTreeWidgetItem([
                str(index),
                "-" if record.get("episode") is None else str(record["episode"]),
                record.get("rule") or "-", record.get("field") or "-",
                time_text, record.get("review_status", "未确认"),
            ])
            item.setToolTip(2, record.get("message") or "")
            item.setData(0, Qt.UserRole, record.get("issue_id"))
            self.quality_issue_tree.addTopLevelItem(item)
        for column in range(self.quality_issue_tree.columnCount()):
            self.quality_issue_tree.resizeColumnToContents(column)

    def on_mark_quality_issue(self, status):
        name, report_dir = self._current_quality_report()
        item = self.quality_issue_tree.currentItem()
        if not name or not report_dir or not item:
            QMessageBox.warning(self, "提示", "请先选择一条已有报告中的问题。")
            return
        try:
            import dataset_quality

            records = dataset_quality.load_report_records(report_dir)
            issue_id = item.data(0, Qt.UserRole)
            for record in records:
                if record.get("issue_id") == issue_id:
                    record["review_status"] = status
                    break
            dataset_quality.save_review_status(report_dir, records)
        except Exception as exc:
            QMessageBox.warning(self, "提示", f"保存确认状态失败: {exc}")
            return
        self._refresh_quality_report_panel()

    def on_open_quality_report(self):
        name, report_dir = self._current_quality_report()
        if not name or not report_dir:
            QMessageBox.warning(self, "提示", "当前数据集没有可打开的检查报告。")
            return
        summary = report_dir / "summary.html"
        target = summary if summary.is_file() else report_dir
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def on_export_quality_report(self):
        name, report_dir = self._current_quality_report()
        if not name or not report_dir:
            QMessageBox.warning(self, "提示", "当前数据集没有可导出的检查报告。")
            return
        zip_path = report_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in report_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(report_dir.parent))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(zip_path.parent.resolve())))
        self.status.setText(f"已导出检查报告 ZIP: {zip_path}")

    def on_clear_quality_reports(self):
        name, report_dir = self._current_quality_report()
        if not name or not report_dir:
            QMessageBox.information(self, "提示", "当前数据集没有检查报告产物。")
            return
        answer = QMessageBox.question(
            self, "确认清理", f"确认删除当前数据集的检查报告？\n{report_dir.parent}")
        if answer != QMessageBox.Yes:
            return
        shutil.rmtree(report_dir.parent, ignore_errors=True)
        self._quality_status[name] = "未检查"
        self._quality_reports.pop(name, None)
        self._quality_records.pop(name, None)
        self._save_quality_records()
        self._update_quality_status_cells(name)
        self._refresh_quality_report_panel()

    def on_clear_quality_cache(self):
        cfg = _CHECKS_CFG.get("local_quality") or {}
        cache_dir = Path(cfg.get("remote_cache_dir", ".quality_cache"))
        answer = QMessageBox.question(
            self, "确认清理", f"确认删除远程检查缓存？\n{cache_dir.resolve()}")
        if answer != QMessageBox.Yes:
            return
        shutil.rmtree(cache_dir, ignore_errors=True)
        self.status.setText(f"已清理检查缓存: {cache_dir}")

    def _start_pico_check(self):
        """Start an explicit PICO scan for the currently selected dataset."""
        dataset = self._selected_dataset()
        dataset_dir = self._dataset_dir(dataset)
        if dataset_dir is None:
            self.pico_check_status.setText("当前数据集未下载到本地")
            return
        key = str(dataset_dir.resolve())
        self._pico_seq += 1
        seq = self._pico_seq
        self.pico_check_button.setEnabled(False)
        self.pico_check_status.setText("检测中…")
        self._pico_cache.pop(key, None)
        worker = PicoMotrackerWorker(
            dataset_dir,
            _CHECKS_CFG.get("pico_motracker", {}),
            seq,
        )
        worker.done.connect(
            self._guarded("渲染 PICO 检查结果", self._on_pico_check_done, pass_args=True))
        worker.finished.connect(
            lambda worker=worker: self._forget_worker("_pico_workers", worker))
        self._pico_workers.append(worker)
        worker.start()

    def _on_pico_check_done(self, seq, key, result, error):
        self._pico_workers = [worker for worker in self._pico_workers
                              if worker.isRunning()]
        if result is None:
            self._pico_cache[key] = (None, error)
        else:
            self._pico_cache[key] = (result, "")
        if seq != self._pico_seq:
            return
        current = self._selected_dataset()
        current_dir = self._dataset_dir(current)
        current_key = str(current_dir.resolve()) if current_dir else None
        if current_key != key:
            return
        self.pico_check_button.setEnabled(current_dir is not None)
        self._render_pico_result(result, error)

    def _render_pico_result(self, result, error=""):
        """Render trajectory events as expandable Episode/event tree nodes."""
        # Remove a previous PICO result while retaining the ordinary rules.
        for index in range(self.check_tree.topLevelItemCount() - 1, -1, -1):
            item = self.check_tree.topLevelItem(index)
            if item.data(0, Qt.UserRole) == "pico_motracker":
                self.check_tree.takeTopLevelItem(index)
        if error:
            self.pico_check_status.setText(f"检测失败: {error}")
            return
        if result is None:
            return
        if result.total_events == 0:
            self.pico_check_status.setText(
                f"未发现超过阈值的跳变（扫描 {result.scanned_transitions:,} 个帧间隔）")
            parent = QTreeWidgetItem(["✅ PICO MoTracker 轨迹：未发现异常"])
            parent.setData(0, Qt.UserRole, "pico_motracker")
            self.check_tree.addTopLevelItem(parent)
            return

        self.pico_check_status.setText(
            f"发现 {result.total_events} 个事件，涉及 {len(result.affected_episodes)} 个 Episode")
        parent = QTreeWidgetItem([
            f"❌ PICO MoTracker 轨迹：{result.total_events} 个异常事件，"
            f"{len(result.affected_episodes)} 个 Episode"
        ])
        parent.setData(0, Qt.UserRole, "pico_motracker")
        self.check_tree.addTopLevelItem(parent)
        by_episode = {}
        for event in result.events:
            by_episode.setdefault(event.episode_index, []).append(event)
        for episode_index, events in sorted(by_episode.items()):
            ep_item = QTreeWidgetItem([
                f"Episode {episode_index}（{len(events)} 个事件）"])
            parent.addChild(ep_item)
            for event in events:
                hit_parts = []
                for axis in event.axis_hits:
                    hit_parts.append(
                        f"{axis} Δ{event.deltas[axis]:+.3f} "
                        f"(阈值 {result.thresholds['axis_step_threshold'][axis]:.3f})")
                if event.xyz_hit:
                    hit_parts.append(
                        f"XYZ {event.xyz_step:.3f} "
                        f"(阈值 {result.thresholds['xyz_step_threshold']:.3f})")
                line = QTreeWidgetItem([
                    f"{event.hand}_tcp · frame {event.previous_frame}→{event.frame_index} · "
                    f"{event.previous_time:.3f}–{event.timestamp:.3f}s · "
                    + ", ".join(hit_parts)
                ])
                line.setToolTip(0, line.text(0))
                ep_item.addChild(line)
            ep_item.setExpanded(False)
        if result.truncated:
            parent.addChild(QTreeWidgetItem([
                f"其余事件未展开（详情上限 {result.thresholds['max_event_details']}）"
            ]))
        parent.setExpanded(True)

    def _set_episode_length_note(self, message):
        """Show a single muted status row in the episode-length tree."""
        self.episode_length_tree.clear()
        item = QTreeWidgetItem([message, "", ""])
        item.setForeground(0, QBrush(QColor("#999999")))
        self.episode_length_tree.addTopLevelItem(item)
        item.setFirstColumnSpanned(True)

    def _refresh_episode_lengths(self, d):
        """Populate duration ranges and their episode rows from local metadata."""
        dataset_dir = self._dataset_dir(d)
        if dataset_dir is None:
            self._set_episode_length_note("需要先下载本地数据集，才能查看 episode 时长。")
            return

        episodes, error = ep_len.load_episode_lengths(dataset_dir)
        if error:
            self._set_episode_length_note(error)
            return

        groups = [group for group in ep_len.group_episode_lengths(episodes)
                  if group["episodes"]]
        if not groups:
            self._set_episode_length_note("没有可显示的 episode 时长数据。")
            return

        self.episode_length_tree.clear()
        last_index = len(groups) - 1
        for index, group in enumerate(groups):
            members = group["episodes"]
            parent = QTreeWidgetItem([group["label"], str(len(members)), ""])
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            self.episode_length_tree.addTopLevelItem(parent)
            for episode in members:
                child = QTreeWidgetItem([
                    f"ep {episode['episode_index']}",
                    f"{episode['length_seconds']:.1f}s",
                    f"{episode['frames']} frames",
                ])
                child.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                child.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
                parent.addChild(child)
            parent.setExpanded(index == 0 or index == last_index)

    # ---- Viewer /report analysis (async → STATISTICS/FILTERING/INSIGHTS) --- #
    _VERDICT_COLOR = {
        "Smooth": "#2e7d32", "Consistent": "#2e7d32",
        "Moderate": "#ef8c00", "Moderate variance": "#ef8c00",
        "Jerky": "#c62828", "High variance": "#c62828", "N/A": "#9e9e9e",
    }

    def _report_set_note(self, msg, busy=False):
        """Put a status/placeholder message in the three report-driven boxes."""
        note = f"<span style='color:{UI_COLORS['text_muted']}'>{_esc(msg)}</span>"
        for lbl in (self.stat_view, self.filter_view, self.insight_view):
            lbl.setText(note)
        self.report_progress.setVisible(busy)

    def _refresh_report(self, d):
        """Fill STATISTICS / FILTERING / ACTION INSIGHTS from the viewer /report
        analysis for the selected dataset. Fetched in a background thread (can
        take tens of seconds); stale selections are ignored."""
        if self._closing:
            return
        self._report_seq += 1
        seq = self._report_seq
        if not self.viewer.is_ready():
            self._report_set_note("Viewer 尚未就绪；顶栏点「启动」后显示分析。")
            return
        rel = self.viewer.dataset_rel_path(d, root=self._viewer_root())
        if not rel:
            self._report_set_note("该数据集不在 Viewer 数据根，暂无分析。")
            return
        self._report_set_note("分析中…（首次约 10–30s）", busy=True)
        w = ReportWorker(self.viewer, rel, seq)
        w.done.connect(
            self._guarded("渲染 Viewer 分析", self._on_report_done, pass_args=True))
        w.finished.connect(
            lambda worker=w: self._forget_worker("_report_workers", worker))
        self._report_workers.append(w)
        w.start()

    def _on_report_done(self, seq, rel, report, err):
        self._report_workers = [w for w in self._report_workers if w.isRunning()]
        if seq != self._report_seq:
            return  # user moved to another dataset; ignore stale result
        if report is None:
            self._report_set_note(f"分析失败: {err}")
            return
        self._render_report(report)

    def _render_report(self, r):
        """Split the /report fields across STATISTICS / FILTERING / ACTION
        INSIGHTS as clean rich-text (label muted, value bold, verdict colored)."""
        self.report_progress.setVisible(False)
        ds = r.get("dataset") or {}
        q = r.get("quality") or {}
        t = r.get("training") or {}
        sm = q.get("smoothness")

        def b(val):  # bold value
            return f"<b>{_esc(fmt_value(val) if isinstance(val, (int, float)) else val)}</b>"

        def verdict(label):
            c = self._VERDICT_COLOR.get(label, "#333")
            return f"<b style='color:{c}'>{_esc(label)}</b>"

        def detail(text):  # muted sub-line under a value (like the viewer badge)
            return (f"<div style='color:{UI_COLORS['text_muted']}; font-size:8pt;"
                    f" margin-top:2px'>{_esc(text)}</div>")

        def kv_table(rows, pairs_per_row=1, label_bg="#eef7ee",
                     cell_bg="transparent", label_width=None):
            """Render one or more metric/value pairs per HTML table row."""
            chunks = [rows[i:i + pairs_per_row]
                      for i in range(0, len(rows), pairs_per_row)]
            body_parts = []
            for chunk in chunks:
                cells = []
                for lab, val in chunk:
                    width = f" width:{label_width};" if label_width else ""
                    cells.extend([
                        "<td style='padding:4px 8px; color:#5b6b5b;"
                        f" background:{label_bg}; border:1px solid #d4e7d4;"
                        f"{width} white-space:nowrap; vertical-align:top'>" + lab + "</td>",
                        f"<td style='padding:4px 8px; background:{cell_bg};"
                        " border:1px solid #d4e7d4; vertical-align:top;"
                        " white-space:normal; word-wrap:break-word'>" + val + "</td>",
                    ])
                body_parts.append("<tr>" + "".join(cells) + "</tr>")
            body = "".join(body_parts)
            return ("<table width='100%' cellspacing='0' cellpadding='0' "
                    "style='border-collapse:collapse'>" + body + "</table>")

        # --- STATISTICS (内嵌表格，与「数据集统计分区」风格一致) ---
        integ = r.get("integrity") or {}
        st = integ.get("status", "?")
        st_col = "#2e7d32" if st == "ok" else "#c62828"
        st_html = f"<b style='color:{st_col}'>{_esc(st)}</b>"
        if integ.get("issues"):
            st_html += (f"<br><span style='color:{UI_COLORS['red']}; font-size:8pt'>"
                        f"{_esc('; '.join(integ['issues']))}</span>")

        summary_rows = [("完整性", st_html),
                 ("Episodes", b(ds.get("total_episodes"))),
                 ("Frames", b(ds.get("total_frames"))),
                 ("摄像头", b(len(ds.get("cameras") or []))),
                 ("fps", b(ds.get("fps")))]
        el = q.get("episodeLength")
        if el:
            summary_rows.append(("时长 最短/最长 (s)",
                          f"{b(el.get('shortest'))} / {b(el.get('longest'))}"))
            summary_rows.append(("时长 均值/中位 (s)",
                          f"{b(el.get('mean'))} / {b(el.get('median'))}"))
            summary_rows.append(("时长 std", b(el.get("std"))))

        # Keep the compact summary at exactly four rows by rendering two
        # metric/value pairs per row. Less prominent quality counts remain
        # visible as a small note below the 4×4 table.
        self.stat_view.setText(kv_table(
            summary_rows[:8], pairs_per_row=2,
            label_bg="#ffffff", cell_bg="#ffffff"))
        extra_quality = []
        if q:
            extra_quality.append(
                f"抖动集 {len(q.get('jerkyEpisodes') or [])} · "
                f"低运动集 {len(q.get('lowMovementEpisodes') or [])}")
        if len(summary_rows) > 8:
            extra_quality.append(" · ".join(
                _esc(f"{lab}: {val}") for lab, val in summary_rows[8:]))
        if extra_quality:
            self.stat_view.setText(
                self.stat_view.text() +
                "<div style='color:#777; font-size:11px; margin-top:4px'>" +
                _esc(" · ".join(extra_quality)) + "</div>")

        # --- FILTERING: smoothness "Overall" + breakdown lines ---
        if sm:
            label = (sm.get("verdict") or {}).get("label") or "—"
            html = f"<div style='line-height:150%'>Overall: {verdict(label)}"
            lines = sm.get("lines") or []
            if lines:
                html += "<ul style='margin:6px 0 0 -24px;'>" + "".join(
                    f"<li>{_esc(l)}</li>" for l in lines) + "</ul>"
            if sm.get("tip"):
                html += f"<div style='color:{UI_COLORS['text_muted']}; margin-top:4px'>{_esc(sm['tip'])}</div>"
            html += "</div>"
            self.filter_view.setText(html)
        else:
            self.filter_view.setText(
                f"<span style='color:{UI_COLORS['text_muted']}'>无平滑度数据</span>")

        # --- ACTION INSIGHTS: training config (内嵌表格) ---
        irows = []
        sc = t.get("suggestedChunkLength")
        if sc:
            secs = round(sc.get("seconds", 0), 2)
            val = (f"<b>{sc.get('steps')} 步 ({secs}s)</b>"
                   + detail("自相关首次降到 0.5 以下的中位滞后步数（跨各动作维度）——"
                            "即动作可预测、可打包为一个 chunk 的时域长度。"))
            irows.append(("建议 chunk 长度", val))
        elif "training" in r:
            irows.append(("建议 chunk 长度", "—"))
        cd = t.get("controlDelay")
        if cd:
            steps = cd.get("meanSteps")
            secs = round(cd.get("seconds", 0), 3)
            causal = ("<span style='color:#2e7d32'>· 因果 ✓</span>" if cd.get("causalOk")
                      else "<span style='color:#c62828'>· 非因果 ✗</span>")
            if isinstance(steps, (int, float)) and steps > 0:
                d = (f"状态变化平均滞后动作约 {steps} 帧；"
                     f"建议将 action[t] 与 state[t+{steps}] 对齐。")
            elif isinstance(steps, (int, float)) and steps < 0:
                d = f"动作平均滞后状态变化约 {-steps} 帧（预测性动作）。"
            else:
                d = "动作与状态基本同步，无明显时延。"
            val = f"<b>{steps} 步 ({secs}s)</b> {causal}" + detail(d)
            irows.append(("控制延迟", val))
        sv = t.get("speedVariance")
        if sv:
            tail = (" <span style='color:#c62828'>· 需速度归一</span>"
                    if sv.get("needsVelocityNorm") else "")
            val = (f"{verdict((sv.get('verdict') or {}).get('label'))} "
                   f"<span style='color:{UI_COLORS['text_muted']}'>cv {round(sv.get('cv', 0), 3)}</span>{tail}")
            irows.append(("速度方差", val))
        meta = r.get("meta") or {}
        if meta.get("sampledEpisodes") is not None:
            irows.append(("抽样",
                          f"<span style='color:#aaa'>{meta.get('sampledEpisodes')} 集</span>"))
        self.insight_view.setText(kv_table(
            irows, label_bg="#ffffff", cell_bg="#ffffff", label_width="28%"))

    def _refresh_tasks(self, inline, path):
        """Fill the task-instruction list. Prefers inline task rows (from the
        stats/pull record); falls back to reading a local tasks.parquet.
        Returns the task count."""
        self.task_list.clear()

        def note(msg):
            self.task_list.setVisible(False)
            self.task_note.setVisible(True)
            self.task_note.setText(msg)

        if inline:
            tasks = inline
        elif path:
            tasks, err = tsk.load(path)
            if err:
                note(err)
                return 0
        else:
            note("无 Language Instruction(该数据集未提供 tasks.parquet)。")
            return 0

        if not tasks:
            note("无 Language Instruction(该数据集未提供 tasks.parquet)。")
            return 0

        for row in tasks:
            item = QListWidgetItem(row["task"])
            item.setToolTip(row["task"])
            self.task_list.addItem(item)
        self.task_list.setVisible(True)
        self.task_note.setVisible(False)
        return len(tasks)

    def _refresh_annotations(self, path):
        """Fill the viewer-annotation tree. Returns (annotated_eps, total_eps)."""
        doc, err = ann.load(path) if path else ({"episodes": {}}, None)
        self._prompt_doc = doc
        eps = ann.episodes_with_atoms(doc)
        total_eps = len(doc.get("episodes", {}))

        def note(msg):
            self.prompt_ep_wrap.setVisible(False)
            self.prompt_tree.setVisible(False)
            self.anno_note.setVisible(True)
            self.anno_note.setText(msg)

        if not path:
            note("暂无 viewer 语言标注(可在 viewer 中编辑生成)。")
            return 0, 0
        if err:
            note(err)
            return 0, total_eps
        if not eps:
            note("暂无 viewer 语言标注(可在 viewer 中编辑生成)。")
            return 0, total_eps

        self.prompt_ep.blockSignals(True)
        self.prompt_ep.clear()
        for ep in eps:
            self.prompt_ep.addItem(f"ep {ep}", ep)
        self.prompt_ep.setCurrentIndex(0)
        self.prompt_ep.blockSignals(False)

        self.anno_note.setVisible(False)
        self.prompt_ep_wrap.setVisible(True)
        self.prompt_tree.setVisible(True)
        self._refresh_prompt_tree()
        return len(eps), total_eps

    def _refresh_prompt_tree(self):
        self.prompt_tree.clear()
        ep = self.prompt_ep.currentData()
        if ep is None:
            return
        atoms = ann.atoms_for_episode(self._prompt_doc, ep)
        for style, group in ann.group_by_style(atoms):
            parent = QTreeWidgetItem([f"{ann.style_label(style)} ({len(group)})"])
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            self.prompt_tree.addTopLevelItem(parent)
            for atom in group:
                text = ann.atom_text(atom)
                if ann.is_event_style(style):
                    ts = atom.get("timestamp")
                    if isinstance(ts, (int, float)):
                        text = f"{ts:.1f}s  {text}"
                child = QTreeWidgetItem([text])
                child.setToolTip(0, text)
                parent.addChild(child)
            parent.setExpanded(True)

    def _refresh_trends(self, daily_rows=None, date_from="", date_to="",
                        range_label="全部日期"):
        if not hasattr(self, "trend_plot"):
            return
        daily_rows = daily_rows or []
        series = self._trend_series_from_daily_rows(daily_rows, date_from, date_to)
        self.trend_plot.clear()
        self.trend_cum_view.clear()
        self.trend_plot.hideAxis("right")
        self.trend_plot.getAxis("left").setLabel("小时")
        self.trend_plot.setTitle(f"{range_label} 每日新增与累计小时数")
        if not self.history and not daily_rows:
            self.trend_hint.setText(
                "暂无历史数据。执行「仅拉取统计信息」或「拉取组织及其下所有数据集」后按天积累趋势。")
            return
        if not series:
            self.trend_hint.setText("当前时间范围内暂无可归因到真实日期的新增数据。")
            return
        self.trend_hint.setText(
            "按 HF last_modified 真实归日；无新增日期按 0 显示，累计线为当前范围内新增小时累计。")
        axis_from, axis_to = self._trend_date_bounds(series, date_from, date_to)
        x, new_hours, total_hours = self._trend_plot_points(series, axis_from)
        ticks = [self._trend_x_ticks(axis_from, axis_to)]
        bg = pg.BarGraphItem(x=x, height=new_hours, width=0.8, brush="#4C8BF5")
        self.trend_plot.addItem(bg)
        if self._trend_uses_dual_axis(new_hours, total_hours):
            self.trend_hint.setText(
                self.trend_hint.text() + " 量级差距较大，累计小时使用右侧纵轴。")
            self.trend_plot.getAxis("left").setLabel("新增小时")
            self.trend_plot.getAxis("right").setLabel("累计小时")
            self.trend_plot.showAxis("right")
            self.trend_cum_view.addItem(pg.PlotDataItem(
                x, total_hours,
                pen=pg.mkPen("#34A853", width=2), symbol="o",
                symbolBrush="#34A853"))
            self._sync_trend_cum_view()
            self.trend_plot.setYRange(0, (max(new_hours, default=0) or 1) * 1.08,
                                      padding=0)
            self.trend_cum_view.setYRange(0, (max(total_hours, default=0) or 1) * 1.08,
                                          padding=0)
        else:
            self.trend_plot.plot(
                x, total_hours,
                pen=pg.mkPen("#34A853", width=2), symbol="o",
                symbolBrush="#34A853")
            max_y = max(max(new_hours, default=0), max(total_hours, default=0), 1)
            self.trend_plot.setYRange(0, max_y * 1.08, padding=0)
        self.trend_plot.getAxis("bottom").setTicks(ticks)
        span = days_between(axis_from, axis_to)
        span = span if span is not None else max(x, default=0)
        self.trend_plot.setXRange(-0.5, span + 0.5, padding=0.02)

    @staticmethod
    def _trend_series_from_daily_rows(daily_rows, date_from="", date_to=""):
        rows = [row for row in daily_rows if row.get("date")]
        if not rows:
            return []
        dates = [row["date"] for row in rows]
        axis_from = date_from or min(dates)
        axis_to = date_to or max(dates)
        span = days_between(axis_from, axis_to)
        if span is None or span < 0:
            return []
        daily_hours = {(
            dt.datetime.strptime(axis_from, "%y%m%d").date()
            + dt.timedelta(days=offset)
        ).strftime("%y%m%d"): 0.0 for offset in range(span + 1)}
        for row in rows:
            date = row.get("date")
            if date not in daily_hours:
                continue
            daily_hours[date] += row.get("hours", 0) or 0
        cumulative = 0.0
        out = []
        for date in sorted(daily_hours):
            new_hours = round(daily_hours[date], 3)
            cumulative = round(cumulative + new_hours, 3)
            out.append({
                "date": date,
                "new_hours": new_hours,
                "total_hours": cumulative,
            })
        return out

    @staticmethod
    def _trend_date_bounds(series, date_from="", date_to=""):
        if date_from and date_to:
            return date_from, date_to
        dates = [row.get("date") for row in series if row.get("date")]
        if not dates:
            return date_from or "", date_to or ""
        return date_from or min(dates), date_to or max(dates)

    @staticmethod
    def _trend_plot_points(series, axis_from):
        points = []
        for row in series:
            offset = days_between(axis_from, row.get("date") or "")
            if offset is not None:
                points.append((
                    offset,
                    row.get("new_hours", 0),
                    row.get("total_hours", 0),
                ))
        return (
            [offset for offset, _new_hours, _total_hours in points],
            [new_hours for _offset, new_hours, _total_hours in points],
            [total_hours for _offset, _new_hours, total_hours in points],
        )

    @staticmethod
    def _trend_uses_dual_axis(new_hours, total_hours):
        max_new = max(new_hours, default=0) or 0
        max_total = max(total_hours, default=0) or 0
        if max_new <= 0 or max_total <= 0:
            return False
        return max_total / max_new >= 4

    def _trend_x_ticks(self, date_from, date_to):
        """Adaptive date labels for the compact trend chart's calendar axis.

        Data points keep their real day offsets. Only labels are thinned so the
        x-axis remains readable in the right-side panel.
        """
        span = days_between(date_from, date_to)
        if span is None or span < 0:
            return []
        day_count = span + 1
        width = max(1, self.trend_plot.width() if hasattr(self, "trend_plot") else 0)
        target_px_per_label = 100
        max_labels = max(7, min(12, width // target_px_per_label))
        if day_count <= 7:
            label_count = day_count
        else:
            max_labels_by_spacing = span // 2 + 1
            label_count = max(7, min(day_count, max_labels, max_labels_by_spacing))
        if day_count <= label_count:
            offsets = list(range(day_count))
        else:
            offsets = [
                round(i * span / (label_count - 1))
                for i in range(label_count)
            ]
        start = dt.datetime.strptime(date_from, "%y%m%d").date()
        return [
            (offset, self._trend_x_label(
                (start + dt.timedelta(days=offset)).strftime("%y%m%d")))
            for offset in offsets
        ]

    @staticmethod
    def _trend_x_label(yymmdd):
        full = fmt_day(yymmdd)
        return full[5:] if len(full) == 10 else full

    def _refresh_daily_group_table(self, rows, dim):
        self.daily_group_table.setSortingEnabled(False)
        self.daily_group_table.setRowCount(len(rows))
        if not rows:
            self.daily_group_hint.setText(f"暂无可归因到「{dim}」的每日新增数据。")
        else:
            self.daily_group_hint.setText(
                f"按 HF last_modified 归日；数值优先使用 HF commit 差分，缺失时用本地快照兜底统计每个「{dim}」分组每日新增小时。")
        for i, row in enumerate(rows):
            values = [
                fmt_day(row.get("date")),
                row.get("hours", 0),
                row.get("episodes", 0),
                row.get("datasets", 0),
            ]
            values.insert(1, row.get("group") or "—")
            for j, value in enumerate(values):
                if j >= 2:
                    item = NumericItem(fmt_value(value), value)
                else:
                    item = QTableWidgetItem(str(value))
                self.daily_group_table.setItem(i, j, item)
        self.daily_group_table.setSortingEnabled(True)
        self.daily_group_table.sortItems(0, Qt.DescendingOrder)

    def _refresh_rollup(self):
        self._set_rollup_range_defaults()
        self.rollup_table.setRowCount(0)
        self.rollup_plot.clear()
        self.rollup_hint.setText("")
        dim = self.dim_combo.currentText()
        key_fn = ROLLUP_DIMS[dim]
        range_from, range_to = self._selected_rollup_range()
        range_text = f"{fmt_day(range_from)} ~ {fmt_day(range_to)}"
        if range_from == range_to:
            range_text = fmt_day(range_from)
        if not range_from and not range_to:
            range_text = "全部日期"
        daily_rows = [
            row for row in dd.hf_last_modified_daily_group_series(
            self.report, self.history, self.hf_changes, key_fn)
            if dd._date_in_range(row.get("date") or "", range_from, range_to)
        ]
        self._refresh_daily_group_table(daily_rows, dim)
        self._refresh_trends(daily_rows, range_from, range_to, range_text)
        if hasattr(self, "range_group_table"):
            self.range_group_table.setSortingEnabled(False)
            self.range_group_table.setRowCount(0)
        if self.report:
            self.rollup_hint.setText(f"当前时间范围：{range_text}")
        if not self.report:
            return
        if self.report.get("source") == "manual":
            self.rollup_hint.setText("该手动快照仅包含总量，无法按日期区间生成分组统计。")
            rows = dd.rollup(self.report.get("datasets", []), key_fn)
            self._render_rollup_summary(rows, "当前快照")
            self.rollup_plot.setTitle("各分组小时数")
            return
        rows = dd.hf_last_modified_group_range_totals(
            self.report, self.history, self.hf_changes, key_fn,
            date_from=range_from, date_to=range_to)
        self._render_rollup_summary(rows, range_text)

    def _render_rollup_summary(self, rows, range_label):
        self.rollup_table.setRowCount(len(rows))
        if hasattr(self, "range_group_hint"):
            self.range_group_hint.setText(f"当前汇总范围：{range_label}")
        self.rollup_hint.setText(f"当前时间范围：{range_label}")
        total_hours = sum(g.get("hours") or 0 for g in rows) or 1
        for i, g in enumerate(rows):
            hours = g.get("hours", 0)
            episodes = g.get("episodes", 0)
            datasets = g.get("datasets", g.get("count", 0))
            vals = [g["group"], datasets, episodes, hours, round(100 * hours / total_hours, 1)]
            for j, v in enumerate(vals):
                if j == 0:
                    item = QTableWidgetItem(str(v))
                else:
                    item = NumericItem(fmt_value(v), v)
                self.rollup_table.setItem(i, j, item)
        # Horizontal bars: one row per group so the labels (中文名 / 任务名) read
        # left-to-right and never overlap, however many groups there are. Cap the
        # chart to the top 20 by hours (the table above still lists them all).
        plot_rows = rows[:20]
        n = len(plot_rows)
        ys = [n - 1 - i for i in range(n)]  # rows are hours-desc -> largest on top
        bg = pg.BarGraphItem(x0=0, y=ys, height=0.7,
                             width=[g["hours"] for g in plot_rows], brush="#F9A825")
        self.rollup_plot.addItem(bg)

        def _short(s, k=42):
            s = str(s)
            return s if len(s) <= k else s[:k - 1] + "…"

        labels = [_short(g["group"]) for g in plot_rows]
        left = self.rollup_plot.getAxis("left")
        left.setTicks([[(ys[i], labels[i]) for i in range(n)]])
        # Widen the y-axis to the longest label so nothing is clipped; CJK glyphs
        # take ~2x the width of a latin char, so weight them double when sizing.
        vis = max((sum(2 if ord(c) > 0x2E80 else 1 for c in s) for s in labels),
                  default=8)
        left.setWidth(min(440, max(70, 12 + vis * 8)))
        self.rollup_plot.getAxis("bottom").setTicks(None)  # auto numeric hour scale
        self.rollup_plot.setYRange(-0.5, n - 0.5, padding=0.02)
        max_h = max((g["hours"] for g in plot_rows), default=1) or 1
        self.rollup_plot.setXRange(0, max_h, padding=0.05)  # bars start at 0, no left gap
        title = f"{range_label} 各分组小时数"
        self.rollup_plot.setTitle(
            f"{title}（前 {n}/{len(rows)}）" if len(rows) > n else title)

    # ---- Login / visibility indicator ------------------------------------- #
    def on_switch_account(self):
        """Prompt for an account label + HF token, apply it, and re-check identity.

        The token is what actually authenticates; the account field is just a
        note (the real login name is confirmed by whoami in the indicator). The
        token is kept in memory for this session only — it is never written to
        disk. For a persistent login use `huggingface-cli login` or $HF_TOKEN.
        """
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("切换账号 / Token")
        dlg.setMinimumWidth(440)
        form = QFormLayout(dlg)

        acc_edit = QLineEdit()
        acc_edit.setPlaceholderText("可留空，登录后会自动从 token 识别真实账号")
        tok_edit = QLineEdit()
        tok_edit.setPlaceholderText("hf_… 粘贴 HF access token")
        tok_edit.setEchoMode(QLineEdit.Password)
        show_btn = QPushButton("显示")
        show_btn.setCheckable(True)
        show_btn.setFixedWidth(48)
        show_btn.toggled.connect(
            lambda on: tok_edit.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        tok_row = QHBoxLayout()
        tok_row.setContentsMargins(0, 0, 0, 0)
        tok_row.addWidget(tok_edit, 1)
        tok_row.addWidget(show_btn)
        tok_wrap = QWidget()
        tok_wrap.setLayout(tok_row)

        form.addRow("账号(选填):", acc_edit)
        form.addRow("Token:", tok_wrap)
        hint = QLabel("Token 会保存到本地 .hf_token（已被 git 忽略，不会上传或"
                      "同步给他人），下次启动自动使用。清除请删除该文件。")
        hint.setStyleSheet(MUTED_TEXT_STYLE)
        hint.setWordWrap(True)
        form.addRow(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() != QDialog.Accepted:
            return
        token = tok_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "提示", "Token 不能为空。")
            return
        self.token = token
        save_token(token)  # persist locally (gitignored) for next runs
        acc = acc_edit.text().strip()
        self.status.setText(
            f"已应用并保存 Token{'（'+acc+'）' if acc else ''}，正在校验身份与可见数量 ...")
        self._refresh_identity()

    def _refresh_identity(self, *_):
        """Kick off a background check of who we are + how many datasets we see."""
        if self._closing:
            return
        org = self.org_combo.currentText().strip()
        if not org:
            return
        self.identity_label.setText("登录状态: 检测中…")
        self.identity_label.setStyleSheet(f"color:{UI_COLORS['text_muted']};")
        self._id_seq += 1
        seq = self._id_seq
        w = IdentityWorker(org, self.token)
        w.done.connect(lambda name, has, o, cnt, seq=seq:
                       self._on_identity(seq, name, has, o, cnt))
        w.finished.connect(
            lambda worker=w: self._forget_worker("_id_workers", worker))
        self._id_workers.append(w)  # hold a ref so the QThread isn't GC'd mid-run
        w.start()

    def _forget_worker(self, attr, worker):
        """Drop a completed short-lived QThread and schedule Qt-side deletion."""
        workers = getattr(self, attr, None)
        if workers is not None and worker in workers:
            workers.remove(worker)
        worker.deleteLater()

    def _qt_object_alive(self, obj):
        """True while a PySide wrapper still points at a live C++ QObject."""
        if obj is None:
            return False
        try:
            return bool(qt_is_valid(obj))
        except RuntimeError:
            return False

    def _on_one_shot_worker_finished(self, attr, worker=None):
        """Release an owned one-shot QThread after its native thread is stopped."""
        worker = worker or self.sender()
        if worker is getattr(self, attr, None):
            setattr(self, attr, None)
        if worker is self.worker:
            self.worker = None
        self._refresh_action_states()
        if worker is not None:
            worker.deleteLater()

    def _handle_ui_exception(self, action, exc, *, stop_speed=True):
        """Report an exception raised by a main-thread Qt slot without crashing."""
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        if stop_speed and hasattr(self, "speed_timer"):
            self._stop_speed()
        msg = f"{action}失败: {exc}"
        if hasattr(self, "status"):
            self.status.setText(msg)
        if not getattr(self, "_closing", False):
            try:
                QMessageBox.critical(self, "错误", msg)
            except Exception:
                pass

    def _guarded(self, action, func, *, pass_args=False, stop_speed=True):
        """Wrap Qt callbacks so Python exceptions do not escape the event loop."""
        def wrapped(*args, **kwargs):
            if getattr(self, "_closing", False):
                return None
            try:
                if pass_args:
                    return func(*args, **kwargs)
                return func()
            except Exception as exc:
                self._handle_ui_exception(action, exc, stop_speed=stop_speed)
                return None
        return wrapped

    def _on_identity(self, seq, name, has_token, org, count):
        # Only the most recent check may update the label — a slower older worker
        # (e.g. the startup one) must not clobber a fresh account-switch result.
        if seq != self._id_seq:
            return
        cnt = f"可见 {count} 个数据集" if count >= 0 else "数据集数查询失败"
        if not has_token:
            who, color = "未登录(匿名)", "#F9A825"
        elif name:
            who, color = f"已登录: {name}", "#34A853"
        else:
            who, color = "已登录: token 无效/过期", "#EA4335"
        self.identity_label.setText(f"{who} · {org} {cnt}")
        self.identity_label.setStyleSheet(f"color:{color}; font-weight:bold;")

    def _prepare_shutdown(self):
        if self._shutdown_done:
            return
        self._closing = True
        self._id_seq += 1
        self._report_seq += 1
        self._doctor_seq += 1
        self._pico_seq += 1
        if hasattr(self, "speed_timer"):
            self.speed_timer.stop()
        if hasattr(self, "quality_status_timer"):
            self.quality_status_timer.stop()
        # Let in-flight background reads finish so QThreads are not destroyed
        # mid-run. Identity can be a network call on startup; give it a real
        # chance before falling back to terminate.
        for attr, timeout_ms in (
            ("_id_workers", 10000),
            ("_report_workers", 10000),
            ("_doctor_workers", 10000),
            ("_pico_workers", 10000),
        ):
            for w in list(getattr(self, attr, [])):
                if not w.wait(timeout_ms):
                    w.terminate()
                    w.wait(2000)
        # Same for the one-shot workers (pull/stats/download/edit/push/op) and
        # especially the deep quality scan, which can run for minutes: ask it
        # to cancel, give it a moment, then terminate as a last resort —
        # otherwise Qt aborts with "QThread: Destroyed while thread is still
        # running" when the window closes mid-scan.
        qw = getattr(self, "quality_worker", None)
        if qw is not None and qw.isRunning():
            qw.cancel()
            if not qw.wait(8000):
                qw.terminate()
                qw.wait(2000)
        for w in list(self._download_workers):
            if w.isRunning():
                if not w.wait(5000):
                    w.terminate()
                    w.wait(2000)
        for name in ("worker", "_pull_worker", "_check_worker",
                     "_edit_worker", "_push_worker", "_op_worker"):
            w = getattr(self, name, None)
            if w is not None and hasattr(w, "wait") and w.isRunning():
                if hasattr(w, "cancel"):
                    w.cancel()
                if not w.wait(5000):
                    w.terminate()
                    w.wait(2000)
        # Stop the viewer subprocess we launched after viewer-backed workers have
        # settled, so their HTTP calls do not hang on a disappearing service.
        try:
            self.viewer.stop()
        except Exception:
            pass
        self._shutdown_done = True

    def closeEvent(self, event):
        self._prepare_shutdown()
        super().closeEvent(event)

    # ---- Button handlers -------------------------------------------------- #
    def _refresh_action_states(self):
        global_busy = any(
            getattr(self, attr, None) is not None
            for attr in ("_pull_worker", "_check_worker",
                         "_edit_worker", "_push_worker", "_op_worker"))
        stats_busy = getattr(self, "_stats_worker", None) is not None
        downloads_busy = bool(getattr(self, "_download_workers", ()))
        any_busy = global_busy or stats_busy or downloads_busy

        self.btn_pull.setEnabled(
            not global_busy and not stats_busy and not downloads_busy)
        self.btn_stats.setEnabled(not global_busy and not stats_busy)
        self.btn_download.setEnabled(not global_busy)
        self.btn_check.setEnabled(
            not global_busy and not stats_busy and not downloads_busy)
        self.btn_manual_stats.setEnabled(
            not global_busy and not stats_busy and not downloads_busy)
        self.btn_open.setEnabled(not any_busy)
        if hasattr(self, "btn_make_copy"):
            self.btn_make_copy.setEnabled(not any_busy)
            self.btn_run_op.setEnabled(not any_busy)
            self.btn_push_copy.setEnabled(
                not any_busy and self._last_copy_dir is not None)

    def _set_busy(self, busy):
        if busy:
            for b in (self.btn_pull, self.btn_stats, self.btn_download,
                      self.btn_check, self.btn_manual_stats, self.btn_open):
                b.setEnabled(False)
            if hasattr(self, "btn_make_copy"):
                self.btn_make_copy.setEnabled(False)
                self.btn_run_op.setEnabled(False)
                self.btn_push_copy.setEnabled(False)
        else:
            self._refresh_action_states()

    def on_pull(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        self._set_busy(True)
        self.bar.setValue(0)
        self.status.setText(f"开始拉取 {org} ...")
        self._watch_dir = Path(OUT_DIR)
        self._prev_bytes = dir_size(self._watch_dir)
        self._prev_t = time.monotonic()
        self.speed_label.setText("0.0 B/s")
        self.speed_timer.start()
        worker = PullWorker(org, OUT_DIR, self.token)
        self.worker = worker
        self._pull_worker = worker
        worker.log.connect(self.status.setText)
        worker.progress.connect(
            self._guarded("刷新同步进度", self._on_progress, pass_args=True))
        worker.done.connect(
            self._guarded("完成同步全部数据集", self._on_pull_done, pass_args=True))
        worker.error.connect(
            self._guarded("同步全部数据集", self._on_pull_error, pass_args=True))
        worker.finished.connect(
            self._guarded("结束同步全部数据集", self._on_pull_worker_finished))
        worker.start()

    def on_download_selected(self):
        """Download all datasets selected in the 看板 table in parallel."""
        datasets = self._selected_datasets()
        if not datasets:
            QMessageBox.warning(self, "提示", "请先在「看板」表格里选中一个数据集。")
            return
        if not self._download_workers:
            self._download_started = 0
            self._download_completed = 0
            self._download_successes = []
            self._download_failures = []
        running = {w.repo_id for w in self._download_workers}
        downloaded = self._downloaded_leaves()
        pending = [
            d for d in datasets
            if d.get("dataset_name") not in running
            and (d.get("dataset_name") or "").split("/")[-1] not in downloaded
        ]
        if not pending:
            selected_names = {d.get("dataset_name") for d in datasets}
            if selected_names and selected_names.issubset(running):
                msg = "选中的数据集已在下载中。"
            else:
                msg = "选中的数据集已下载或正在下载，未重复写入本地数据。"
            QMessageBox.information(self, "提示", msg)
            return

        self._watch_dir = Path(OUT_DIR)
        self._prev_bytes = dir_size(self._watch_dir)
        self._prev_t = time.monotonic()
        self.speed_label.setText("0.0 B/s")
        if not self.speed_timer.isActive():
            self.speed_timer.start()
        for d in pending:
            repo_id = d["dataset_name"]
            self._download_started += 1
            worker = DownloadOneWorker(repo_id, OUT_DIR, self.token)
            self._download_workers.append(worker)
            worker.log.connect(self.status.setText)
            worker.done.connect(
                self._guarded("完成下载数据集", self._on_download_one_done, pass_args=True))
            worker.error.connect(
                self._guarded("下载数据集", self._on_download_error, pass_args=True))
            worker.finished.connect(
                self._guarded("结束下载数据集", self._on_download_worker_finished))
            worker.start()
        skipped = len(datasets) - len(pending)
        suffix = f"，跳过已下载/下载中 {skipped} 个" if skipped else ""
        self.status.setText(f"开始下载 {len(pending)} 个数据集{suffix} ...")
        self._refresh_download_progress()
        self._refresh_action_states()

    def _on_download_one_done(self, local_dir):
        refresh_ok = True
        try:
            # The newly downloaded row now shows 已下载.  Keep this guarded:
            # it runs in the GUI thread after a background worker completes.
            self._refresh_table()
        except Exception as exc:
            refresh_ok = False
            self._handle_ui_exception("下载完成后刷新表格", exc, stop_speed=False)
        if refresh_ok and self._download_workers:
            self.status.setText(f"下载完成: {local_dir}")
        self._refresh_download_progress()

    def _on_download_error(self, msg):
        """Show a download failure; other workers keep running."""
        worker = self.sender()
        if worker is not None:
            try:
                worker.error_msg = msg
            except RuntimeError:
                pass
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self._refresh_download_progress()

    def _on_download_worker_finished(self):
        """Finalize the download UI and release the worker after QThread stops.

        With multiple workers, this slot owns the final batch summary; the done
        signal only refreshes the table and status as each dataset lands.
        """
        worker = self.sender()
        worker_alive = self._qt_object_alive(worker)
        local_dir = ""
        error_msg = ""
        if worker_alive:
            try:
                local_dir = worker.local_dir
                error_msg = worker.error_msg
            except RuntimeError:
                worker_alive = False

        if worker_alive and worker in self._download_workers:
            self._download_workers.remove(worker)
            if local_dir:
                self._download_successes.append(local_dir)
            elif error_msg:
                self._download_failures.append(error_msg)
            self._download_completed += 1
        elif worker_alive:
            if local_dir:
                self._download_successes.append(local_dir)
            elif error_msg:
                self._download_failures.append(error_msg)
        if worker_alive:
            self._retired_download_workers.append(worker)
            QTimer.singleShot(
                DOWNLOAD_WORKER_RELEASE_DELAY_MS,
                lambda w=worker: self._release_download_worker(w))
        self._refresh_download_progress()
        if not self._download_workers:
            self._finish_download_batch()
        self._refresh_action_states()

    def _release_download_worker(self, worker):
        """Release a finished download QThread after queued signals settle."""
        try:
            if worker in self._retired_download_workers:
                self._retired_download_workers.remove(worker)
            if self._qt_object_alive(worker):
                worker.deleteLater()
        except RuntimeError:
            pass

    def _refresh_download_progress(self):
        if self._download_workers:
            if not self.speed_timer.isActive():
                self.speed_timer.start()
            self.bar.setMaximum(max(self._download_started, 1))
            self.bar.setValue(self._download_completed)
        elif self._pull_worker is None and self._stats_worker is None:
            self._stop_speed()

    def _finish_download_batch(self):
        self._stop_speed()
        ok = len(self._download_successes)
        failed = len(self._download_failures)
        if ok == 1 and not failed:
            msg = f"下载完成: {self._download_successes[0]}"
        elif ok and not failed:
            msg = f"下载完成: {ok} 个数据集"
        elif ok:
            msg = f"下载完成: {ok} 个，失败 {failed} 个"
        else:
            msg = f"下载失败: {failed} 个数据集"
        self.status.setText(msg)
        self.bar.setMaximum(1)
        self.bar.setValue(1 if ok else 0)
        if ok:
            lines = list(self._download_successes[:10])
            if len(self._download_successes) > 10:
                lines.append(
                    f"… 其余 {len(self._download_successes) - 10} 个")
            if failed:
                lines.append(f"失败 {failed} 个数据集")
            box = QMessageBox(
                QMessageBox.Information, "完成", "\n".join(lines),
                QMessageBox.Ok, self)
            box.setAttribute(Qt.WA_DeleteOnClose)
            self._download_message_box = box
            box.finished.connect(
                lambda *_: setattr(self, "_download_message_box", None))
            box.open()
        self._download_successes = []
        self._download_failures = []
        self._download_started = 0
        self._download_completed = 0

    def on_stats(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        if self._stats_worker is not None:
            QMessageBox.warning(self, "提示", "刷新统计已在运行。")
            return
        self.bar.setValue(0)
        self.status.setText(f"开始统计 {org}（仅读取信息，不下载）...")
        worker = StatsWorker(org, self.token)
        self.worker = worker
        self._stats_worker = worker
        worker.log.connect(self.status.setText)
        worker.progress.connect(
            self._guarded("刷新统计进度", self._on_progress, pass_args=True))
        worker.done.connect(
            self._guarded("完成刷新统计", self._on_stats_done, pass_args=True))
        # Keep the busy lock until QThread.run() has actually returned.  The
        # worker emits done just before returning, so unlocking in _on_stats_done
        # allowed a second click to replace a still-running QThread.
        worker.error.connect(
            self._guarded("刷新统计", self._on_stats_error, pass_args=True))
        worker.finished.connect(
            self._guarded("结束刷新统计", self._on_stats_worker_finished))
        worker.start()
        self._refresh_action_states()

    def _on_stats_error(self, msg):
        """Show a stats failure; _on_stats_worker_finished unlocks the UI."""
        self._stop_speed()
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _on_stats_worker_finished(self):
        """Release the stats worker only after its native thread is stopped."""
        worker = self.sender()
        if worker is self._stats_worker:
            self._stats_worker = None
        if worker is self.worker:
            self.worker = None
        self._refresh_action_states()
        self._refresh_download_progress()
        if worker is not None:
            worker.deleteLater()

    def on_manual_stats(self):
        """Add an aggregate-only historical snapshot from another computer."""
        from PySide6.QtWidgets import (
            QDateEdit, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("手动补录总统计")
        dlg.setMinimumWidth(430)
        form = QFormLayout(dlg)

        date_edit = QDateEdit(QDate.currentDate())
        date_edit.setCalendarPopup(True)
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setMaximumDate(QDate.currentDate())
        org_edit = QLineEdit(self.org_combo.currentText().strip())

        def int_spin():
            spin = QSpinBox()
            spin.setRange(0, 2_000_000_000)
            spin.setGroupSeparatorShown(True)
            return spin

        datasets_spin = int_spin()
        episodes_spin = int_spin()
        frames_spin = int_spin()
        hours_spin = QDoubleSpinBox()
        hours_spin.setRange(0, 1_000_000_000)
        hours_spin.setDecimals(3)
        hours_spin.setSingleStep(0.1)
        hours_spin.setGroupSeparatorShown(True)

        form.addRow("统计日期:", date_edit)
        form.addRow("组织:", org_edit)
        form.addRow("数据集总数:", datasets_spin)
        form.addRow("总 episodes:", episodes_spin)
        form.addRow("总 frames:", frames_spin)
        form.addRow("总小时数:", hours_spin)
        hint = QLabel(
            "仅保存累计总量；今日新增会与前一个有记录日期自动比较。"
            "该快照不包含数据集表格、分组和 MVP 明细。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:12px;")
        form.addRow(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.Accepted:
            return

        org = org_edit.text().strip()
        if not org:
            QMessageBox.warning(self, "提示", "组织不能为空。")
            return
        day = date_edit.date()
        date = day.toString("yyMMdd")
        previous_report = self.report
        previous_was_live = bool(previous_report) and not self.stale_banner.isVisible()
        try:
            manual_report = dd.upsert_manual_totals(
                date=date,
                org=org,
                total_datasets=datasets_spin.value(),
                total_episodes=episodes_spin.value(),
                total_frames=frames_spin.value(),
                total_hours=hours_spin.value(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "补录失败", str(exc))
            return

        self.history = dd.load_history(OUT_DIR)
        self.report = self.history[-1] if self.history else None
        self._refresh_all()
        if self.report:
            if self.report.get("source") == "manual":
                self._show_stale_banner(self.report)
            elif previous_was_live and previous_report \
                    and (previous_report.get("pulled_at") or "") \
                    >= (manual_report.get("pulled_at") or ""):
                self._hide_stale_banner()
            else:
                self._show_stale_banner(self.report)
        self.status.setText(
            f"已补录 {day.toString('yyyy-MM-dd')} {org} 的总统计；"
            "今日新增按前一个记录日自动计算。")

    def _on_progress(self, done, total):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)

    def _tick_speed(self):
        now = time.monotonic()
        try:
            cur = dir_size(self._watch_dir)
        except Exception as exc:
            self.speed_timer.stop()
            self.status.setText(f"测速暂停: {exc}")
            return
        elapsed = now - (self._prev_t or now)
        if elapsed > 0:
            self.speed_label.setText(fmt_speed((cur - self._prev_bytes) / elapsed))
        self._prev_bytes = cur
        self._prev_t = now

    def _stop_speed(self):
        self.speed_timer.stop()
        self.speed_label.setText("—")

    def _tick_clock(self):
        now = dt.datetime.now()
        week = "一二三四五六日"[now.weekday()]
        self.clock_label.setText(now.strftime(f"%Y-%m-%d 周{week} %H:%M:%S"))

    def _on_pull_done(self, report, out_path):
        self._stop_speed()
        try:
            self.report = report
            self.history = dd.load_history(
                OUT_DIR, org=report.get("org"))  # new snapshot just written
            self.hf_changes = dd.load_hf_change_history()
            self._refresh_all()
            self._hide_stale_banner()  # data is now live
            fails = len(report.get("failures", []))
            msg = (
                f"拉取完成: {report.get('count', 0)}/"
                f"{report.get('requested', 0)} 个数据集")
            if fails:
                msg += f"，{fails} 个失败"
            self.status.setText(
                msg + (f"  ->  {out_path}" if out_path else ""))
        except Exception as exc:
            self._handle_ui_exception("拉取完成后刷新界面", exc, stop_speed=False)

    def _on_pull_error(self, msg):
        """Show a pull failure; finished releases the worker/busy lock."""
        self._stop_speed()
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _on_pull_worker_finished(self):
        """Release the pull worker only after QThread has stopped."""
        worker = self.sender()
        if worker is self._pull_worker:
            self._pull_worker = None
        if worker is self.worker:
            self.worker = None
        self._refresh_action_states()
        self._refresh_download_progress()
        if worker is not None:
            worker.deleteLater()

    def _on_stats_done(self, report):
        try:
            self.report = report
            # Record the day's totals so 趋势 / 今日新增 have a daily baseline. 统计
            # produces per-dataset detail (from each info.json), so this snapshot is
            # a full baseline — previously only 拉取 wrote history, which is why days
            # that were only 统计'd never showed up.
            hist_note = ""
            try:
                dd.append_pull(report)
                self.history = dd.load_history(OUT_DIR)
                self.hf_changes = dd.load_hf_change_history()
            except Exception as exc:
                hist_note = f"（历史未写入: {exc}）"
            self._refresh_all()
            self._hide_stale_banner()  # data is now live
            fails = len(report.get("failures", []))
            msg = (
                f"统计完成: {report.get('count', 0)}/"
                f"{report.get('requested', 0)} 个数据集，"
                f"共 {report.get('total_hours', 0)} 小时")
            if fails:
                msg += f"，{fails} 个读取失败"
            self.status.setText(msg + hist_note)
        except Exception as exc:
            self._handle_ui_exception("统计完成后刷新界面", exc, stop_speed=False)

    def on_check(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        self._set_busy(True)
        self.status.setText(f"检查 {org} 是否有新增数据集 ...")
        worker = CheckWorker(org, OUT_DIR, self.token)
        self.worker = worker
        self._check_worker = worker
        worker.result.connect(
            self._guarded("显示新增检查结果", self._on_check_result, pass_args=True))
        worker.error.connect(
            self._guarded("检查新增", self._on_worker_error, pass_args=True))
        worker.finished.connect(
            lambda worker=worker:
            self._on_one_shot_worker_finished("_check_worker", worker))
        worker.start()

    def _on_check_result(self, new, removed, hub_count, local_count):
        self.status.setText(
            f"Hub {hub_count} 个 / 本地 {local_count} 个，"
            f"新增 {len(new)}，本地多出 {len(removed)}")
        lines = []
        if new:
            lines.append("🆕 新增 (Hub 上有、本地未拉取):\n  " + "\n  ".join(new))
        if removed:
            lines.append("⚠️ 本地多出 (Hub 上已无):\n  " + "\n  ".join(removed))
        if not lines:
            lines.append("本地与 Hub 数据集名称一致，无新增。")
        QMessageBox.information(self, "检查结果", "\n\n".join(lines))

    def on_open_dir(self):
        latest = dd.find_latest_report(OUT_DIR)
        target = Path(latest).parent if latest else Path(OUT_DIR)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
        self.status.setText(f"已打开: {target}")

    def _on_error(self, msg):
        self._stop_speed()
        self._refresh_action_states()
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _on_worker_error(self, msg):
        """Show a worker failure; finished releases the worker/busy lock."""
        self._stop_speed()
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)


APP_ID = "tacverse-workbench"  # WM class / desktop-file base name (taskbar match)


def main():
    app = QApplication(sys.argv)
    configure_application_ui(app)
    # Taskbar/dock icon: an app-level icon plus a stable WM class that matches an
    # installed <APP_ID>.desktop, so GNOME/Ubuntu show the logo instead of the
    # generic gear. (setWindowIcon on the window alone is not enough on Linux.)
    app.setApplicationName(APP_ID)
    app.setApplicationDisplayName("TacVerse 多模态物理具身数据集工作台")
    app.setDesktopFileName(APP_ID)
    if LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
