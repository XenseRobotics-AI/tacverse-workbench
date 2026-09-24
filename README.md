# TacVerse 多模态物理具身数据集工作台

（tacverse-workbench）批量拉取 Hugging Face 组织下的数据集，统计团队产能、贡献与增长趋势。

面向数据采集团队负责人：一屏掌握「今天产出多少小时数据、谁贡献的、质量如何、增长趋势如何」。

数据基于 [LeRobot](https://github.com/huggingface/lerobot) 数据集格式（`meta/info.json`、`stats.json` 等）；上传者取自各数据集 HF 提交记录的初始 author。

---

## 功能

- **批量拉取**：自动发现某组织（默认 `TacVerse`）下全部数据集，统一同步到 `datasets/<组织名>/`，并生成聚合报告。
- **仅统计**（快）：只读取每个数据集的 `meta/info.json`（不下载数据文件），秒级获得 episodes / frames / 时长等统计。
- **手动补录统计**：从其他电脑或截图补录某一天的数据集数、episodes、frames、总小时数，参与 KPI、趋势和每日新增计算。
- **检查新增**：对比 Hub 与本地的数据集名称，列出新增 / 缺失。
- **团队看板 GUI**（PySide6），页签：
  - **看板**：KPI 卡片（数据集总数 / 总小时 / 总 episodes / 今日新增小时 / 今日新增 episodes / 目标完成度 / **今日 MVP ⭐**）+ 可排序筛选的数据集表格（含 **均时长(s)** 质量指标、robot_type、任务数、HF ID、上传者中文名、最后更新、今日新增）。表格默认按 HF「最近更新」排序，和网页一致。
  - **趋势**：每日新增小时（柱，仅显示实际拉取过的日期，不留空白）+ 累计小时（折线）。
  - **分组统计**：按 上传者 / 任务 / robot_type 维度汇总（横向柱状，中文名不重叠），并以 Hugging Face `last_modified` 归日、优先使用 commit history 差分、缺失时用本地快照兜底展示单组单日新增总时长。
  - **数据集编辑**：左侧是与看板一致的数据集详情表（选中要操作的数据集）；右侧两组功能——① **改名 / 改 Prompt**（本地 pyarrow 生成新副本，可推送回 Hub）；② 调用 lerobot 官方 `dataset_tools` 的 **删除 episodes / 拆分 / 合并 / 增加特征 / 删除特征**。详见下方[「数据集编辑」](#数据集编辑生成新副本不改动原数据)。
  - **Viewer**：内嵌 `xense_lerobot_viewer`（Next.js），3D 回放 / 语言标注 / 标签编辑；Viewer 数据根固定为 `datasets/<组织名>/`，可直接发现该组织目录下的全部本地数据集。
  - **Episode 时长定位**：选中已下载的数据集后，检查面板中的 `STATISTICS` 会按时长区间列出 episode 数量；展开区间即可查看具体 `ep`、实际秒数和 frames，用于快速定位过短或过长的数据。
  - **检查面板布局**：左侧依次显示 `ANNOTATIONS`、`FILTERING`、`ACTION INSIGHTS`，右侧显示 `STATISTICS` 和 `检查规则`；统计概要默认折叠，Episode 时长明细优先显示。
- **质量检查**：命名规范、均时长（20~600s）、Prompt 词数（10~50 词）等规则内置，看板表格用 ✅/⚠️/❌ 标注；阈值在 `config.json` 的 `checks` 段可调。
- **PICO MoTracker 轨迹检查**：在右侧「检查规则」中点击按钮后，针对当前已下载数据集按固定阈值扫描 `left_tcp` / `right_tcp` 的相邻帧；同时检查单轴位移和 XYZ 三维位移，并按 Episode 展示帧号、时间、变化量和命中的阈值。该扫描不会随切换数据集自动启动。
- **Viewer Doctor 诊断**：右侧「数据集检查分区」可切换到 Doctor 页面，调用 vendored viewer 的 TypeScript 诊断接口；支持前 N 个、全部或自定义 Episode 范围，流式显示进度，按 PASS/WARN/FAIL 展开检查结果，并可导出 JSON 报告。Doctor 与 Workbench 的 PICO 固定阈值检查相互独立。
- **双行功能工具栏**：顶部操作按职责分为两行。第一行集中 `数据源 / 数据获取 / 数据维护`，提供组织选择、刷新统计、下载选中、同步全部、检查新增、手动补录和数据目录；第二行集中 `账号 / Viewer / 目标`，显示登录状态、Viewer 服务控制、打开选中数据集、每日小时目标和当前时间。完整操作说明可通过按钮 tooltip 查看。
- **Episode 深度质量检查**：手动扫描本地数据集，或按需缓存远程检查文件；定位首尾位姿偏差、轨迹突变、长时间静止、时长离群、画面闪烁/模糊/曝光/冻结和 JPEG 日志异常。检查在后台运行并支持取消，生成 Markdown / HTML / CSV / JSON 报告、问题视频切片和逐自由度轨迹图；问题可标记为确认、误报或已修复。触觉相机不会套用普通相机的模糊和冻结规则。
- **登录状态指示器**：顶栏实时显示「已登录: xxx · 可见 N 个数据集」，一眼判断 token 权限是否正确（私有库需要有权限的账号才可见）。
- **切换账号**：顶栏按钮，运行时粘贴新的 HF token 即可切换（仅本次运行有效，不落盘）。

---

## 环境安装

本项目支持 Windows、Ubuntu/Linux 和 macOS，使用 Python 3.10 或更高版本。
**不要求创建或激活虚拟环境**：系统 Python、conda/mamba 环境或 `venv` 均可。安装依赖和启动程序时使用同一个 Python 解释器即可。

### 1. 准备 Python 与 Git

检查当前解释器：

```powershell
# Windows PowerShell（也可使用 PATH 中的 python）
py -3 --version
git --version
```

```bash
# Ubuntu / macOS
python3 --version
git --version
```

- Windows：可从 [python.org](https://www.python.org/downloads/windows/) 安装 64 位 Python；安装后若没有 `py` 命令，可将下文的 `py -3` 换成 `python`。
- Ubuntu：如未安装，执行 `sudo apt update && sudo apt install -y python3 python3-pip git`。
- macOS：可使用系统已有的 Python 3，或执行 `brew install python git`。不使用 Homebrew 时也可从 python.org 安装。

如果是首次克隆仓库，建议同时拉取 Viewer 子模块：

```bash
git clone --recurse-submodules https://github.com/dctx479/tacverse-workbench.git
```

已经克隆的仓库可在项目根目录补拉子模块：

```bash
git submodule update --init --recursive
```

### 2. 完整安装依赖

完整功能包括 GUI、统计/下载、改名 / 改 Prompt、深度质量检查、Viewer / Doctor，以及调用 LeRobot 官方 `dataset_tools` 的删除 / 拆分 / 合并 / 增删特征。安装 [Bun](https://bun.sh/) 后，在项目根目录执行：

```powershell
# Windows PowerShell
py -3 -m pip install --upgrade pip
py -3 scripts/install_all.py
```

```bash
# Ubuntu / macOS
python3 -m pip install --upgrade pip
python3 scripts/install_all.py
```

`scripts/install_all.py` 会安装 `requirements-full.txt`，初始化 `third_party/lerobot_viewer` 源码引用，并在 `vendor/lerobot_viewer_runtime` 生成运行时副本后执行 `bun install`。使用 `py -3` / `python3` 可确保依赖装到实际启动程序的解释器中。

如果当前机器暂时不需要 LeRobot 数据集操作，可改用轻量安装：

```bash
python scripts/install_all.py --skip-lerobot
```

如果只想安装 Python 依赖、不安装 Viewer，可加 `--skip-viewer`。

> 某些 Linux 发行版会限制向系统 Python 安装包。如果 `pip` 提示 `externally-managed-environment`，请使用发行版提供的 Python 包、conda/mamba，或自行选择 `venv`；这不是本项目的强制要求。

### 3. 平台依赖

Windows 和 macOS 通常无需额外的 GUI 系统库。

Ubuntu/Linux 上，PySide6 6.5+ 的 Qt xcb 平台插件需要 `libxcb-cursor0`。缺少时会报错并无法启动：

```
From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
```

安装：

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

如果仍提示缺少 xcb 库，请根据报错用当前 Linux 发行版的包管理器安装对应库；无桌面环境的服务器还需要 X11/Wayland 会话才能显示 GUI。

### 4. 单独安装 Viewer

**Viewer：** 需要 Bun 和 viewer 运行时副本。安装 [Bun](https://bun.sh/) 后，在项目根目录执行：

```bash
python scripts/install_viewer.py
```

运行时副本位于 `vendor/lerobot_viewer_runtime/`，`third_party/lerobot_viewer/` 只保留为上游源码引用。

上述命令在 Windows PowerShell、Ubuntu 与 macOS 中通用。Viewer 未安装时，其他页签仍可正常使用。

---

## HF Token 配置（拉取私有数据集的关键 ⚠️）

TacVerse 的大部分数据集是**私有**的。HF 接口只会返回「当前 token 有权限看到」的仓库——**匿名或无权限的 token 只能看到公开的少数几个**。所以拉全部数据集，必须用一个**属于该组织、有读权限**的账号 token。

### token 从哪来

用有组织权限的账号登录 https://huggingface.co/settings/tokens ，**新建 token 时 Token type 选 `Read`（经典读 token）**——它能访问该账号有权限的所有仓库（含组织私有库）。
> 不要用 fine-grained（细粒度）token，除非你在其 scope 里显式勾上了目标组织的「Read access to contents of all repos」，否则照样看不到私有库。

### 怎么让程序用上 token（任选其一）

- **方式 A（推荐，一次生效）**：命令行登录，程序会自动读取缓存的登录 token：
  ```bash
  hf auth login              # Windows / Ubuntu / macOS 通用
  ```
- **方式 B（临时）**：设置环境变量后启动。Windows PowerShell：
  ```powershell
  $env:HF_TOKEN = "hf_你的token"
  py -3 main_app.py
  ```
  Ubuntu / macOS：
  ```bash
  export HF_TOKEN=hf_你的token
  python3 main_app.py
  ```
- **方式 C（运行时）**：直接开 GUI，点顶栏 **「切换账号」** 粘贴 token。

程序取 token 的优先级：`HF_TOKEN` 环境变量 → `hf auth login` 缓存 → 匿名。

### 验证 token 是否有权限

```powershell
# Windows PowerShell
py -3 -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('TacVerse/taccap-g1-candybowl-0702').private)"
```

```bash
# Ubuntu / macOS
python3 -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('TacVerse/taccap-g1-candybowl-0702').private)"
```
能打印 `True` 说明有权限；报 `404` 说明账号 / token 权限不够（需组织管理员把你的账号加进组织，或换经典 Read token）。启动 GUI 后，顶栏指示器显示的「可见 N 个」也能直接反映权限是否正确。

---

## 用法

以下命令均在项目根目录执行。

### 命令行（批量拉取整个组织）

Windows PowerShell：

```powershell
py -3 download_dataset.py                                # 拉取默认组织全部数据集
py -3 download_dataset.py --org TacVerse                 # 指定组织
py -3 download_dataset.py --repo-id A/x --repo-id B/y    # 只拉指定数据集
```

Ubuntu / macOS：

```bash
python3 download_dataset.py                                # 拉取默认组织全部数据集
python3 download_dataset.py --org TacVerse                 # 指定组织
python3 download_dataset.py --repo-id A/x --repo-id B/y    # 只拉指定数据集
python3 download_dataset.py --out-dir datasets/TacVerse    # 指定组织级本地目录
```

### 图形界面（团队看板）

Windows PowerShell：

```powershell
py -3 main_app.py
```

Ubuntu / macOS：

```bash
python3 main_app.py
```

进去后：点 **「仅拉取统计信息」**（快，只读信息，不下载）、**「下载当前选中数据集」**（只下选中的一个，省时）或 **「拉取组织及其下所有数据集」**（全量下载 + 累积历史，较慢）。

新版顶部工具栏使用更短的按钮名称，对应关系如下：

| 按钮 | 功能 |
|---|---|
| `刷新统计` | 只获取数据集元信息，不下载 Parquet 和视频 |
| `下载选中` | 下载表格中当前选中的一个或多个数据集（Ctrl/Shift 多选，并行执行） |
| `同步全部` | 下载当前组织下全部数据集并更新统计历史 |
| `检查新增` | 对比 Hub 与本地统计记录中的数据集名称 |
| `手动补录` | 补录指定日期的累计统计快照 |
| `数据目录` | 打开本地 `datasets/<组织名>/` 目录 |
| `打开选中` | 在 Viewer 中打开当前选中的本地数据集 |

下载后的本地数据集统一存放在：

```text
datasets/<组织名>/<数据集名>/
```

不再按日期创建中间目录。这样 Viewer、任务/语言标注读取和数据集编辑都使用同一个组织级路径；统计历史中的日期仍保留在 `dataset_log.json`，仅用于趋势和每日新增计算。

### Episode 时长定位

Workbench 不显示直方图，而是直接提供可展开的时长分组：

```text
时长区间          数量
▼ 200.0–225.0s    22
    ep 166         201.3s    6039 frames
    ep 171         204.7s    6141 frames
    ep 172         211.2s    6336 frames

▶ 225.0–250.0s    27
▶ 250.0–275.0s    26
```

该信息来自本地 `meta/episodes` 元数据，按 viewer 相同的分组规则计算。此功能目前针对已下载的 LeRobot v3 数据集；仅统计、未下载或缺少 episode 元数据时，界面会显示原因。

### PICO MoTracker 轨迹检查

选中一个已下载的数据集，在右侧「检查规则」中点击 **检查 PICO MoTracker 轨迹**。结果按以下层级展开：

```text
❌ PICO MoTracker 轨迹：5 个异常事件，2 个 Episode
  Episode 0（3 个事件）
    right_tcp · frame 3752→3753 · 125.067–125.100s · x Δ-1.339，XYZ 1.502
```

当前只启用固定阈值的两项标准：

- 单轴单帧跳变：`abs(Δx)`、`abs(Δy)` 或 `abs(Δz)` 达到对应阈值；
- XYZ 三维跳变：`sqrt(Δx² + Δy² + Δz²)` 达到三维阈值。

阈值位于 `config.json` 的 `checks.pico_motracker`：

```json
"pico_motracker": {
  "source": "observation.state",
  "hands": ["left", "right"],
  "axis_step_threshold": {"x": 0.2, "y": 0.2, "z": 0.2},
  "xyz_step_threshold": 0.35
}
```

修改配置后重启 Workbench；当前版本暂不提供图形化阈值设置窗口，也不包含自适应阈值、恢复形态或加速度标准。

### Viewer Doctor 诊断

选中一个已下载的数据集，在右侧「数据集检查分区」顶部切换到 **Doctor**，然后选择检查范围并点击 **运行 Doctor**：

```text
数据检查 | Doctor
检查范围: [前 25 个 Episode]  [运行 Doctor]
进度: Running Action Quality... 65%
PASS 8   WARN 2   FAIL 1
```

Doctor 由 viewer 运行时副本提供，Workbench 通过 HTTP 接口调用，不修改 viewer 源码；`third_party/lerobot_viewer` 只作为上游源码引用。诊断包括元数据、时间一致性、动作质量、视频完整性、数据分布、Episode 健康度、特征一致性、训练准备度、异常检测、可移植性和逐 Episode 汇总等项目。全量诊断可能需要更多时间和内存，建议先使用前 25 或自定义范围。

### 数据集编辑（生成新副本，不改动原数据）

在 **「数据集编辑」** 页签，左表选中一个**已下载**的数据集（未下载的行不能编辑；可先用「下载当前选中数据集」拉下来）。所有操作都**输出为新副本**到 `datasets/<组织名>/<输出名>/`，**不会改动原数据集**。

**① 改名 / 改 Prompt（本地实现，无需 lerobot）**
- 直接编辑该数据集的 Prompt（`meta/tasks.parquet` 里的任务指令）和/或输出名，点 **「生成新副本」**。
- 只改写元数据（`meta/`），`data/` 与 `videos/` 用硬链接进副本、**不复制大文件**，秒级完成。
- 需要时点 **「推送到 Hub」** 把副本上传（默认私有仓库）。

**② 数据集操作（调用 lerobot 官方 `dataset_tools`）**
在下拉框选择操作，填参数后点 **「执行操作」**：

| 操作 | 参数 | 说明 |
|---|---|---|
| 删除 episodes | 序号，如 `0,2,5` | 删除指定 episode 并重建索引 |
| 拆分数据集 | `train:0.8,val:0.2` 或 `train:0-4,val:5-6` | 按比例或序号区间拆成多个 `<输出名>_train`… |
| 合并数据集 | 勾选多个已下载数据集 | 合并成一个（输出名用「输出数据集名」） |
| 增加特征 | 特征名 / dtype / shape / 填充值 | 新增一个常量填充的特征列 |
| 删除特征 | 勾选要删的特征 / 相机 | 移除特征（必填字段不可删） |

> 这些操作以**子进程**方式运行 lerobot，保证与官方框架一致，并把崩溃与主界面隔离。删除/拆分/合并会**重编码视频**（用 CPU `libx264`，较慢，请耐心等待）。完成后新数据集自动出现在表格里（标记为「已下载」）。

---

## 配置与数据文件

- **`config.json`**（手工维护的配置，随仓库提交）：
  - `uploader_names`：**你手工维护**的 `HF ID -> 中文名` 映射。新增成员在这里加一行 `"hf_id": "中文名"`（改完重启 GUI 生效）；查不到的 ID 在界面显示为 `未知`。
- **`dataset_log.json`**（统计历史，随仓库提交）：
  - 每次“仅拉取统计信息”或完整拉取后更新，用于恢复 KPI、每日新增和趋势；同一组织每天只保留时间最新的一次结果，后拉取的数据会覆盖当天早先记录。因此**克隆仓库的人不需要 `datasets/` 也能看到历史趋势**。
  - 顶部“手动补录统计”可写入指定日期的四项累计总量。同一组织和日期再次录入会覆盖旧的手动值；同日后续真实统计会替代手动快照。
  - 手动快照不包含逐数据集信息，因此作为最新记录时不会显示数据集表格、分组统计或 MVP；“今日新增”由它与前一个有记录日期的累计值相减得到。
- **`pull_history.local.json`**（本地运行历史，已被 git 忽略）：
  - 兼容旧版的本地快照历史文件；新版本优先使用 `dataset_log.json`。
  - 该文件只保存在本机，不提交到仓库。
- **`hf_change_history.local.json`**（本地 HF 变更缓存，已被 git 忽略）：
  - 程序以 Hugging Face `last_modified` 归日，优先使用 commit history 中 `meta/info.json` 的差分计算今日新增、MVP 和单组单日新增；缺失时使用本地快照兜底。
  - 该缓存只保存在本机，不提交到仓库。
- **`datasets/`**：拉取下来的原始数据集（含多 GB 视频），**已被 git 忽略**，不随代码同步，以节省仓库体积；组织目录下不再按日期分层。

---

## 文件

- `main_app.py` —— PySide6 团队看板（GUI 入口）。
- `download_dataset.py` —— 拉取 / 统计 / 分析 / 配置读写的核心逻辑（CLI 与 GUI 共用）。
- `checks.py` —— 数据集质量检查插件注册表（命名 / 均时长 / Prompt 等规则）。
- `dataset_quality.py` —— Qt-free Episode 深度质量扫描、问题切片、报告和复核状态持久化。
- `pico_motracker.py` —— Qt-free PICO MoTracker 固定阈值轨迹跳变检测器。
- `dataset_editor.py` —— 「改名 / 改 Prompt」的本地 pyarrow 实现（Qt-free，不依赖 lerobot）。
- `lerobot_ops.py` / `lerobot_ops_runner.py` —— 删除 / 拆分 / 合并 / 增删特征：workbench 侧封装 + 调用 lerobot `dataset_tools` 的子进程执行器。
- `config.json` —— 上传者中文名映射 + 质量检查阈值。
- `pull_history.local.json` —— 本地拉取 / 统计历史（自动生成，git 忽略）。
- `dataset_log.json` —— 自动统计和手动补录共用的轻量历史快照。
- `hf_change_history.local.json` —— 本地 Hugging Face 变更历史缓存（自动生成，git 忽略）。
