# LeRobot ACT 全流程实战手册
## 从 Featurize 环境配置 → 训练 → 真机推理（含本次实验结果）

> **适用对象**：已用 SO-100 / SO-101 机械臂采集好示教数据，在 Featurize 云算力平台上用 LeRobot 训练 ACT 策略，并把权重拿回本地真机部署。
>
> **本文档所有数字均来自实测**（数据集为 `datass`，40 条示教，两路相机）。

---

## 目录

| 章节 | 内容 |
| --- | --- |
| [1](#1-先看清你手上的数据集) | 数据集画像与 LeRobotDataset v3.0 的存储机制 |
| [2](#2-全流程总览) | 全流程总览与成本预估 |
| [3](#3-环境配置) | Featurize 注册、上传数据、租实例、装环境、搬数据 |
| [4](#4-训练) | `lerobot-train` 完整命令、参数调优、读懂日志 |
| [5](#5-取回模型权重) | 5 种把权重取回本地的方式 |
| [6](#6-本地环境准备与版本排错) | Windows 本地环境、`dtype` 版本报错、CUDA 版 torch |
| [7](#7-真机推理so-101) | 标定、推理命令、节奏报告、**安全重置环境** |
| [8](#8-常见报错速查表) | 25 条报错的现象 / 原因 / 解法 |
| [9](#9-省钱与踩坑清单) | 计费规则与消耗品清单 |
| [10](#10-本次实验结果) | ★ 训练与推理的实测结果 |
| [11](#11-一页速查复制即用) | 复制即用的命令清单 |

---

## 1. 先看清你手上的数据集

读取 `D:\Desktop\datass\meta\info.json` 后的实测画像：

| 项目 | 值 |
| --- | --- |
| 数据格式版本 | `codebase_version: v3.0`（需要 LeRobot ≥ 0.4，建议用 main 分支源码） |
| 机器人类型 | `so_follower`（SO-100 / SO-101 系列，6 自由度） |
| episode 数 | **40 条** |
| 单条时长 | 448 ~ 450 帧（均值 449），30 fps 下约 **14.97 秒/条** |
| 总帧数 | **17,961 帧**，合计约 **10 分钟**示教 |
| 任务数 | 1 个，任务文本为 **`Grab Box`**（40 条 episode 全部标注为同一任务） |
| 帧率 | 30 fps |
| 状态 / 动作维度 | 各 **6 维**：shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper |
| 相机 | 2 路：`observation.images.front`（前视）+ `observation.images.wrist`（腕部），均 480×640 |
| 视频编码 | **AV1**（`video.codec: av1`） |
| 目录体积 | **351 MiB**（≈ 367 MB） |
| 统计量 | `action` 与 `observation.state` **都含 `q01` / `q99`** —— 这一点后面微调 π0.5 时很关键 |

**结论：可以直接拿来训练，无需任何格式转换。** 40 条示教对 ACT 来说是正合适的量级。

### 1.1 LeRobotDataset v3.0 的视频存储：多条 episode 被"打包"进同一个 mp4

这是 v3.0 相比 v2.x 最大的变化，很多人会看错。用这份数据集的真实数据说明：

| 相机 | 视频文件 | 打包了哪些 episode | 文件内时间轴 |
| --- | --- | --- | --- |
| `observation.images.front` | `file-000.mp4`（203.7 MB） | **31 条**（episode 0 ~ 30） | 0 ~ 464.0 s |
| `observation.images.front` | `file-001.mp4`（59.1 MB） | **9 条**（episode 31 ~ 39） | 0 ~ 134.7 s |
| `observation.images.wrist` | `file-000.mp4`（104.2 MB） | **40 条全部** | 0 ~ 598.7 s |

要点：

- **v2.x** 是「一条 episode 一个 mp4」（`episode_000000.mp4`、`episode_000001.mp4` …）。
- **v3.0** 改成**按体积切分、多条 episode 首尾相接拼进同一个 mp4**。阈值由 `meta/info.json` 里的 `video_files_size_in_mb`（这份是 **200 MB**）决定：
  - `front`：前 31 条累计到 203.7 MB 触发切分 → 开 `file-001.mp4` 装剩下 9 条。
  - `wrist`：40 条总共才 104.2 MB，没到 200 MB → **全塞进一个文件**。
  - `data/` 下的 parquet 同理按 `data_files_size_in_mb`（100 MB）切分；17,961 帧只占 772 KB，所以只有一个 `file-000.parquet`。
- **每条 episode 在文件中的位置由 `meta/episodes/chunk-000/file-000.parquet` 记录**：

```
videos/observation.images.front/chunk_index
videos/observation.images.front/file_index      → 该 episode 在第几个 mp4 里
videos/observation.images.front/from_timestamp  → 在该 mp4 中的起始秒
videos/observation.images.front/to_timestamp    → 结束秒
```

  实测前 3 条：episode 0 = `0.000 → 14.967s`，episode 1 = `14.967 → 29.933s`，episode 2 = `29.933 → 44.933s` —— 首尾严丝合缝，**没有间隙也没有填充帧**（14.967 s = 449 帧 ÷ 30 fps）。

**这对你意味着什么：**

- **不需要自己切视频。** LeRobot 读取时会按 `from_timestamp` / `to_timestamp` 精确 seek，语义上等价于"每条 episode 一个视频"。
- 好处：40 条数据只有 3 个 mp4，**文件数大幅减少**，云盘挂载时的小文件开销几乎消失。
- 注意：因为多条 episode 共用一个文件，**读视频需要 seek 定位**。自己写脚本解析或做可视化时，必须读 `meta/episodes/*.parquet` 的时间戳，**不能按文件名对应 episode**。
- 想直观浏览：`lerobot-dataset-viz --repo-id=local/datass --root=/home/featurize/work/datasets/datass`（需 `pip install -e ".[dataset_viz]"`）。

### 1.2 一个容易产生误会的地方：MiB vs MB

上面「目录 351 MiB」和「视频 367 MB」看起来矛盾，其实是单位不同：

```
203.7 + 59.1 + 104.2 = 367.0 MB（10^6 字节）
367.0 × 10^6 ÷ 2^20   = 350.0 MiB（2^20 字节）
```

Windows 资源管理器显示的是 **MiB**（但它写"MB"），LeRobot 的 `info.json` 里用的是 **MB**。差 ~4.7%，不是数据丢了。

---

## 2. 全流程总览

```
本地打包 datass.zip
      ↓
① 上传到 Featurize「云端数据集」          （351 MiB，几分钟）
      ↓
② 租 GPU 实例（RTX 4090 性价比最高）      （¥1.87/小时）
      ↓
③ 装环境：conda + ffmpeg + lerobot[training]
      ↓
④ 在实例里下载数据集到 ~/work/datasets/datass
      ↓
⑤ wandb login（可选，看训练曲线）
      ↓
⑥ lerobot-train 开始训练                  （实测 5 万步 ≈ 74 分钟）
      ↓
⑦ 打包下载 checkpoints → 回本地真机推理
```

**实测总花费：训练 5 万步约 ¥2.3**（RTX 4090，74.3 分钟）；加上环境搭建与调试，整条链路 **¥5 ~ ¥10**。

---

## 3. 环境配置

### 3.1 注册 Featurize

1. 打开邀请链接注册：`https://featurize.cn?s=d7ce99f842414bfcaea5662a97581bd1`
2. 微信扫码登录。
3. **加官方用户群，跟客服说明来自教程，领取代金券**（别忘了这一步）。
4. 学生认证可额外拿 4090 单卡 9.9 元日租券；首充满 5 元送 1 小时体验券。

### 3.2 本地打包数据集

1. 打开 `D:\Desktop`，右键 `datass` 文件夹 → **压缩为 ZIP**，得到 `datass.zip`。
2. ⚠️ 压缩包里**第一层必须就是 `datass` 文件夹本身**（解压后是 `datass/meta/info.json`，而**不是** `meta/info.json`）。
3. 351 MiB，系统自带压缩即可，不用分卷。

> **可选：不打包，直接传目录。** `featurize dataset upload` 支持直接上传整个目录（压缩算法比 zip/gz 更高效，支持断点续传和多线程）。Windows 11 把可执行文件放进 `C:\Users\$USERNAME\AppData\Local\Microsoft\WindowsApps` 就能全局调用。

### 3.3 上传到 Featurize「云端数据集」

Featurize 数据集是永久保存在你账号下的云存储，**上传一次可反复下载，不占同步盘**。

1. 登录 → 左侧导航 **「数据集」** → **「新建数据集」**。
2. 起名（例如 `lerobot_datass_2026`），备注写清：机器人 = SO-101，任务 = Grab Box，40 条。
3. 把 `datass.zip` 拖进上传区（351 MiB 通常 1~5 分钟）。
4. 上传完成后，点 **「查看」→「复制实例下载命令」**，拿到类似：

```bash
featurize dataset download 7f40bdaa-b1a4-4c00-9652-ff26fd079109
```

> 把这行命令存到记事本，第 3.6 步要用。**UUID 换成你自己的。**

**替代方案（二选一）：**

- **A. JupyterLab 拖拽**：进工作区后把 `datass.zip` 拖进左侧文件区。351 MiB 也很快，但**每次换实例都要重传**。
- **B. 走 HuggingFace Hub**：本地 `hf upload <用户名>/datass ./datass --repo-type=dataset`，实例里直接 `--dataset.repo_id=<用户名>/datass`，连解压都省了。适合要反复换机器的情况。

### 3.4 租用 GPU 实例

| 显卡 | 价格 | 显存 | 是否推荐 |
| --- | --- | --- | --- |
| **RTX 4090** | ¥1.87/小时 | 24 GB | ⭐ **首选**，ACT 实测 74 分钟，性价比最高 |
| RTX 3090 | ¥1.62/小时 | 24 GB | 预算更紧可选，CPU 只有 6 核，注意 `--num_workers` |
| PRO 6000 96G | ¥6.19/小时 | 96 GB | 训 ACT 纯属浪费；除非要顺便试 π0.5 / SmolVLA |
| RTX 2080 Ti 22G | ¥0.69/小时 | 22 GB | 最省钱，但训练时间明显拉长 |

**选择镜像**：选平台「App 推荐」里 **PyTorch 2.x + CUDA 12.x + Python 3.12** 的镜像。LeRobot 0.6.x 要求 `python >= 3.12`、`torch >= 2.7`，别选 Python 3.8/3.10 的老镜像。

> **关于「要不要单独选 Ubuntu 系统」—— 不需要。**
>
> Featurize 对镜像的定义是：**「镜像是实例的预装环境」**。一个镜像里已经打包好了「操作系统 + 驱动 + 框架 + 开发工具」，你选哪个镜像，系统和软件栈就同时定下来了，**没有第二步**。
>
> Featurize 的 GPU 实例跑的都是 **Ubuntu 系 Linux**，JupyterLab / SSH / VSCode 远程全部开箱即用，你不会看到"Ubuntu / CentOS / Windows"这种独立的系统下拉框。
>
> 对照理解：「先选系统镜像、再选应用环境」那种**两级选择**是阿里云 / 腾讯云轻量服务器（镜像市场）的交互模式，Featurize 不是这套逻辑。
>
> **进阶**：把 conda 环境和依赖都调好后，可在「我租用的实例」页面点 **「创建镜像」**，把当前实例整个存成自己的镜像，之后新建实例直接选它，环境秒级就绪。（注意：创建镜像时会自动删掉 `~/.cache`、apt / conda 缓存来省空间；创建期间实例不能使用或归还。）

**存储规则（务必记住，搞错会丢数据）：**

- 只有 **`/home/featurize/work`（同步盘）** 是持久化的，默认 30 GB 免费。
- 实例**本地磁盘**在「归还实例 / 重置」后会被清空；`/tmp`、`/dev/shm` 随时可能丢。
- 所以：**代码、数据集、模型权重全部放 `~/work` 下**。

点「开始使用」→ 等实例就绪 → 「打开工作区」进 JupyterLab → 左上角 `File → New → Terminal`。

### 3.5 安装环境

在实例终端里逐段执行：

```bash
# 1) 建一个干净环境
conda create -y -n lerobot python=3.12
conda activate lerobot

# 2) 装 ffmpeg（LeRobot 读写视频强依赖，必须装）
conda install ffmpeg=7.1.1 -c conda-forge -y

# 3) 拉 LeRobot 源码
cd ~/work
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 4) 装训练依赖
pip install -e ".[training]"
pip install wandb --upgrade

# 5) 国内网络加速 + 登录 HuggingFace
export HF_ENDPOINT=https://hf-mirror.com
hf auth login          # 老版本命令是 huggingface-cli login

# 6) 验证 GPU 与环境（必须输出 True）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

#### ⚠️ 关于安装参数：不要用 `[pi]`

原始文档写的是 `pip install -e ".[pi]"`，但对照 `pyproject.toml` 可以确认：

- **`[pi]` 只装 π0 / π0.5 系列模型的依赖**（transformers、scipy、sentencepiece），**不包含训练所需的 `datasets` / `pyarrow` / `wandb` / `accelerate`**。
- 只训 ACT → **请用 `[training]`**（内部 = `dataset` + `wandb` + `accelerate`）。
- 想一步到位、以后还想试 π0.5：

```bash
pip install -e ".[pi,training]"
```

#### AV1 视频的坑

这份数据的视频编码是 **AV1**。新版 LeRobot 默认用 `torchcodec` 解码，某些环境对 AV1 支持不好会直接报错。**如果训练时报视频解码相关错误**，加一个参数切到 PyAV 后端：

```bash
--dataset.video_backend=pyav
```

（conda 装的 ffmpeg 7.1.1 对 AV1 支持完整，这一步大概率用不上，但先记住。）

### 3.6 把数据集搬进实例

```bash
# 1) 建目录
mkdir -p ~/work/datasets
cd ~/work

# 2) 粘贴第 3.3 步复制到的下载命令（UUID 换成你自己的）
featurize dataset download 7f40bdaa-b1a4-4c00-9652-ff26fd079109

# 3) 解压
unzip datass.zip -d ~/work/datasets/

# 4) 检查，应能看到 data / meta / videos 三个文件夹
ls ~/work/datasets/datass
```

#### 四种把数据弄进实例的方式

| 方式 | 怎么做 | 适合场景 |
| --- | --- | --- |
| **① 工作区可视化（推荐）** | 工作区左下角 **Featurize 图标 → 添加数据集**，搜索并点「下载」。进度在底部「后台任务」里看，可中途取消 | 不想敲命令、想看进度条 |
| **② 命令行下载** | `featurize dataset download <UUID>` + `unzip` | 用 SSH，或想脚本化 |
| **③ 命令行流式解压** | 把 `download` 换成 `featurize dataset extract` | 大数据集（省磁盘、更快） |
| **④ 只读挂载** | `featurize dataset mount <数据集ID>`，默认挂到 `/featurize/datasets/<数据集ID>` | 超大 / 不需改动的数据集 |

关于 ③：`featurize dataset download` 会先把原始压缩包落到本地磁盘再解压，**至少占 2 倍磁盘**（官方举例：200 GB 数据集要 400 GB 磁盘）。`extract` 是**流式解压**，不落地原始文件。

关于 ④：挂载目录**只读**，文件**首次访问时才自动下载并缓存到本地磁盘**，之后就是本地读取速度。训练是只读的，所以这样也能跑。

> 补充：Featurize 的数据集下载速度是**万兆**（1 GB 约 10 秒）。正确做法是「一次上传到云端数据集，之后每台新实例都秒下」，而不是在实例之间来回搬。

解压后正确路径：

```
/home/featurize/work/datasets/datass/
├── data/chunk-000/file-000.parquet
├── meta/info.json          ← 训练配置会读这个
├── meta/stats.json
├── meta/tasks.parquet
├── meta/episodes/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.front/chunk-000/file-000.mp4, file-001.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

#### 强烈建议：先跑一次加载校验

```bash
python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/datass", root="/home/featurize/work/datasets/datass")
print("episodes:", ds.num_episodes)   # 期望 40
print("frames  :", ds.num_frames)     # 期望 17961
print("fps     :", ds.fps)            # 期望 30
print("features:", list(ds.features.keys()))
s = ds[0]
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items()})
EOF
```

能打印出 `episodes: 40`、两路 `observation.images.*` 和 `action`，说明数据没问题。

> ⚠️ `--dataset.root` 要指向**直接包含 `meta/` 的那一层目录**（也就是 `datass` 本身），不能指到 `datasets` 或 `datass/meta`。

### 3.7 登录 wandb（可选，强烈建议）

```bash
wandb login
```

终端提示粘贴 API Key：去 `https://wandb.ai/authorize` 复制，粘贴回车。成功后会显示 `Currently logged in as: <你的用户名>`。

**国内连不上 wandb**：把训练命令里的 `--wandb.enable=true` 改成 `false`，靠终端打印的 loss 也能判断收敛，只是没有网页曲线。

---

## 4. 训练

### 4.1 核心命令（实测可用）

```bash
cd ~/work/lerobot
export HF_ENDPOINT=https://hf-mirror.com

lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.type=act \
  --output_dir=outputs/train/act_datass \
  --job_name=act_datass \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=8
```

#### 参数逐条解释

| 参数 | 含义 | 针对本数据集的建议 |
| --- | --- | --- |
| `--dataset.repo_id` | 数据集标识。写 `local/xxx` 即可，**不会真的上传到 HuggingFace** | `local/datass` |
| `--dataset.root` | 本地数据集目录，必须指向含 `meta/` 的那一层 | `/home/featurize/work/datasets/datass` |
| `--policy.type` | 策略类型 | `act` |
| `--output_dir` | 权重、日志、训练配置的输出目录（**相对路径**） | `outputs/train/act_datass` |
| `--job_name` | 任务名，wandb 上显示的名字 | 随便起，别用中文 |
| `--policy.device` | 训练设备 | `cuda` |
| `--policy.push_to_hub` | 是否自动推到 HuggingFace | `false` |
| `--wandb.enable` | 是否上报 wandb | `true`；网络不通改 `false` |
| `--batch_size` | 批大小 | 4090 24G 用 `8` 稳（实测显存仅 3.73 GB，可试 `16`~`32`） |
| `--steps` | 训练总步数 | **50000** 足够（实测 74 分钟）；想更充分可上 100000 |
| `--save_freq` | 每隔多少步存一次权重 | `5000` |
| `--num_workers` | 数据加载进程数 | **按机型 CPU 核数给**，见 4.2 |
| `--log_freq` | 每多少步打印一次日志 | `100` |

> **关于「修改权重保存频率」**：旧文档让你去改 `lerobot/src/lerobot/configs/train.py` 里的 `save_freq`。**新版 LeRobot 不需要改源码**，直接命令行加 `--save_freq=5000` 即可。好处是训练早期就能拿到可用权重，中途中断也不至于全丢。

### 4.2 数据加载调优：`--num_workers` 到底该给多少

**先纠正一个常见误解：`--num_workers` 不是"越大越快"，它的上限由你租的实例有几个 CPU 核决定。**

LeRobot 默认 `num_workers=4`。每个 worker 是独立进程，各自解码视频、缩放图像。给多了会：

- 抢 CPU 时间片 → 主进程准备 batch 变慢，**GPU 反而被拖累**
- 进程切换开销、内存带宽争用
- 每个 worker 内存叠加。而且默认 `dataloader_multiprocessing_context="spawn"`（官方注释：`spawn` 比 `fork` 更安全，因为 PyAV / torchcodec / ffmpeg 不是 fork-safe），**spawn 模式下每个 worker 都要重新 import torch**，启动开销随 worker 数线性增长

Featurize 各机型 CPU 核数差别很大，`--num_workers` 必须跟着机型变：

| Featurize 机型 | CPU | 建议 `--num_workers` |
| --- | --- | --- |
| RTX 3090 | **6 核** Xeon Gold 6142 | **4 ~ 5**（给 8 是超卖） |
| RTX 4090 | 14 核 AMD EPYC 9354 | **8 ~ 12** |
| RTX 4090 48G | 14 核 AMD EPYC 7453 | **8 ~ 12** |
| RTX PRO 6000 96G | 30 核 AMD EPYC 9J14 | **12 ~ 16** |

经验公式：**`num_workers ≈ CPU 物理核数 × 0.75`**，给主进程留 1~2 核。

#### 先诊断，再调，不要盲调

1. **看 GPU 利用率**：`nvidia-smi -l 1`。长期低于 80% 或呈锯齿状（算一批冲高、取下一批掉到接近 0）→ 加 worker 有效；稳定 95% 以上 → 加 worker 没用，白烧 CPU。
2. **看训练日志里的两个耗时**：LeRobot 每 `log_freq` 步打印 `updt_s`（模型前向 + 反向，GPU 干的活）和 `data_s`（DataLoader 取数）。**两者同量级甚至 `data_s` 更大就是数据瓶颈**；若 `data_s` 只有 `updt_s` 的十分之一，别动它。

> **⚠️ 实测修正（重要）**：早期推断「AV1 双路解码 + 36 万帧，大概率是数据加载瓶颈」—— **被实测推翻了**。
>
> 在这份数据上跑 ACT 时，`data_s` 稳定在 **0.004 s**、`updt_s` 约 **0.080 s** —— 取数据只占单步耗时的约 **5%**，GPU 忙碌占比达 **94%**。这份数据在实际训练中**根本不是数据加载瓶颈**，`--num_workers=8` 已绰绰有余，再加不会有任何提速；把数据集拷到本地盘对本例也没有明显收益。
>
> **正确姿势：先量出 `data_s / updt_s` 的比值，再决定要不要动。** 比值 < 0.2 就别碰了。

#### 这两个参数不用手动加（新版默认就是最优）

```python
prefetch_factor: int = 4          # 每个 worker 预取 4 个 batch
persistent_workers: bool = True   # worker 常驻，避免重复重建（配合 spawn 尤其重要）
```

#### 比加 worker 更立竿见影的一招：把数据集挪到实例本地盘

`/home/featurize/work` 是**跨实例同步的网络盘**（官方文档自己说过"直接读云盘性能极低"）。训练要反复随机 seek 解码视频，走网络存储会拖后腿：

```bash
# 训练前：拷到实例本地磁盘（不在 ~/work 下）
mkdir -p ~/localdata
cp -r ~/work/datasets/datass ~/localdata/

# 训练时把 root 指向本地盘
--dataset.root=/home/featurize/localdata/datass
```

权重只有几十 MB，继续留在 `~/work` 就行；**只有大数据集值得为速度放本地盘**。（代价：归还实例后本地盘清空，所以 `~/work` 里务必保留一份原始副本。）

#### 推荐调优顺序

**① 数据集放本地盘 → ② 按机型给 `num_workers` → ③ 再看 `nvidia-smi` 微调。**

#### 整合后的完整命令（以 RTX 4090 为例）

```bash
lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/localdata/datass \
  --policy.type=act \
  --output_dir=outputs/train/act_datass \
  --job_name=act_datass \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=10 \
  --prefetch_factor=4 \
  --persistent_workers=true
```

### 4.3 防掉线、断点续训

网页端 JupyterLab 断连可能导致进程被杀，建议用 tmux 或 nohup：

```bash
# 方式一：tmux（推荐，可随时回来看）
tmux new -s train
# 在 tmux 里执行训练命令，然后按 Ctrl+B 再按 D 退出会话
# 回来查看：tmux attach -t train

# 方式二：nohup
nohup lerobot-train ... > train.log 2>&1 &
tail -f train.log
```

也可以用 Featurize 的 SSH / VSCode 远程连接，比网页稳。

**断点续训**：

```bash
cd ~/work/lerobot
lerobot-train --resume=true \
  --config_path=outputs/train/act_datass/checkpoints/last/pretrained_model/train_config.json
```

### 4.4 读懂训练日志（附本数据集实测基线）

终端会每 `--log_freq` 步打印一行：

```text
step:1200 smpl:10K ep:21 epch:0.53 loss:1.709 grdn:65.141 lr:1.0e-05
data_s:0.003 prep_s:0.000 updt_s:0.080 step_s:0.084 smp/s:95 mem_gb:3.73
l1_loss:0.322 kld_loss:0.139
```

| 字段 | 含义 | 怎么读 |
| --- | --- | --- |
| `step` | 优化器步数（不是帧数） | 总目标由 `--steps` 决定 |
| `smpl` | 累计喂给模型的样本数 = `step × batch_size` | batch 8 时，10K 样本 ≈ 1250 步 |
| `ep` | 累计读过的 episode 数 | 数据集被帧级打散，涨得比 epoch 快 |
| `epch` | 数据集"轮数" = `smpl ÷ 总帧数` | 17961 帧时，40 万样本 ≈ 22 轮 |
| `loss` | **总损失 = `l1_loss` + 10 × `kld_loss`** | ⚠️ 别只看它，见下 |
| `l1_loss` | 动作预测的 L1 误差 | ✅ **这才是"学得好不好"的指标** |
| `kld_loss` | CVAE 隐变量正则项 | 单任务数据下趋近 0，正常 |
| `grdn` | **裁剪前**的梯度范数 | 几百 → 几十并稳定即可 |
| `lr` | 学习率 | ACT 用自己的 preset，**全程恒定 1e-5** |
| `data_s` | DataLoader 取一个 batch 的耗时 | 与 `updt_s` 比，判断是否数据瓶颈 |
| `prep_s` | 数据搬到 GPU 的耗时 | 一般 < 1 ms，可忽略 |
| `updt_s` | 前向 + 反向 + 优化器更新的耗时 | GPU 真正干的活 |
| `step_s` | 单步总耗时 | `updt_s / step_s` ≈ GPU 忙碌占比 |
| `smp/s` | 样本吞吐 | = `batch_size / step_s` |
| `mem_gb` | 显存占用 | 判断 batch_size 还能加多少 |

#### 关键：`loss` 不等于模仿质量

ACT 的损失定义是 `重建损失 + kl_weight × KL 损失`，LeRobot 中 `kl_weight = 10.0`（已在 `config.json` 中核实）。用真实日志验算：

| step | l1_loss | kld_loss | 验算 | 日志里的 loss |
| --- | --- | --- | --- | --- |
| 100 | 0.712 | 0.869 | 0.712 + 10×0.869 = **9.402** | 9.397 |
| 200 | 0.571 | 0.309 | 0.571 + 10×0.309 = **3.661** | 3.658 |
| 1200 | 0.322 | 0.139 | 0.322 + 10×0.139 = **1.712** | 1.709 |

（零头差异来自日志的浮点显示舍入。）

**所以 `loss` 里约 85% 是 KL 项。** KL 从 0.869 掉到 0.139，贡献了 loss 下降的绝大部分 —— 但那只是"隐变量被关掉"，不等于动作学得更准。**判断学得好不好，盯 `l1_loss`**：1200 步内从 0.712 降到 0.322，减半，说明动作预测在稳定变准。

#### 该盯什么、什么时候可以测真机

1. **盯 `l1_loss`，不是 `loss`。** 曲线进入平台期（连续几千步不再明显下降）才是收敛信号。
2. **不必等全部跑完。** `--save_freq=5000` 每 5000 步落一个 checkpoint，第一个出来后就能下载去真机测一次。
3. **最终看真机成功率，不看 loss。** 建议在 5K / 10K / 50K 三个 checkpoint 上各测 10~20 次，成功率最高的那个才是你要的模型 —— loss 最低的不一定是。

---

## 5. 取回模型权重

**先解释原因**：JupyterLab 文件浏览器**只对「文件」提供 Download，对「文件夹」不提供**。你对 `checkpoints`、`last`、`pretrained_model` 这些**目录**右键，菜单里是没有下载项的 —— 这不是 bug。

### 5.0 先在终端把事实钉死

界面的文件树容易让人怀疑人生，先确认文件真的存在：

```bash
# ① 定位训练输出到底在哪（--output_dir 是相对路径，取决于敲命令时所在目录）
find /home/featurize -maxdepth 8 -type d -name "act_datass" 2>/dev/null

# ② 看有哪些 checkpoint
ls -la <上面找到的路径>/checkpoints/

# ③ 确认权重文件真的落盘（最硬的证据）
find <上面找到的路径> -name "model.safetensors" -exec ls -lh {} \;

# ④ 看总占用
du -sh <上面找到的路径>
```

### 5.1 「点不进去子文件夹」的三个原因

1. **要双击。** JupyterLab 文件浏览器里**单击只是选中高亮，双击才进入目录** —— 这是最常见的原因。
2. **文件树没刷新。** 后台进程新建的目录不会自动出现，点工具栏的**刷新按钮（↻）**，或空白处右键 → Refresh。
3. **路径不对。** `--output_dir=outputs/train/act_datass` 是**相对路径**。如果敲命令时不在 `~/work/lerobot` 下，目录就落在别处了。用上面的 `find` 一查便知。

### 5.2 五种下载方式

#### 方案 A — 逐个文件下载（最快，零配置）

点进 `checkpoints/last/pretrained_model/`，里面**一共 7 个文件**，逐个右键 → Download：

```
config.json                                                     ← 策略结构配置
model.safetensors                                               ← 权重本体（实测 206.7 MB）
policy_preprocessor.json                                        ← 预处理流程定义
policy_preprocessor_step_3_normalizer_processor.safetensors     ← 归一化统计量（mean/std）
policy_postprocessor.json                                       ← 后处理流程定义
policy_postprocessor_step_0_unnormalizer_processor.safetensors  ← 反归一化（把动作还原成关节角）
train_config.json                                               ← 训练配方（续训/复现用，很小）
```

> ⚠️ **必须 7 个文件一起拷，不能只拷 `model.safetensors` + `config.json`。**
>
> 训练时对 state/action 做了 `MEAN_STD` 归一化，推理时必须用同一套 mean/std 反归一化回关节角 —— 这些统计量存在 `policy_preprocessor_*.safetensors` / `policy_postprocessor_*.safetensors` 里。**缺了它们要么直接报错，要么输出完全错误的动作（机械臂乱飞）**，而后者更危险。
>
> 最省事：直接把整个 `pretrained_model` 文件夹打包拷走，别挑文件。

#### 方案 B — 先打包，再下一次（推荐）

```bash
cd ~/work
tar -czf act_datass_last.tar.gz \
  -C ~/work/lerobot/outputs/train/act_datass/checkpoints/last pretrained_model
```

然后回文件树对 `act_datass_last.tar.gz` 右键 → Download（这次是**文件**，所以有下载项）。

> `tar` 是 Ubuntu 自带；想用 `zip` 需先 `sudo apt install zip`；也可以试 `featurize compress`。

#### 方案 C — VSCode 远程连接（批量最舒服）

连上实例后，在 VSCode 资源管理器里对**任意文件夹**右键 → **Download**，会自动打包给你。下多个 checkpoint 时最省事。

#### 方案 D — 纯浏览器，走端口转发

```bash
cd ~/work/lerobot/outputs/train/act_datass/checkpoints/last
python -m http.server 8000 --bind 0.0.0.0
```

另开终端：

```bash
featurize port export 8000
```

返回形如 `workspace.featurize.cn:48625` 的地址，浏览器打开即可。

> ⚠️ **必须绑定 `0.0.0.0`。** 绑成默认的 `127.0.0.1` 是访问不到的（官方文档明确提示过）。

#### 方案 E — 推到 HuggingFace Hub（最正统，跨机器复用）

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf upload <你的HF用户名>/act_datass \
  outputs/train/act_datass/checkpoints/last/pretrained_model --repo-type=model
```

以后在任何机器上都能直接 `--policy.path=<你的HF用户名>/act_datass`；也可以在训练时就加 `--policy.push_to_hub=true --policy.repo_id=...` 自动上传。

> ⚠️ **不要**想着用 Featurize 的「数据集」功能把模型传回本地 —— 官方明确说明"不提供对外下载数据集"。数据集是**单向**的（本地上传 → 实例下载）。

### 5.3 三个容易踩的坑

1. **别下整个 `outputs/`。** 里面除了 checkpoint 还有 wandb 日志和所有中间权重。`--save_freq=5000` + 50000 步会落 **10 个 checkpoint**，每个含模型 + 优化器状态，**单个目录接近 1 GB，合计可能十几 GB**。
2. **只要 `pretrained_model/` 子目录**即可（除非要换机器续训，才需要优化器状态 —— AdamW 两个动量是参数量的 2 倍，比模型本身还大）。
3. **注意同步盘配额**：这些 checkpoint 放 `~/work`（30 GB 免费额度）里会实实在在占空间，超出按 ¥0.001/GB/小时计费。确认不再续训后可清掉中间 checkpoint 只留 `last` —— 但**务必先确认本地或 HF 上已有备份**，删掉就真没了。

---

## 6. 本地环境准备与版本排错

### 6.1 真机电脑上装推理依赖

```bash
pip install -e ".[core_scripts,feetech]"
```

### 6.2 `The fields 'dtype' are not valid for ACTConfig`（最容易撞上的第一个报错）

```text
draccus.utils.DecodingError: The fields `dtype` are not valid for ACTConfig
```

**不是模型坏了**，而是**本地 lerobot 版本和 Featurize 上训练时用的版本不一致**。

原理：checkpoint 里的 `config.json` 是**训练时那个版本的代码**序列化出来的，包含当时全部配置字段。本地版本更旧，就不认识新出现的字段，draccus 解析配置时直接抛错。（`dtype` 是较新版本才加入 `PreTrainedConfig` 的基础字段。差得多的版本会一次报一串，比如同时报 `device`、`use_amp`、`use_peft`。）

#### 修法 1（根治，需要 Git）：对齐版本

先在 Featurize 实例上记下训练时的 commit：

```bash
cd ~/work/lerobot && git log -1 --format="%h %cd"
```

再在本地 Windows 的 lerobot 目录里比对并更新：

```cmd
cd <你的 lerobot 目录>
git log -1 --format="%h %cd"
git pull
```

如果本地当初是用 `pip install lerobot` 从 PyPI 装的（很可能落后于 main）：

```cmd
pip uninstall lerobot -y
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[core_scripts,feetech]"
```

#### 修法 2（应急，1 分钟，**不需要 Git**）：删掉被点名的字段

手工改 JSON 很容易把逗号改坏，**推荐用一条命令安全地删**（先备份）：

```cmd
copy "D:\Desktop\pretrained_model\config.json" "D:\Desktop\pretrained_model\config.json.bak"

python -c "import json,pathlib; p=pathlib.Path(r'D:\Desktop\pretrained_model\config.json'); c=json.loads(p.read_text(encoding='utf-8')); c.pop('dtype',None); p.write_text(json.dumps(c,indent=4,ensure_ascii=False),encoding='utf-8'); print('dtype removed')"
```

它只删 `dtype` 一个键，其余字段和顺序原样保留，不会破坏 JSON 结构。跑完直接重试推理命令即可。

> **小技巧**：draccus 会把**所有**不合法字段一次性列出来。你的报错里**只有 `dtype` 一个**，说明本地版本只差了一点点（大约就差一个 commit），所以修法 2 大概率能直接跑通。如果一次报出一长串，说明版本差得多，老实走修法 1。

#### 本地没装 Git 怎么办

```cmd
'git' 不是内部或外部命令，也不是可运行的程序或批处理文件。
```

三个选择：

**① 装 Git for Windows** —— 从 [git-scm.com/download/win](https://git-scm.com/download/win) 下载，一路默认安装。⚠️ **装完必须关掉当前 cmd 窗口重新开一个**，否则 PATH 不刷新。用 `git --version` 验证。

**② 不装 git，也能对齐最新 main** —— 浏览器下载源码包：

```text
https://github.com/huggingface/lerobot/archive/refs/heads/main.zip
```

解压到 `D:\Desktop\lerobot-main`，然后：

```cmd
cd /d D:\Desktop\lerobot-main
pip uninstall lerobot -y
pip install -e ".[core_scripts,feetech]"
```

> ⚠️ 这种情况下**不能**用 `pip install git+https://...` —— 那个语法本身就依赖 git。

**③ 干脆先不折腾版本**，直接走修法 2（删字段），1 分钟就能跑起来。

#### 不依赖 git 的版本查询

```cmd
pip show lerobot
```

看 `Version`（版本号）和 `Location`（安装位置）两行就够了 —— 如果 `Location` 指向 `D:\Desktop\lerobot`，说明是那个目录的 editable 安装。

再确认那个目录到底是不是 git 仓库：

```cmd
if exist D:\Desktop\lerobot\.git (echo [是] git 仓库) else (echo [否] 不是 git 仓库，可能是解压的 ZIP)
```

#### 记住这条通用规律

**训练用的 lerobot 版本和推理用的必须一致。** 以后不管是换机器、还是过几个月再回来跑，第一件事都是确认两边版本对得上 —— 否则 checkpoint 里的 `config.json` 就是一颗定时炸弹。

### 6.3 `Device 'cuda' is not available. Switching to 'cpu'`：显卡好着，但 torch 是 CPU 版

先分清是**没显卡**还是**装了 CPU 版 torch**：

```cmd
nvidia-smi
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

**实测案例（本机）**：

```
nvidia-smi              → NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, 驱动 546.30 / CUDA 12.3   ✅ 显卡没问题
torch                   → 2.11.0+cpu        ← 罪魁祸首，注意 +cpu 后缀
torch.version.cuda      → None
cuda.is_available()     → False
device_count            → 0
torchvision             → 0.26.0+cpu
```

**`+cpu` 后缀就是证据** —— `pip install torch` 时 pip 从 PyPI 解析到的是 CPU-only 构建。

#### 修复（版本号锁定，避免和 lerobot 的依赖约束打架）

```cmd
:: ① 卸掉 CPU 版
pip uninstall -y torch torchvision

:: ② 装 CUDA 版（cu126 对应 546.30 驱动）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126

:: ③ 验证
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望看到 `2.11.0+cu126 True NVIDIA GeForce RTX 3050 ...`。

**为什么是 `cu126`**：驱动原生支持 CUDA 12.3，CUDA 12.x 家族内部向下兼容，取最接近的档位最稳。各索引的 wheel 可用性（cp312 / win_amd64）实测：

| 索引 | torch 2.11.0 | torchvision 0.26.0 |
| --- | --- | --- |
| cu124 | ✗ 无 | — |
| **cu126** | ✓ | ✓ |
| cu128 | ✓ | ✓（项目 `pyproject.toml` 里的默认源就是它） |
| cu129 | ✗ 无 | — |
| cu130 | ✓ | ✓ |

版本号必须带 `==`：lerobot 的约束是 `torch>=2.7,<2.12.0`、`torchvision>=0.22.0,<0.27.0`，锁死同版本最安全。另外 **CUDA wheel 有 2~3 GB**（自带 CUDA 运行时 DLL），下载会慢。

#### 修好之后要做真实前向冒烟测试

`is_available()` 返回 True 不代表真能算，用 ResNet18（ACT 的主干）喂两张 480×640 图实测：

```cmd
python -c "import torch,torchvision.models as M,time; net=M.resnet18(weights=None).cuda().eval(); x=torch.randn(2,3,480,640,device='cuda'); torch.cuda.synchronize(); 
import contextlib; 
[net(x) for _ in range(3)]; t=time.time(); [net(x) for _ in range(20)]; torch.cuda.synchronize(); print('resnet18 前向 x20 :', round((time.time()-t)/20*1000,1), 'ms/次'); print('峰值显存:', round(torch.cuda.max_memory_allocated()/1024**3,2), 'GB')"
```

**本机实测**：`resnet18 前向 x20 : 15.3 ms/次`（batch = 2 张 480×640），峰值显存 `0.13 GB`。

> ⚠️ **别用这个数字去外推 ACT 整模型的耗时。** 真机 rollout 的实测数据显示，一次完整的 ACT 重规划（两张图预处理 + 主干 + Transformer 编解码 + 后处理）在 4 GB 的 3050 Laptop 上要 **200 ~ 560 ms**。好消息是 ACT 每 100 步才重规划一次，摊到每个 tick 只有 ~7 ms，`--fps=30` 依然跑得满。

**⚠️ 一个容易忽略的坑：换 torch 会让 `pip` 顺带检查一堆依赖。** 如果安装被中断（比如关掉终端），可能出现"torch 卸载了但没装上"的裸奔状态，此时任何 `import torch` 都会失败。重跑上面第 ② 步即可恢复。

> 小提示：`pip` 卸载/安装时会遍历几万个文件，**看起来"没进度"是正常的**（输出被缓冲了），别以为卡死就 Ctrl+C —— 中断会留下上一条说的裸奔状态。

---

## 7. 真机推理（SO-101）

### 7.1 标定：`--robot.id` 是生死线

#### 标定文件的命名规则（记死这一条）

```
Windows: %USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\<robot.id>.json
Linux  : ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<robot.id>.json
```

源码依据（本机 `D:\Desktop\lerobot`）：

```python
# src/lerobot/robots/robot.py:49-53
self.calibration_dir = config.calibration_dir if config.calibration_dir \
    else HF_LEROBOT_CALIBRATION / ROBOTS / self.name      # ← 是 self.name，不是 --robot.type
self.calibration_fpath = self.calibration_dir / f"{self.id}.json"

# src/lerobot/robots/so_follower/so_follower.py:44
name = "so_follower"
```

**两个反直觉的点：**

| 事实 | 含义 |
| --- | --- |
| **标定目录名由 `robot.name`（类属性）决定，跟 `--robot.type` 无关** | `so100_follower` 和 `so101_follower` 在 `config_so_follower.py:56-57` 注册到**同一个 `SOFollower` 类**，写哪个都进 `robots/so_follower/` 目录 |
| **真正决定成败的只有 `--robot.id`** | 漏了它就变成 `None.json`，永远找不到 → **每次启动都被迫重新标定** |

> 顺带解释日志里那句 `Connecting robot (so100_follower)`：`RobotConfig.type` 是**只读属性**（`configs/robots/config.py:38-40` 的 `@property def type`），从注册表反查类名，而 `so100_follower` 先注册 → **恒回显 `so100_follower`**。纯回显假象，不影响功能。
>
> 另外，旧版代码可能在 `robots/so101_follower/` 下留过标定文件，**现在的代码根本不会去读那个目录**，别被它误导。

#### 启动时的那句提示：看到它 = 标定文件已经找到了

```
Press ENTER to use provided calibration file associated with the id zihao_follower_arm,
or type 'c' and press ENTER to run calibration:
```

**看到它就直接按回车。** 如果没出现这句、而是直接开始 `Running calibration of ...`，说明还有别的问题。

> ⚠️ **千万别按 `c`。** 按了会立刻用当前实测值**覆盖**掉这份标定。你的策略是在这套标定下采的数据、训出来的模型，标定一换关节角映射就变了，机械臂会乱动。**建议现在就把标定文件备份一份**（零成本）：
>
> ```cmd
> copy "%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\zihao_follower_arm.json" "%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\zihao_follower_arm.json.bak"
> ```

#### 怎么判断一份标定文件是不是"真的有效"

打开那个 JSON，看 6 个关节的 `range_min` / `range_max` 跨度。**有效标定的跨度都在 1000 以上**：

| 关节 | homing_offset | range_min | range_max | 跨度 |
| --- | --- | --- | --- | --- |
| shoulder_pan | 1967 | 820 | 3458 | 2638 |
| shoulder_lift | -1223 | 783 | 3175 | 2392 |
| elbow_flex | -1885 | 800 | 3012 | 2212 |
| wrist_flex | 2040 | 828 | 3107 | 2279 |
| wrist_roll | -1093 | 0 | 4095 | 4095（不转，满量程正常） |
| gripper | 1346 | 1538 | 2781 | 1243 |

如果某个关节 `range_min ≈ range_max`，说明**这个关节没标上**，必须重来。

#### ⚠️ 绝对不要在 `lerobot-rollout` 里现场标定

`lerobot-rollout` 发现标定缺失会**自动进入交互式标定**，这很危险：一旦存下"行程为 0"的标定文件，推理就会以错误的关节角映射驱动机械臂，**可能直接把手臂开到限位撞死**。正确顺序是先单独标定：

```cmd
:: ① 找串口（拔插 USB 确认）
lerobot-find-port

:: ② 单独标定（id 自己取名，务必记牢）
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm

:: ③ 确认相机编号
python -c "import cv2;c=[(i,cv2.VideoCapture(i)) for i in range(4)];print([(i,x.isOpened()) for i,x in c]);[x.release() for _,x in c]"

:: ④ 推理（--robot.id 必须和 ② 里完全一致）
lerobot-rollout --strategy.type=base --policy.path="D:\Desktop\pretrained_model" --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=10
```

**标定的校验收据就是那张表：任一关节 `MIN ≈ MAX`，就说明这个关节没标上，必须重来。**

#### 标定表里所有关节 `MIN/POS/MAX` 全是 2047 是什么意思

2047（≈2048）是 Feetech STS3215 **12 位编码器**（0~4095）的正中值，也是 SO-ARM 组装说明书让你"所有关节摆到行程中点"时的读数。`MIN = POS = MAX` 只有两种可能：

| 可能 | 特征 |
| --- | --- |
| (A) 关节**没被转动过** | 通常只出现在个别关节（`gripper` 最常漏） |
| (B) **根本没读通舵机** | **所有**关节同一固定值，转动关节时 `POS` 纹丝不动 |

**排查顺序**：① 舵机驱动板没接 12V 电源（**最高概率，USB 只通信不供电**）→ ② 串口不对（跑 `lerobot-find-port`）→ ③ 舵机 ID 未设置/冲突（出厂默认全是 1，要用飞特工具逐个改成 1~6）→ ④ 3-pin 线松动或插反。

#### `--display_data=true` 会"吃掉"你的回车

Rerun 窗口会抢键盘焦点，导致你按的回车进不到 cmd，但 Ctrl+C 却会直接打给 Python（表现为 `Caught Ctrl-C, quitting Rerun Viewer…` 然后程序退出）。**调试标定阶段先去掉 `--display_data=true`**，或先点回 cmd 窗口再按键。

### 7.2 推理命令

**Windows 本机推理（cmd 一行版，复制即用）**：

```cmd
lerobot-rollout --strategy.type=base --policy.path="D:\Desktop\pretrained_model" --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=10
```

**Linux 版本**：

```bash
lerobot-rollout --strategy.type=base --policy.path=/home/你的用户名/pretrained_model --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=10
```

#### `lerobot-rollout` 常用参数

| 参数 | 作用 | 默认 |
| --- | --- | --- |
| `--policy.path` | 策略路径（本地目录或 HF 仓库 ID） | 必填 |
| `--robot.type` / `--robot.port` / `--robot.id` | 机器人型号 / 串口 / **标定 ID（必填）** | 必填 |
| `--robot.cameras` | 相机配置 JSON | 必填 |
| `--device` | 推理设备 `cpu`/`cuda`/`mps` | `auto` |
| `--fps` | 控制循环频率 | `30`（与数据集一致） |
| `--duration` | 运行秒数，0 = 无限 | `0` |
| `--task` | 任务描述 | — |
| `--display_data` | 把遥测流到 Rerun 可视化 | `false` |
| `--interactive` | 交互模式（见 7.5） | `false` |
| `--play_sounds` | 事件语音提示 | `true` |

> 两个 Windows 上的小坑：① `--display_data=true` 需要装了 `viz` extra（rerun-sdk），用 `pip install -e ".[core_scripts,feetech]"` 一次装齐；② 启动时报语音/TTS 相关错误就加 `--play_sounds=false`。

#### ⚠️ 路径与串口必须和运行平台匹配 —— 最常见的翻车点

`/dev/ttyACM0` 是 **Linux** 设备名，`D:\Desktop\...` 是 **Windows** 路径，两者不可能同时成立。

| 运行平台 | `--policy.path` | `--robot.port` | 备注 |
| --- | --- | --- | --- |
| Linux（推荐，机械臂直连） | `/home/你的用户名/pretrained_model` | `/dev/ttyACM0` | 拔插/重启后可能变 `ACM1`，用 `ls /dev/ttyACM*` 确认 |
| WSL2 | `/mnt/d/Desktop/pretrained_model` | `/dev/ttyACM0` | Windows 的 `D:\` 在 WSL 里是 `/mnt/d/`；串口需用 usbipd 把 USB 转发进 WSL |
| Windows 原生 | `D:/Desktop/pretrained_model` | `COM7` 之类 | 端口在「设备管理器 → 端口(COM 和 LPT)」看；路径用正斜杠 |

**另外三点**：

- **写成一行时，把行尾的 `\` 全部删掉** —— 反斜杠只在多行写法里表示"续行"，留在单行命令里会被当成参数。
- 运行前先 `ls` 一下 `--policy.path` 指向的目录，必须能看到 `model.safetensors` 和 `config.json`。
- 用 `lerobot-find-cameras opencv` 确认 `index_or_path` 哪个是前视、哪个是腕部 —— **front 和 wrist 接反了策略直接失效**。

相机 key 写 `front` / `wrist`，LeRobot 会自动映射成 `observation.images.front` / `observation.images.wrist`，正好和数据集对上。`--task` 要写数据集里的任务文本 **`Grab Box`**（可以用 `python -c "import pyarrow.parquet as pq;print(pq.read_table('meta/tasks.parquet').to_pydict())"` 随时确认；只有一个任务时 ACT 可以省略，但 π0.5 这类 VLA **必须写**）。

### 7.3 `--duration` 和 `--fps` 是什么意思

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--duration` | **这次运行总共跑多少秒**，到点自动停止、断开机械臂、退出 | `0` = **无限**，跑到手动 Ctrl+C |
| `--fps` | 目标**控制循环频率**（每秒下发多少动作指令） | `30`（要和数据集一致） |

两者关系：`--duration=60` × `--fps=30` ≈ **1800 个控制步**。

因为 ACT 是 **action chunking**（一次预测 100 步动作、然后连续执行这 100 步），这 60 秒里策略网络**实际只推理约 18 次**，其余时间都在回放已算好的动作块。所以：

- **`--fps` 不是"策略推理频率"**，而是"动作下发频率"。调高它不会让模型更聪明，只会让动作回放更平滑。
- **CPU 推理会拖慢循环**：单次推理耗时超过 `100 / fps` 秒时，实际频率达不到 30 Hz，机械臂会走走停停（这就是修好 CUDA 的价值）。
- **`--duration` 是安全阀**：第一次调试机械臂**务必**用 `--duration=10`。设成 `0` 的话机械臂会一直动下去。

> **别被节奏报告里的 `infer 553 calls` 迷惑。** `infer` 计时区段**每个 tick 都会调用一次**，但它内部走 `ACT.select_action()`：动作队列非空时只是 `popleft()` 弹出一个已算好的动作，**只有队列空了才真正跑网络**（源码 `policies/act/modeling_act.py:119-125`）。所以"553 次调用"≠"553 次推理"，真正的重规划是 `553 / 100 ≈ 6` 次。

**分阶段调试建议**：

1. `--duration=10`，**不加** `--display_data`：确认动作方向对、不抖、不撞。
2. `--duration=60`：完整跑一整轮任务，看能不能真的抓起来。
3. 稳定后再 `--duration=0`，并把 `--display_data=true` 加回来看可视化。
4. 要做**多轮反复评测**时加 `--interactive=true`，用 `/start` ↔ `/reset` 切轮次（见 7.5）。

### 7.4 读懂 rollout 的节奏报告：`Control loop is running slower` 往往是误报

真机跑起来后会看到这种刷屏警告：

```
WARNING le_timer.py:407 Control loop is running slower (4.1 Hz) than the target FPS (30 Hz).
        Robot control might be unstable. Common causes are: 1) Camera FPS not keeping up
        2) Policy inference (action or text) taking too long 3) CPU starvation
```

**先别慌 —— 也不要照着它给的 4.1 Hz 去判断性能。** 关键看运行结束时的那份报告（实测样例）：

```
Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick): 1698 ticks, 1697 judged
  effective cadence: 28.29 Hz over 60.0 s measured          ← ★ 真正的成绩
  ticks over the 33.3 ms work budget: 16/1697 (0.9%)
  work mean 9.1 ms, worst 277.1 ms
  loop-body steps (share of measured work):
    observe      mean   1.38 ms · worst   2.87 ms ·  14.7% of work · 1698 calls
    infer        mean   7.36 ms · worst 464.58 ms ·  78.5% of work · 1698 calls
    send         mean   0.44 ms · worst   1.00 ms ·   4.7% of work · 1698 calls
  pacing headroom: 25.9 ms slept per tick on average — near zero means the loop is saturated
```

#### 为什么警告里的 Hz 不等于真实频率

源码 `src/lerobot/utils/cycle_timer.py:399-411`：

```python
if group_work > self.cycle_interval:
    stats.groups_over += 1
    logger.warning(f"Control loop is running slower ({1 / group_work:.1f} Hz) than the target FPS ...")
```

- 判据是 **`group_work > cycle_interval`**，即"**这一个 tick 的耗时**超过了 33.3 ms 预算"。
- 显示的 Hz 是 **`1 / group_work`** —— "**这一个 tick 工作耗时的倒数**"，**不是**循环的平均频率。

所以一条 `4.1 Hz` 的真实含义是：**"有个 tick 干活花了约 244 ms"**（1/0.244 = 4.1）。它跟"循环跑在 4.1 Hz"完全不是一回事。

> 源码注释还特别说明：**第一组会被跳过**（`if self._groups_closed > 1`），因为启动时 CUDA 上下文初始化、cuDNN 自动调优、相机预热都落在这里，算进去会"每次健康启动都报警"。

#### 那 ~200 ms 的尖刺从哪来：不是噪声，是"重规划 tick"

先看一组对得上的数字：

| 量 | 60 s 实测值 |
| --- | --- |
| 总 tick | 1698 |
| ACT 重规划一次要走的步数 | `n_action_steps = 100` |
| 因此真正的网络前向次数 | `1698 / 100 ≈ 17` 次 |
| 报告里"超过 33.3 ms 预算"的 tick | **16 个（0.9%）** |

**"超预算的 tick 数"几乎等于"重规划次数"** —— 这不是巧合。因为 `interpolation_multiplier` 默认为 1 时，`needs_new_action()` **每个 tick 都返回 True**（源码 `strategies/core.py:113-115` 注释明确），于是 `engine.get_action()` 每 tick 都被调用；但 `ACT.select_action()` 只在队列空时才真跑前向：

```python
# src/lerobot/policies/act/modeling_act.py:119-125
if len(self._action_queue) == 0:
    actions = self.model(...)          # ← 只有这里才真的跑网络
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()    # ← 其余 99 个 tick 走这里（微秒级）
```

所以：**每 3.33 秒就有 1 个 tick 要扛一次完整前向（200~280 ms），这一个 tick 必然超预算并触发警告。** 60 秒 ÷ 3.33 秒 ≈ 18 次，与实测的 16~17 次吻合。

> 验算：`1697` 个被判定的 tick × 平均 `9.1 ms` ≈ 15.4 s 总工作量；其中约 17 次重规划按 ~250 ms 计 ≈ 4.25 s（占 27%），剩下 1680 个 tick 平均 6.6 ms —— 数字闭合。
>
> **`infer worst 464.58 ms`** 是**首次前向**，包含 CUDA 上下文初始化与 cuDNN 算法自动调优，一次性开销，之后不会再现。

**结论：这些警告是 action chunking 的固有节拍，属正常现象，不用治。** 真正的判据永远是 `effective cadence` 和 `pacing headroom`。

#### 怎么判读这份报告

| 字段 | 含义 | 健康标准 |
| --- | --- | --- |
| `effective cadence` | 整段运行的实际循环频率 | **≥ 目标 FPS 的 90%** 就算健康 |
| `ticks over the ... budget` | 超预算的 tick 占比 | **< 5%** 可接受；> 20% 才是真慢 |
| `work mean` | 平均每个 tick 的实际计算耗时 | 应显著小于 33.3 ms |
| `infer mean` / `observe mean` | 各步骤耗时拆解 | 找出瓶颈在哪一环 |
| `pacing headroom` | 每 tick 平均睡了多久 | **越接近 0 越危险** |

想让尖刺更小更稳（尤其当 `effective cadence` 掉到目标 85% 以下时）：

1. **插上电源适配器**，Windows 电源计划切"高性能"（`nvidia-smi` 显示 `Perf: P5` 说明显卡在省电档）。
2. NVIDIA 控制面板 → 管理 3D 设置 → **电源管理模式 = 优先最高性能**。
3. **关掉占显存的程序**（浏览器硬件加速、游戏、录屏）。
4. 两台相机 + 舵机总线尽量别插在同一个 USB 控制器上。
5. 想进一步压平，可试 `--inference.type=rtc`（实时分块推理），把重规划摊到多个 tick 上。

### 7.5 ⚠️ 跑评测时，什么时候才能重置环境？

**结论：不能边跑边重置。必须等它停下来。** 哪怕机械臂此刻"看起来离箱子很远"。

#### 为什么不能：三个独立理由，任何一个都足够

1. **物理安全。** 机械臂正在自主运动，它不会因为你把手伸进去就让开。ACT 输出的动作是**学出来的轨迹**，不是带避障的安全规划。
2. **动作块是开环执行的。** ACT 一次预测 100 步动作，然后**连续盲走 100 步** —— `100 / 30 fps ≈ 3.33 秒`。这 3.33 秒里它**完全不看画面**。
3. **观测错位。** 就算等到下一个重规划 tick，它看到的画面也已经和"算出这批动作时"不一样了 —— 但**已排进队列的那部分动作仍会被执行完**。结果是一段最长 3.33 秒的"照着旧场景做动作"的迷惑行为。

```
t=0.00s  重规划：看一眼画面 → 算出 100 步动作（本机耗时 ~200-560 ms）
t=0.20s  开始盲走 ─────────────────────────────────────┐
           ↑ 你在这里把箱子挪了 5 cm                      │ 这 3.33 秒
           │   策略完全不知道你在干什么                    │ 手臂的动作
t=3.33s  走完第 100 步、队列空了 → 这才重新看一眼画面 ────┘ 早就全部定死了
```

#### 唯一安全的两个时机

| 时机 | 怎么做 | 说明 |
| --- | --- | --- |
| **① 等 `--duration` 自然到点** | 等到 `Duration limit reached (60s)` → `Returning robot to initial position before shutdown` → `zihao_follower_arm SOFollower disconnected` | 默认 `return_to_initial_position=True`，它会**先自己归位、再断开**。看到 `disconnected` 才算绝对安全 |
| **② 用 `--interactive=true` 的 `/reset`** | 见下 | **推荐**，不必反复重启程序 |

> ⚠️ **一定要等手臂真的回到初始位、并且断开连接之后，再伸手。** "归位"这个动作本身也是机械臂在动。

#### 推荐做法：`--interactive=true`

硬件和策略**保持热连接**，用 stdin 打字下命令，机械臂在 `/start` 之前**完全不动**：

```cmd
lerobot-rollout --strategy.type=base --policy.path="D:\Desktop\pretrained_model" --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=60 --interactive=true
```

| 指令 | 作用 |
| --- | --- |
| `/start` | 开始（或重启）策略控制循环 |
| `/reset` | **停下运动 → 把手臂送回初始位 → 恢复任务指令** ← 这就是"重置环境"的那一刻 |
| `/subtask <文本>` | 换任务指令 |
| `/stop` | 结束会话、正常断开 |
| `/vqa <问题>` | 问策略看到了什么（仅带文本头的策略可用） |
| `/help` | 列出全部指令 |

**两个关键细节：**

- `--strategy.type=base` 支持交互模式（源码 `supports_interactive = True`）；`episodic`、`sentry` 也支持。
- **交互模式下 `--duration` 限制的是"每一段 `/start`"，不是整个会话**（源码 `rollout/configs.py:248-250`）。

#### 标准评测循环

```
输入 /start
  ↓  手臂自主跑满 60 秒（期间不要碰任何东西）
  ↓  Duration limit reached (60s) → 控制循环结束、待命
输入 /reset       ← 等它打印 "Robot reset — holding at initial position. /start to run." 再动手
  ↓  这时把箱子摆回起始位置
输入 /start       ← 第二轮
  ↓  ...
输入 /stop        ← 收工
```

| 屏幕提示 | 含义 |
| --- | --- |
| `Resetting — returning the robot to its initial position...` | 还在动，**再等** |
| `Robot reset — holding at initial position. /start to run.` | ✅ **现在才能伸手重置场景** |
| `Robot paused — no initial position captured, holding current pose.` | ⚠️ 启动时没抓到初始位，`/reset` **不会归位** |
| `No initial position captured — skipping the return move` | ⚠️ 同上（来自 `controller.py:_reset_robot`） |

#### 绝对不要做的三件事

1. **不要拿 Ctrl+C 当"重置键"。** 它走中断路径（会连带关掉 Rerun）。要停就用 `/stop`。
2. **不要在手臂运动时用手去"扶"或"挡"。** Feetech 舵机扭矩不小，硬顶会打齿。
3. **不要靠"它现在离箱子远"来判断安全。**

#### 紧急情况：怎么立刻让它停

最可靠的物理急停是**断开舵机驱动板的 DC 电源**（不是拔 USB —— USB 只负责通信，拔了手臂照样保持着力矩）。

> ⚠️ **断电瞬间电机会失去保持力矩，手臂会在重力下直接塌下来。** 断电前要么下面垫好软物，要么先用另一只手托住。

---

## 8. 常见报错速查表

| 报错 / 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `FileNotFoundError: .../meta/info.json` | `--dataset.root` 指错层级 | root 必须指向**直接包含 `meta/`** 的目录 |
| `ffmpeg` not found / 视频读不出来 | 没装 ffmpeg | `conda install ffmpeg=7.1.1 -c conda-forge -y` |
| AV1 / torchcodec 解码报错 | 视频解码后端 | 训练命令加 `--dataset.video_backend=pyav` |
| `CUDA out of memory` | batch 太大 | `--batch_size=4`，或减 `--num_workers` |
| 下载 ResNet18 预训练权重超时 | 网络到不了 HF | `export HF_ENDPOINT=https://hf-mirror.com` |
| `codebase_version v3.0 不支持` | LeRobot 版本太旧 | 用 main 分支源码安装 |
| wandb 一直连不上 | 网络 | 改成 `--wandb.enable=false` |
| 训练很慢、GPU 利用率长期 <80% 或锯齿状 | 可能是数据加载瓶颈（但**先量再调**） | 看 `data_s` vs `updt_s`：比值 <0.2 就**不是**这个原因（实测 0.004/0.080 = 5%） |
| 想加 `num_workers` 提速 | 超卖 CPU 核数只会更慢 | 按 CPU 核数 ×0.75 给（3090 只给 4~5），并确认 `persistent_workers=true` |
| 关机后数据集/代码没了 | 存到了非持久化目录 | 所有文件放 `/home/featurize/work` 下 |
| 右键文件夹没有「下载」项 | 工作区只对**文件**提供 Download | 打包成 `.tar.gz` 再下；或用 VSCode 远程（见 §5） |
| 点击 `checkpoints` 后进不去子文件夹 | ① JupyterLab **单击只选中，要双击**；② 文件树没刷新；③ `--output_dir` 是相对路径 | 先 `find /home/featurize -maxdepth 8 -type d -name "act_datass"` 定位，再**双击**进入 |
| 训练跑完但怀疑"没训好" | 看日志末尾是否有 `End of training` + `Model weights successfully saved to ...` | 两条都出现就是成功了；再 `find ... -name model.safetensors` 复核落盘 |
| `The fields 'dtype' are not valid for ACTConfig` | **本地 lerobot 版本旧于训练版本** | ① 对齐版本（`git pull` / 源码重装）；② 应急：脚本删掉 config.json 里被点名的字段（见 §6.2） |
| 真机推理时机械臂乱飞 / 动作完全不对 | ① 相机 front/wrist 接反；② 缺归一化统计量；③ **用错了标定文件**；④ 版本不一致 | 核对相机索引；确认 `pretrained_model` 7 个文件都在；核对两边版本；确认 `--robot.id` 和采数据时一致 |
| 日志出现 `Running calibration of None SOFollower` | 没传 `--robot.id`，标定文件名变成 `None.json` | 补上 `--robot.id=<采数据时同名>`（见 §7.1） |
| 标定表格里**所有**关节 `MIN/POS/MAX` 全是 `2047` | ① 舵机驱动板没接 12V 电源；② 串口不对；③ 舵机 ID 未设置/冲突；④ 根本没转动 | 先接 12V 电源 + `lerobot-find-port`；标定时转动关节看 `POS` 有没有跟着变 |
| 标定表里**个别**关节 `MIN ≈ MAX` | 该关节没被转满行程（`gripper` 最常被漏） | 重新标定，逐个关节**缓慢**推到两端 |
| `Device 'cuda' is not available. Switching to 'cpu'` | 本机没 N 卡，或装了 CPU 版 torch | `nvidia-smi` + `torch.cuda.is_available()` 判断；重装 CUDA 版 torch（见 §6.3） |
| 刷屏 `Control loop is running slower (4.1 Hz) ...` | **多半是误报** —— 显示的是"某个 tick 工作耗时的倒数"；真凶是每 3.33 秒一次的**重规划 tick** | 只看结尾 `Cadence summary`：`effective cadence` ≥ 目标 90% 且超预算 tick <5% 就健康（见 §7.4） |
| **想在 rollout 运行中重置环境 / 挪箱子 / 扶机械臂** | ① 动作块**开环盲走 100 步 ≈ 3.33 秒**；② 人身与机械危险 | **绝对不要在跑的过程中动手。** 等 `Duration limit reached` + `disconnected`，或用 `--interactive=true` + `/reset`（见 §7.5） |
| `/reset` 后手臂没回初始位，打印 `no initial position captured` | 启动时没记录下初始位 | 重启程序，让它在手臂处于标准起始位时抓一次 |
| 交互模式下 `--duration` 好像不起作用 | 交互模式下 `--duration` 限制的是**每一段 `/start`** | 这是设计行为；想一直跑就设 `--duration=0` |
| `Connecting robot (so100_follower)`，但我写的是 `so101_follower` | `type` 是只读属性，从注册表反查类名，恒回显先注册的别名 | 无需处理，两个名字指向同一个类 |
| 交互式标定刚要按回车，程序就退出了（`Caught Ctrl-C, quitting Rerun Viewer`） | Rerun 窗口抢走键盘焦点 | 调试阶段先去掉 `--display_data=true`，或先点回 cmd 窗口再按键 |
| 标定时报 `Magnitude xxxxx exceeds 2047 (max for sign_bit_index=11)` | 舵机当前位置/零点异常（Feetech 用 1 位符号 + 11 位幅值编码） | 用飞特调试工具做一次**中位标定**（把当前位置写成 2048），再重新整臂标定 |

---

## 9. 省钱与踩坑清单

1. **开机才计费，关机/归还即停止计费。** 不用的时候一定关掉，别挂着。
2. 训练前先 `du -sh ~/work` 看同步盘用量，超过 30 GB 免费额度按 ¥0.001/GB/小时计费。
3. **不要在 `/tmp` 里存任何要保留的东西。**
4. 数据集放云端「数据集」里一份（永久保存），`~/work` 里一份（训练用），双保险。
5. **每次换实例都要重新 `conda activate lerobot`** —— conda 环境装在家目录，归还实例就没了（`~/work` 里的文件会保留）。
6. 想进一步加速：把数据集拷一份到实例本地盘再训，训完把模型拷回 `~/work`。
7. **首次跑建议先 `--steps=1000` 验证全流程**，再开长训。
8. **权重只要 `pretrained_model/` 那 7 个文件**，别把优化器状态一起下（那部分比模型还大）。

---

## 10. 本次实验结果

> 硬件：Featurize RTX 4090（24 GB）训练 / 本地 RTX 3050 Laptop（4 GB）推理
> 数据：`datass`，40 条示教，17,961 帧，2 路 480×640
> 策略：ACT，`chunk_size=100`，`kl_weight=10.0`，`optimizer_lr=1e-5`

### 10.1 训练阶段（数值来自真实训练日志）

| 指标 | 实测值 | 解读 |
| --- | --- | --- |
| 完成步数 | **50000 / 50000** | 日志末尾 `End of training` + `Model weights successfully saved to ...` |
| 训练耗时 | **74.3 分钟** | RTX 4090 |
| 吞吐 | **11.21 step/s** | 与 `smp/s` 字段一致 |
| 训练花费 | **≈ ¥2.3** | 74.3 min = 1.238 h × ¥1.87/h |
| 显存占用 | **3.73 GB** | 24 GB 卡余量极大，batch 还能加 |
| `data_s` / `updt_s` | 0.004 / 0.080 s = **5%** | **完全不是**数据加载瓶颈 |
| GPU 忙碌占比 | 0.080 / 0.085 = **94%** | GPU 没闲着 |
| 梯度范数 `grdn` | **202 → 65** | 快速回落并稳定（`grad_clip_norm=10`，早期一直被裁剪属正常） |
| `l1_loss` | **0.712 → 0.322**（1200 步内） | 减半，动作预测在稳定变准 ✅ |
| `kld_loss` | 0.869 → 0.139 | 单任务数据下趋近 0，正常 |
| 数据覆盖 | 400K 样本 ÷ 17961 帧 ≈ **22 轮** | 对 40 条数据足够 |
| 权重体积 | **206.7 MB**（fp32）→ **≈ 51.67 M 参数** | 与配置文件里的结构（ResNet18 + 4 层编码器 + 1 层解码器，`dim_model=512`）吻合 |

**训练结论：一次成功。** 5 万步、74 分钟、¥2.3，得到一份可部署的 ACT 权重。

### 10.2 推理阶段（两次真机 rollout 对比）

| 指标 | 20 秒冒烟测试 | **60 秒正式测试** | 说明 |
| --- | --- | --- | --- |
| 总 tick | 553 | **1698** | ≈ 28.3 tick/s |
| `effective cadence` | 27.62 Hz | **28.29 Hz** | 目标 30 Hz 的 **94.3%** ✅ |
| 超预算 tick | 5 / 552（0.9%） | **16 / 1697（0.9%）** | 占比完全一致，说明是**结构性节拍**而非随机抖动 |
| `work mean` | 9.2 ms | **9.1 ms** | 远小于 33.3 ms 预算 ✅ |
| `work worst` | 265.9 ms | **277.1 ms** | 重规划 tick |
| `infer mean` / `worst` | 8.23 / 563.32 ms | **7.36 / 464.58 ms** | 首次前向的一次性开销 |
| `observe mean` | 1.38 ms | **1.38 ms** | 两路相机取帧，极稳定 |
| `send mean` | 0.44 ms | **0.44 ms** | 舵机总线写入，极稳定 |
| `pacing headroom` | 25.9 ms | **25.9 ms** | 每 tick 还能睡 25.9 ms，**没有饱和** ✅ |
| 重规划次数（理论） | 5.53 | **16.98** | 与实测超预算 tick 数（5 / 16）**精确吻合** |
| 掉出循环体的 tick | — | **1 个** | 60 秒里仅 1 次 sleep 过冲 |

### 10.3 结果判读

**结论：整条链路第一次全部打通，且性能健康。** 逐项核对：

| 日志行 | 含义 | 判定 |
| --- | --- | --- |
| `Policy loaded: type=act, device=cuda` | 模型加载成功，CUDA 生效 | ✅ |
| `zihao_follower_arm SOFollower connected` | **标定文件命中**，`--robot.id` 生效 | ✅ |
| `OpenCVCamera(2) / OpenCVCamera(1) connected` | 两路相机都连上 | ✅ |
| `Captured initial robot position (6 keys)` | 读到 6 个关节初始位置 | ✅ |
| `SyncInferenceEngine initialized (device=cuda, action_keys=6)` | 推理引擎就绪，输出 6 维动作 | ✅ |
| `Duration limit reached (60s)` → `Rollout finished` | 跑满 60 秒，正常收尾 | ✅ |
| `Returning robot to initial position before shutdown` | 断开前先把手臂归位 | ✅ 好设计 |

**三条判读要点：**

1. **那 17 条 `4.1 Hz` / `6.4 Hz` 警告是误报。** 它们显示的只是"某个 tick 耗时的倒数"；60 秒里 1698 个 tick 只有 16 个超预算（0.9%），且每 tick 还能睡 25.9 ms。真正的成绩是 **28.29 Hz**。
2. **两组数据高度可复现。** 20 秒与 60 秒两次运行的超预算占比都是 **0.9%**，`observe` / `send` / `pacing headroom` 三项的均值几乎逐位相同 —— 说明这些数字反映的是**模型本身的固有节拍**（每 3.33 秒一次重规划），不是环境噪声。
3. **性能瓶颈不在数值计算，也已经不在硬件。** `observe` + `send` 合计只占 ~19%，`infer` 那 78.5% 里绝大部分是每 tick 的观测预处理，真正的网络前向只在 17 个 tick 上发生。**4 GB 的 3050 都能跑到目标频率的 94%。**

### 10.4 尚未闭合的环节（诚实说明）

日志能证明**软件链路全通**，但证明不了**策略学会了任务**。以下三项需要肉眼观察并记录，是真正的验收标准：

| 待确认项 | 判断标准 |
| --- | --- |
| 这 60 秒里手臂是否朝箱子方向运动 | 方向对 = 相机没接反、标定没错位 |
| 夹爪是否闭合 | 有闭合动作 = 策略学到了"抓"这个动作模块 |
| 动作是一顿一顿还是基本连续 | 28.29 Hz 应有轻微顿挫但整体连贯 |
| 是否撞限位 / 朝奇怪方向冲 | 若有则需回查相机与标定 |

**建议的下一步**：用 `--interactive=true`，把 `--duration` 设为 0 或 60，连续跑 **10 ~ 20 轮**，记录成功率。这才是"ACT 训练是否成功"的最终答案 —— **loss 低不等于真机成功率高**。

---

## 11. 一页速查（复制即用）

```bash
# ---------- 环境 ----------
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge -y
cd ~/work && git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[training]"
pip install wandb --upgrade
export HF_ENDPOINT=https://hf-mirror.com
hf auth login
wandb login

# ---------- 数据 ----------
mkdir -p ~/work/datasets && cd ~/work
featurize dataset download <你的数据集UUID>
unzip datass.zip -d ~/work/datasets/

# ---------- 训练 ----------
cd ~/work/lerobot
lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.type=act \
  --output_dir=outputs/train/act_datass \
  --job_name=act_datass \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=10      # 按机型 CPU 核数×0.75：3090→4~5，4090→8~12，PRO6000→12~16

# ---------- 续训 ----------
lerobot-train --resume=true \
  --config_path=outputs/train/act_datass/checkpoints/last/pretrained_model/train_config.json
```

**真机推理（Windows cmd，一行版）**：

```cmd
:: 1) 找串口（拔插 USB 确认）
lerobot-find-port

:: 2) 标定（只做一次；id 必须和采数据时同一个，推理时也要带上）
lerobot-calibrate --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm

:: 3) 确认相机编号
python -c "import cv2;c=[(i,cv2.VideoCapture(i)) for i in range(4)];print([(i,x.isOpened()) for i,x in c]);[x.release() for _,x in c]"

:: 4) 推理（第一次务必 --duration=10 试跑，且先不要 --display_data=true）
::    ⚠️ 跑起来后【绝对不要】伸手去挪箱子/扶机械臂：动作块是开环盲走 100 步（≈3.33 秒），
::       策略看不到你的改动；要重置环境必须等它自己停下并断开。
lerobot-rollout --strategy.type=base --policy.path="D:\Desktop\pretrained_model" --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=10

:: 4b) 交互式评测（推荐多轮时用）：/start 跑一轮 → /reset 归位（这时才去重置场景）→ /start 再来 → /stop 收工
lerobot-rollout --strategy.type=base --policy.path="D:\Desktop\pretrained_model" --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --fps=30 --duration=60 --interactive=true

:: 5) 本机 torch 是否支持 CUDA
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

**ACT 关键超参（本次训练实际使用，来自 checkpoint 的 `config.json`）**：

| 参数 | 值 |
| --- | --- |
| `chunk_size` / `n_action_steps` | 100 / 100 |
| `vision_backbone` | `resnet18`（ImageNet 预训练） |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 |
| `latent_dim` / `n_vae_encoder_layers` | 32 / 4 |
| `kl_weight` | 10.0 |
| `optimizer_lr` / `optimizer_lr_backbone` | 1e-5 / 1e-5 |
| `optimizer_weight_decay` | 1e-4 |
| `normalization_mapping` | VISUAL / STATE / ACTION 全部 `MEAN_STD` |
| 参数量 | ≈ **51.67 M**（权重 206.7 MB fp32） |
