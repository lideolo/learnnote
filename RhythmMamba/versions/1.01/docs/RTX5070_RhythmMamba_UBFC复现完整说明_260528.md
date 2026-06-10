# RhythmMamba UBFC-rPPG 复现完整说明

## 一、实验环境

### 硬件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 (12 GB VRAM) |
| 计算能力 (Compute Capability) | 12.0 (Blackwell, sm_120) |
| NVIDIA 驱动 | 596.49 (支持 CUDA 13.2) |
| 系统内存 | 15 GB |
| OS | WSL2 (Ubuntu 22.04, glibc 2.35) |

### 软件

| 软件包 | 版本 | 安装来源 |
|--------|------|----------|
| Python | 3.11 | conda (conda-forge) |
| PyTorch | 2.12.0+cu130 | pip (PyTorch 官方) |
| CUDA Toolkit (nvcc) | 13.0 | conda (nvidia/label/cuda-13.0.2) |
| mamba-ssm | 2.3.2.post1 | 源码编译 (sm_120) |
| causal-conv1d | 1.6.2.post1 | 源码编译 (sm_120) |
| OpenCV | 4.13.0 | conda-forge |
| NumPy | 2.4.6 | pip |
| SciPy | 1.17.1 | pip |
| Pandas | 3.0.3 | pip |
| tqdm | 4.67.3 | pip |
| yacs | 0.1.8 | pip |
| ffmpeg | 8.0.1 | conda-forge |

> 完整依赖清单见仓库中的 `requirements_rtx5070.txt` (pip) 和 `conda_env_rtx5070.txt` (conda)。

---

## 二、代码修改详情

### 修改 1: `main.py` — UBFC-rPPG 数据集名称兼容

**修改位置**: 4 处数据集名称判断（第 134、178、224、271 行）

**改前**:
```python
elif config.TRAIN.DATA.DATASET == "UBFC":
```

**改后**:
```python
elif config.TRAIN.DATA.DATASET in ["UBFC", "UBFC-rPPG"]:
```

**原因**: 配置文件中写的是 `DATASET: UBFC-rPPG`，但原代码只判断 `== "UBFC"`，导致 `Unsupported dataset!` 错误。

**影响范围**: train_loader、valid_loader、test_loader、unsupervised_loader 四个 DataLoader 创建分支。

**缺陷**: 无。这是纯粹的字符串匹配扩展，不影响已有逻辑。

---

### 修改 2: `main.py` — num_workers 调整（防止 OOM）

**修改位置**: 第 166、211、258、295 行（4 处）

**改前**:
```python
num_workers=16,
```

**改后**:
```python
num_workers=4,
```

**原因**: 每个预处理后的 npy 文件约 61 MB。使用 16 个 DataLoader worker + prefetch_factor=2 时，系统会预加载约 512 个文件 (~31 GB)，远超 15 GB 系统内存，触发 Linux OOM Killer 杀掉 worker 进程。

**影响**: 数据加载速度略有下降，但训练稳定性得到保证。

**缺陷**: 4 个 worker 对 15 GB 内存仍偏保守，如果内存更大可以适当提高。GPU 利用率约 16-27%，说明数据加载仍有优化空间。

---

### 修改 3: `main.py` — batch_size 配置化

**说明**: 原代码 batch_size 从 yaml 配置读取 (`config.TRAIN.BATCH_SIZE`)，本身无需改动。但在实验过程中我们从 4 调到 16 再调到 8，最终确定 batch_size=8 是 RTX 5070 12 GB 显存的最佳值：
- batch_size=16: GPU 显存 11.8/12 GB，几乎爆满，利用率仅 1%
- batch_size=8: GPU 显存 7.4/12 GB，正常训练
- batch_size=4: GPU 显存过低，利用率不足

**最终配置**: `BATCH_SIZE: 8`（在 yaml 文件中设置）

---

### 修改 4: `dataset/data_loader/BaseLoader.py` — 串行预处理模式 + spawn 启动

**修改位置**: 第 16-20 行（import 区域），第 446-458 行（multi_process_manager 方法）

**新增代码 1** — spawn 启动方式:
```python
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
```

**新增代码 2** — 串行预处理模式:
```python
if os.environ.get('RHYTHMMAMBA_SERIAL_PREPROCESS', '0') == '1':
    print('Using serial preprocessing mode')
    file_list_dict = {}
    for i in choose_range:
        self.preprocess_dataset_subprocess(data_dirs, config_preprocess, i, file_list_dict)
        pbar.update(1)
    return file_list_dict
```

**原因**: 这是本次复现遇到的最核心问题。conda 环境中的 libstdc++.so.6.0.34 与 WSL2 Ubuntu 22.04 的系统 glibc 2.35 存在兼容性问题。当 Python 使用 `multiprocessing.Process` fork 子进程时，子进程继承了父进程的内存空间但不继承动态库加载状态，导致 glibc 的 malloc/free 内部数据结构不一致，触发 `corrupted double-linked list` 崩溃。

**解决方案**:
1. 设置 multiprocessing 启动方式为 `spawn`（而非默认的 `fork`），让子进程从头启动全新 Python 解释器
2. 提供环境变量 `RHYTHMMAMBA_SERIAL_PREPROCESS=1` 完全绕过 multiprocessing，在主进程串行执行预处理

**使用方式**:
```bash
RHYTHMMAMBA_SERIAL_PREPROCESS=1 python main.py --config_file config.yaml
```

**缺陷**:
- 串行模式比多进程慢（但 42 个视频文件差异不大，约几分钟）
- `spawn` 方式在部分场景下可能与 fork 行为有细微差异（目前未观察到）
- 理想方案是彻底解决 conda libstdc++ 与系统 glibc 的兼容性，但需要升级系统 glibc 或使用容器化方案

---

### 修改 5: `dataset/data_loader/UBFCrPPGLoader.py` — ffmpeg CLI 替代 OpenCV VideoCapture

**修改位置**: 第 10 行（新增 import subprocess），第 100-126 行（read_video 方法重写）

**改前**:
```python
@staticmethod
def read_video(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)
```

**改后**:
```python
@staticmethod
def read_video(video_file):
    """Reads a video file, returns frames(T, H, W, 3) using ffmpeg CLI."""
    # 用 ffprobe 获取视频尺寸
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
        video_file
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split(','))

    # 用 ffmpeg 解码视频到原始 RGB 管道
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_file, '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-v', 'error', 'pipe:'
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_data = proc.stdout.read()
    proc.wait()

    # 将原始字节转为 numpy 数组: (T, H, W, 3)
    frames = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, height, width, 3)
    return frames
```

**原因**: `cv2.VideoCapture()` 在打开视频文件时发生 segfault。根源仍然是 conda libstdc++.so.6.0.34 与系统 glibc 2.35 的兼容性问题。OpenCV 的 ffmpeg 后端（无论是 pip opencv-python 自带的还是 conda-forge 的）都会在视频解码时触发 glibc 的 `corrupted double-linked list` 或直接 segfault。

**解决方案**: 完全绕过 OpenCV 的视频解码功能，改用系统 ffmpeg CLI 工具（通过 subprocess 调用）。ffmpeg CLI 作为独立进程运行，拥有自己的内存空间，不受 Python 进程的库冲突影响。

**影响**:
- 读取 UBFC-rPPG 的 `.avi` 文件正常（MJPG 编码）
- 输出格式与原版一致：`np.ndarray` of shape `(T, H, W, 3)`, dtype `uint8`, RGB 通道顺序

**缺陷**:
- 依赖系统安装 ffmpeg CLI（conda 环境已包含）
- subprocess 方式比 OpenCV 内置解码稍慢（额外进程启动 + 管道传输开销）
- 对于其他编码格式的视频（如 H.264），可能需要调整 ffmpeg 参数
- 只改了 UBFCrPPGLoader，其他数据集的 Loader 如果也用 `cv2.VideoCapture` 会有同样问题

---

### 修改 6: `setup/mamba/setup.py` — 添加 sm_100/sm_120 编译目标

**修改位置**: 第 115-119 行

**新增代码**:
```python
if bare_metal_version >= Version("12.0"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_100,code=sm_100")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

**原因**: RTX 5070 是 Blackwell 架构（compute capability 12.0 = sm_120），而 mamba-ssm 的 setup.py 最高只编译到 sm_90。缺少 sm_120 的 CUDA kernel 会导致 `no kernel image is available for execution on the device` 错误。

**注意**: 由于 PyTorch 2.12.0 由 CUDA 13.0 编译，而系统 nvcc 是 12.8，存在版本不匹配。实际上我们使用的是 conda 环境中通过 `conda install -c nvidia/label/cuda-13.0.2 cuda-nvcc` 安装的 CUDA 13.0 nvcc 来编译 mamba-ssm 和 causal-conv1d 的 CUDA extension。

**缺陷**: 此 setup.py 修改针对项目自带的 mamba 源码。实际编译时使用的是 pip 安装的 mamba-ssm 2.3.2.post1 源码（通过 `pip install --no-build-isolation` 配合 `TORCH_CUDA_ARCH_LIST="12.0"` 编译），所以这个 setup.py 的修改更多是记录性质，实际编译并不依赖它。

---

## 三、其他非代码修改

### 7. mamba_ssm `__init__.py` 兼容 transformers 5.x

**文件**: `/root/anaconda3/envs/rhythmmamba527/lib/python3.11/site-packages/mamba_ssm/__init__.py`

**修改**: 将 `MambaLMHeadModel` 的导入用 try/except 包裹

```python
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    pass
```

**原因**: transformers 5.8.0 移除了 `GreedySearchDecoderOnlyOutput`，导致 mamba_ssm 的 `__init__.py` 无条件导入 `MambaLMHeadModel` 时崩溃。RhythmMamba 只使用 `mamba_ssm.modules.mamba_simple.Mamba`，不依赖 LLM 模型功能。

### 8. MMPDLoader.py 注释 scipy 导入

**说明**: 此修改在本次会话之前已完成，文件 `/root/RhythmMamba-main/dataset/data_loader/MMPDLoader.py` 中 `from scipy.__config__ import get_info` 已被注释。

---

## 四、数据预处理流程

### 预处理用到的文件

| 文件 | 作用 |
|------|------|
| `main.py` | 入口，读取配置，调用 DataLoader |
| `dataset/data_loader/BaseLoader.py` | 基类，提供预处理流水线：视频分片、人脸裁剪、resize、标准化、chunk、存储 |
| `dataset/data_loader/UBFCrPPGLoader.py` | UBFC-rPPG 专用，提供视频读取 (`read_video`) 和 BVP 信号读取 (`read_wave`) |
| `dataset/data_loader/__init__.py` | 模块导入 |
| `unsupervised_methods/methods.py` | POS_WANG 等无监督 PPG 估计方法（伪标签生成） |
| `configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml` | 预处理配置 |

### 预处理输入

```
/root/Datasets/UBFC/
├── subject1/
│   ├── vid.avi           # 视频文件 (640×480, 30 fps, MJPG 编码)
│   └── ground_truth.txt  # BVP 信号 (空格分隔的浮点数)
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
├── ...
└── subject42/
    ├── vid.avi
    └── ground_truth.txt
```

### 预处理步骤

1. **读取视频** → `UBFCrPPGLoader.read_video()` 使用 ffmpeg CLI 解码 avi → numpy 数组 `(T, H, W, 3)`
2. **人脸检测 + 裁剪** → OpenCV 人脸检测器 (配置: `DO_CROP_FACE: True`)
3. **Resize** → 缩放到 128×128 (配置: `RESIZE: {H: 128, W: 128}`)
4. **Chunk** → 按 160 帧切分片段 (配置: `CHUNK_LENGTH: 160`)
5. **标准化** → Z-score 标准化 (配置: `DATA_TYPE: ['Standardized']`)
6. **保存** → npy 文件保存到缓存目录

### 预处理输出

```
/root/RhythmMamba_Preprocessed/UBFC/
└── UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_.../
    ├── subject1_input0.npy    (~61 MB each, float32, NDCHW format)
    ├── subject1_label0.npy    (~1-2 KB each)
    ├── ...
    └── subject42_inputN.npy   (966 total npy files, ~29 GB)
```

### 预处理命令

```bash
# 首次运行，设置 DO_PREPROCESS: True
RHYTHMMAMBA_SERIAL_PREPROCESS=1 python main.py \
    --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml
```

---

## 五、训练流程

### 训练用到的文件

| 文件 | 作用 |
|------|------|
| `main.py` | 训练入口，构建 DataLoader 和 Trainer |
| `dataset/data_loader/BaseLoader.py` | 基类，`__getitem__` 懒加载 npy 文件 |
| `dataset/data_loader/UBFCrPPGLoader.py` | UBFC-rPPG 数据加载器 |
| `neural_methods/trainer/RhythmMambaTrainer.py` | RhythmMamba 训练器（训练循环、验证、测试） |
| `neural_methods/trainer/BaseTrainer.py` | 训练器基类 |
| `neural_methods/model/RhythmMamba.py` | RhythmMamba 模型定义（Mamba + 时空注意力） |
| `neural_methods/loss/NegPearsonLoss.py` | 负 Pearson 相关系数损失函数 |
| `mamba_ssm/modules/mamba_simple.py` | Mamba SSM 核心模块（外部库） |
| `causal_conv1d` | 因果卷积 CUDA kernel（外部库） |
| `configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml` | 训练超参数配置 |

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | RhythmMamba |
| Epochs | 30 |
| Batch Size | 8 |
| 学习率 | 3e-4 |
| 优化器 | AdamW (β1=0.9, β2=0.999) |
| 损失函数 | NegPearsonLoss |
| 输入格式 | NDCHW (Batch, Channel, Time, Height, Width) |
| 训练/测试分割 | 72% 训练 (0.0-0.72), 28% 测试 (0.72-1.0) |
| 训练样本数 | 342 |
| 测试样本数 | 141 |

### 训练命令

```bash
# 预处理完成后，将 yaml 中 DO_PREPROCESS 全部改为 False，然后：
python main.py --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml
```

### 训练后的测试评估

模型使用最后一个 epoch 的权重进行测试（`USE_LAST_EPOCH: True`），在 141 个测试样本上评估心率估计性能，使用 FFT 方法从预测的 rPPG 信号中提取心率。

---

## 六、测试结果

| 指标 | 值 | 说明 |
|------|-----|------|
| **MAE** | 0.54 ± 0.17 bpm | 平均绝对误差（心率） |
| **RMSE** | 0.79 ± 0.36 bpm | 均方根误差（心率） |
| **MAPE** | 0.56% ± 0.16% | 平均绝对百分比误差 |
| **Pearson** | 0.997 ± 0.023 | Pearson 相关系数（越接近 1 越好） |
| **SNR** | 8.32 ± 1.45 dB | 信噪比 |

---

## 七、已知问题与改进方向

### 1. libstdc++ / glibc 兼容性（核心问题）

- **现象**: conda libstdc++.so.6.0.34 与系统 glibc 2.35 冲突
- **当前方案**: 串行预处理 + ffmpeg CLI 绕过
- **根治方案**: 使用 Docker 容器统一 glibc/libstdc++ 版本，或升级 WSL2 到 Ubuntu 24.04 (glibc 2.39)

### 2. GPU 利用率偏低

- **现象**: 训练时 GPU 利用率 16-27%，存在优化空间
- **可能原因**: 数据加载（磁盘 I/O 或 subprocess 开销）成为瓶颈
- **改进方向**: 增大 num_workers 到 6-8（需更多系统内存），或用 GPU 直接解码视频

### 3. ffmpeg 方案仅适用于 UBFCrPPGLoader

- **现状**: 只修改了 UBFC-rPPG 的 `read_video`，其他数据集 (PURE, MMPD, VIPL-HR 等) 如果也有 segfault，需要同样修改
- **改进**: 可以将 ffmpeg 解码方案提取到 BaseLoader 作为通用方法

### 4. 串行预处理较慢

- **现状**: 42 个视频串行处理约需数分钟
- **改进**: 如果解决了 libstdc++ 兼容性，可以恢复多进程预处理

---

## 八、快速复现步骤

```bash
# 1. 创建 conda 环境
conda create -n rhythmmamba python=3.11 -y
conda activate rhythmmamba

# 2. 安装 CUDA 13.0 工具链
conda install -c nvidia/label/cuda-13.0.2 cuda-nvcc cuda-toolkit -y

# 3. 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. 源码编译 causal-conv1d 和 mamba-ssm (支持 sm_120)
TORCH_CUDA_ARCH_LIST="12.0" pip install causal-conv1d --no-build-isolation
TORCH_CUDA_ARCH_LIST="12.0" pip install mamba-ssm --no-build-isolation

# 5. 安装其他依赖
pip install -r requirements_rtx5070.txt

# 6. 修复 mamba_ssm transformers 兼容性
# 编辑 site-packages/mamba_ssm/__init__.py: MambaLMHeadModel 导入加 try/except

# 7. 下载 UBFC-rPPG 数据集到 /root/Datasets/UBFC/

# 8. 预处理 (首次)
RHYTHMMAMBA_SERIAL_PREPROCESS=1 python main.py \
    --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml

# 9. 训练
python main.py --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml
```

---

## 九、二次复现：DataLoader / CPU-GPU 协同优化

### 目标

在不改变模型结构、数据划分、训练轮数、学习率、损失函数和评价方式的前提下，优化训练阶段的数据加载与 CPU 到 GPU 的搬运效率，并与第一次复现结果进行对比。

本次二次复现保留第一次复现中的关键绕过方案：

- 预处理仍使用串行预处理模式（如需重新预处理：`RHYTHMMAMBA_SERIAL_PREPROCESS=1`）
- UBFC-rPPG 视频读取仍使用 ffmpeg CLI，绕过 OpenCV `VideoCapture`
- 训练阶段直接读取已缓存的 `.npy` clip

### 二次复现代码改动

#### 1. `main.py`：DataLoader 参数可配置化

新增 `build_data_loader()`，将 DataLoader 的 CPU/GPU 协同参数统一管理，并支持通过环境变量调节：

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `RHYTHMMAMBA_NUM_WORKERS` | `4` | 全局 DataLoader worker 数 |
| `RHYTHMMAMBA_TRAIN_NUM_WORKERS` | 继承全局 | 仅训练集 worker 数 |
| `RHYTHMMAMBA_TEST_NUM_WORKERS` | 继承全局 | 仅测试集 worker 数 |
| `RHYTHMMAMBA_PIN_MEMORY` | CUDA 设备下默认 `True` | 是否使用 pinned memory |
| `RHYTHMMAMBA_PERSISTENT_WORKERS` | `True` | epoch 间是否复用 worker |
| `RHYTHMMAMBA_PREFETCH_FACTOR` | `2` | 每个 worker 预取 batch 数 |

DataLoader 会在启动时打印实际参数，例如：

```text
train DataLoader: batch_size=8, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
test DataLoader: batch_size=2, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
```

#### 2. `neural_methods/trainer/RhythmMambaTrainer.py`：non-blocking 搬运

将训练、验证、测试中的数据搬运改为：

```python
data = data.to(self.device, non_blocking=True)
labels = labels.to(self.device, non_blocking=True)
```

并将训练中的清梯度改为：

```python
self.optimizer.zero_grad(set_to_none=True)
```

作用：

- 配合 `pin_memory=True`，允许 CPU 到 GPU 拷贝以 non-blocking 方式执行
- `set_to_none=True` 可减少部分梯度清零开销

#### 3. 新增监控脚本

新增文件：

| 文件 | 作用 |
|------|------|
| `tools/monitor_training.py` | 每隔固定时间读取训练日志、checkpoint 和 GPU 状态，写入 monitor 日志 |
| `tools/run_ubfc_dataloader_opt_run2.sh` | 启动二次复现训练，并将训练输出和监控输出写入日志文件 |

以后可用如下方式启动带日志监控的训练：

```bash
cd /root/RhythmMamba-main
./tools/run_ubfc_dataloader_opt_run2.sh
```

监控日志查看：

```bash
tail -f logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.monitor.log
```

### 二次复现配置

新增配置文件：

```text
configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2.yaml
```

该配置与第一次复现配置保持一致，仅修改模型保存名，避免覆盖第一次复现 checkpoint：

```yaml
MODEL_FILE_NAME: UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2
```

测试配置文件：

```text
configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2_TEST.yaml
```

该配置使用 `only_test` 模式，加载二次复现最终模型：

```yaml
TOOLBOX_MODE: "only_test"
INFERENCE:
  MODEL_PATH: "/experiment0/user/PreTrainedModels/.../UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2_Epoch29.pth"
```

### 二次复现训练命令

实际使用命令：

```bash
cd /root/RhythmMamba-main

/usr/bin/env \
LD_LIBRARY_PATH=/root/anaconda3/envs/rhythmmamba527/lib \
RHYTHMMAMBA_NUM_WORKERS=4 \
RHYTHMMAMBA_PREFETCH_FACTOR=2 \
RHYTHMMAMBA_PIN_MEMORY=1 \
RHYTHMMAMBA_PERSISTENT_WORKERS=1 \
/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
--config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2.yaml
```

二次复现测试命令：

```bash
cd /root/RhythmMamba-main

/usr/bin/env \
LD_LIBRARY_PATH=/root/anaconda3/envs/rhythmmamba527/lib \
RHYTHMMAMBA_NUM_WORKERS=4 \
RHYTHMMAMBA_PREFETCH_FACTOR=2 \
RHYTHMMAMBA_PIN_MEMORY=1 \
RHYTHMMAMBA_PERSISTENT_WORKERS=1 \
/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
--config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2_TEST.yaml \
> logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.only_test.log 2>&1
```

### 二次复现产物

最终模型：

```text
/experiment0/user/PreTrainedModels/UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2_Epoch29.pth
```

测试日志：

```text
/root/RhythmMamba-main/logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.only_test.log
```

监控日志：

```text
/root/RhythmMamba-main/logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.monitor.log
```

### 第一、第二次复现参数对比

| 参数 | 第一次复现 | 二次复现 Run2 |
|------|------------|---------------|
| Conda 环境 | `rhythmmamba527` | `rhythmmamba527` |
| Python | `3.11.15` | `3.11.15` |
| PyTorch | `2.12.0+cu130` | `2.12.0+cu130` |
| GPU | RTX 5070 12GB | RTX 5070 12GB |
| CUDA Toolkit / Runtime | CUDA 13.0 | CUDA 13.0 |
| 数据集 | UBFC-rPPG | UBFC-rPPG |
| 数据缓存 | `/root/RhythmMamba_Preprocessed/UBFC/` | 同左 |
| 输入尺寸 | `160 x 3 x 128 x 128` | 同左 |
| 数据类型 | `Standardized` | 同左 |
| 标签类型 | `Standardized` | 同左 |
| Train split | `0.0 - 0.72` | 同左 |
| Test split | `0.72 - 1.0` | 同左 |
| 训练样本数 | `342` clips | `342` clips |
| 测试样本数 | `141` clips | `141` clips |
| Epochs | `30` | `30` |
| Batch size | `8` | `8` |
| Inference batch size | `2` | `2` |
| LR | `3e-4` | `3e-4` |
| Optimizer | AdamW | AdamW |
| 训练增强 | `AUG: 1` | `AUG: 1` |
| 测试方法 | FFT | FFT |
| USE_LAST_EPOCH | `True` | `True` |
| DataLoader workers | `4` | `4` |
| `pin_memory` | 未显式开启 | `True` |
| `persistent_workers` | 未显式开启 | `True` |
| `prefetch_factor` | 未显式配置 | `2` |
| CPU->GPU 搬运 | `.to(device)` | `.to(device, non_blocking=True)` |
| 清梯度 | `zero_grad()` | `zero_grad(set_to_none=True)` |

### 第一、第二次复现结果对比

| 指标 | 第一次复现 | 二次复现 Run2 | 变化 |
|------|------------|---------------|------|
| **MAE** | `0.54 ± 0.17 bpm` | `0.540054 ± 0.168391 bpm` | 基本持平 |
| **RMSE** | `0.79 ± 0.36 bpm` | `0.794938 ± 0.361862 bpm` | 基本持平，略高 |
| **MAPE** | `0.56% ± 0.16%` | `0.555209% ± 0.164368%` | 略好 |
| **Pearson** | `0.997 ± 0.023` | `0.997358 ± 0.022973` | 略好 |
| **SNR** | `8.32 ± 1.45 dB` | `8.524459 ± 1.437678 dB` | 提升约 `+0.20 dB` |

二次复现完整输出：

```text
FFT MAE (FFT Label): 0.5400540054005383 +/- 0.1683914215469826
FFT RMSE (FFT Label): 0.7949379717666831 +/- 0.3618622940730566
FFT MAPE (FFT Label): 0.5552086596956083 +/- 0.16436819264736727
FFT Pearson (FFT Label): 0.9973577707377148 +/- 0.022972760280592922
FFT SNR (FFT Label): 8.524458794506911 +/- 1.4376775617814308 (dB)
```

### 二次复现效率记录

从 checkpoint 保存时间和监控日志统计：

| 项目 | 数值 |
|------|------|
| checkpoint 数 | `30` (`Epoch0` - `Epoch29`) |
| 平均 epoch 间隔 | 约 `63s` |
| 最快 epoch 间隔 | 约 `57s` |
| 最慢 epoch 间隔 | 约 `87s` |
| 显存占用 | 约 `7.8GB / 12GB` |
| GPU 利用率采样峰值 | 约 `89%` |
| GPU 温度采样范围 | 约 `40-53°C` |

说明：

- GPU 利用率是 `nvidia-smi` 的瞬时采样，波动较大；有时采到数据等待或同步间隙会显示 `1%`。
- 二次复现的 `pin_memory + persistent_workers + non_blocking` 对训练稳定性和吞吐有帮助，但对最终 HR 误差指标影响不大。
- SNR 相比第一次复现有小幅提升，MAE/RMSE/MAPE/Pearson 基本复现一致，说明该复现结果稳定。

### 结论

二次复现证明：在不改变模型、训练超参数、数据划分和评价方式的前提下，仅优化 DataLoader 与 CPU-GPU 搬运，最终指标与第一次复现高度一致，SNR 有小幅提升。

如果下一步目标是显著提升 MAE/RMSE/Pearson，主要方向应转向模型结构、损失函数、增强策略或测试时后处理，而不是继续只调 DataLoader。
