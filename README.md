# HBFormer
Accepted by IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2025)

# 🛎 Citation

- **多数据集**: Synapse（8器官）、LiTS2017（肝脏/肿瘤）、ACDC（心脏3类）、Bladder（膀胱/肿瘤）。
- **两种验证模式（Synapse）**: 体数据验证（.npy.h5）与切片验证（.npz，与训练一致）。
- **可视化**: 预测对比、注意力/激活热图、3D/切片交互查看。
- **日志与记录**: 训练/验证 CSV 记录、best/latest 权重保存、可配置保存间隔。
---

If you find our work helpful for your research, please cite:

```bib
Coming Soon
```
# 📋HBFormer
HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation

**University of Macau & SIAT CAS**

# 项目结构
```
AFFSegNnet_VMUnetVis/
├── configs/                     # 各数据集训练配置
│   ├── config_setting_synapse.py
│   ├── config_setting_lits2017.py
│   ├── config_setting_ACDC.py
│   └── config_setting_bladder.py
├── datasets/                    # 数据集加载与增广
│   ├── dataset.py               # Synapse（.npz 切片 / .npy.h5 体）
│   ├── dataset_lits2017.py
│   ├── dataset_ACDC.py
│   └── dataset_bladder.py
├── models/                      # 模型实现
│   ├── DWSegNet.py
│   ├── HBFormer.py
│   ├── DZZMamba.py
│   └── vmunet/
├── engine_synapse.py            # 训练/验证/评估（Synapse专用）
├── engine_lits2017.py           # LiTS2017 训练/验证
├── engine_ACDC.py               # ACDC 训练/验证
├── train_synapse.py             # Synapse 训练入口
├── train_lits2017.py            # LiTS2017 训练入口
├── train_ACDC.py                # ACDC 训练入口
├── train_bladder.py             # Bladder 训练入口
├── test_synapse.py              # Synapse 体数据评估（如有）
├── pre_trained_weights/         # 预训练/外部权重
├── results/                     # 输出：日志/权重/评估
├── 3D_Vis/                      # 3D/切片查看工具
└── utils.py                     # 通用工具、日志、优化器/调度器构建
```

---

## 环境安装

### 推荐（Python 3.8 + PyTorch 1.13，CUDA 11.7）
```bash
conda create -n vmunet python=3.8 -y
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging timm==0.4.12 pytest chardet yacs termcolor
pip install submitit tensorboardX triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
pip install opencv-python pandas tqdm nibabel scipy einops
```

可选：
```bash
pip install ptflops  # 计算FLOPs/参数量
```

也可参考：`ui/vmunet_environment.yml` 与 `ui/vmunet_conda_env.txt`。

---

## 数据集准备

### Synapse（8类）
目录要求（与代码默认一致）：
```
./data/Synapse/
├── lists/
│   └── lists_Synapse/
│       ├── all.lst
│       ├── train.txt
│       ├── test_vol.txt        # 体数据验证/测试
│       └── test_slice.txt      # 切片验证（与训练一致的数据预处理）
├── train_npz/                  # 切片级 .npz（训练 & 切片验证使用）
│   └── caseXXXX_sliceYYY.npz   # 含键 image/label
└── test_vol_h5/                # 体数据 .npy.h5（体验证使用）
    └── caseXXXX.npy.h5         # 含键 image/label
```
获取方式：可参考 Swin-UNet 的 Synapse 数据下载与预处理，或使用你已有的预处理结果。若需自制 `.npz`，可参考 `data_proprecessing.py` 示例将 NIfTI 切片为 `.npz`（确保 `.npz` 内含键 `image` 与 `label`，并与 `test_slice.txt` 命名一致）。

`test_slice.txt` 样例（每行一个切片名，不带扩展名）：
```text
case0001_slice001
case0001_slice002
...
```

### LiTS2017（肝脏/肿瘤）
```
./data/LITS2017_nii/
├── CT/                       # volume-*.nii
└── seg/                      # segmentation-*.nii
```

### ACDC（心脏3类）
```
./data/ACDC/
└── database/
    ├── training/patient*/patient*_frame*.nii.gz
    └── testing/patient*/
```

### Bladder（膀胱/肿瘤）
```
./data/Bladder/
├── images ...
└── masks  ...
```

---

## 使用与训练

所有训练入口脚本会读取对应配置文件（`configs/`），日志/权重输出目录位于 `results/{Model}_{Dataset}/`。

### 选择/切换模型（以 Synapse 为例）
在 `configs/config_setting_synapse.py` 中：
- `available_models = ['DWSegNet', 'AFFSegNet', 'vmunet', 'HBFormer']`
- 默认 `network = 'HBFormer'`
- 可通过类方法切换：
```python
from configs.config_setting_synapse import setting_config
setting_config.set_model('DWSegNet')  # 或 'HBFormer'、'vmunet'、'AFFSegNet'
```

### 训练命令
```bash
# Synapse（包含切片验证）
python train_synapse.py

# LiTS2017（支持交叉验证）
python train_lits2017.py

# ACDC
python train_ACDC.py

# Bladder
python train_bladder.py
```

### 验证与测试
- Synapse：
  - 切片验证由 `train_synapse.py` 内调用 `val_one_epoch_slice` 自动完成（读 `train_npz/` 与 `test_slice.txt`）。
  - 体验证（可选）使用 `engine_synapse.val_one_epoch`（读 `test_vol_h5/` 与 `test_vol.txt`）。
- 其他数据集：对应 `engine_*.py` 中的验证逻辑与 `test_*.py` 测试脚本（如提供）。

训练过程会在 `results/.../log/` 写入训练/验证日志，并将指标记录到 CSV：
- `train_record.csv`（loss、avg_dice、mIoU、各器官Dice）
- `val_record.csv`（avg_dice、avg_hd95、mIoU、各器官Dice/HD95）

权重保存：
- `checkpoints/best.pth` / `best_dice.pth`（最佳）
- `checkpoints/latest.pth`（最近）
- 支持 `save_interval` 定期保存（见各配置）。

---

## 预训练权重

目录：`pre_trained_weights/`

### VMUNet 预训练（示例）
1) 将 `vmamba_small_e238_ema.pth` 放至 `pre_trained_weights/`。
2) 在 `configs/config_setting_synapse.py` 中切换到 VMUNet：
```python
setting_config.set_model('vmunet')
```
3) 在同文件的 `vmunet_config` 中设置：
```python
"load_ckpt_path": "pre_trained_weights/vmamba_small_e238_ema.pth"
```
4) 运行训练：
```bash
python train_synapse.py
```

提示：若使用 `AFFSegNet`/`DWSegNet`/`HBFormer` 的外部权重，可在各自 `model_config` 中设置 `load_ckpt_path`（若脚本支持），或在训练脚本中按需加载。

---

## 可视化与3D查看
- 训练/验证过程支持保存预测对比、注意力/激活热图（见 `engine_synapse.py` 相关函数）。
- 目录 `3D_Vis/` 与 `Pathplanning/` 提供三维与切片查看工具，可按需运行其中的 `run.py`/`main.py` 类型脚本进行交互式浏览。

---

## 常见问题（FAQ）
- 切片验证报错尝试访问 `.npy.h5`？请确认 `datasets/dataset.py` 的逻辑：`split in ["train","test_slice"]` 读取 `.npz`，`split in ["test_vol","val_vol"]` 读取 `.npy.h5`，并确保 `list_dir` 下存在对应的 `test_slice.txt`/`test_vol.txt`。
- CUDA/依赖冲突：优先使用上文给定版本；或参考 `ui/vmunet_environment.yml` 创建一致环境。
- 日志/权重找不到：检查 `results/{Model}_{Dataset}/` 是否已创建；脚本会自动创建必要目录。


---

## 引用
若本项目对您的研究有帮助，欢迎引用或在论文中致谢本仓库。


