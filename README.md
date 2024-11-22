## README

### 项目名称: 基于C2FEMBC和GFPN增强高分辨率遥感目标检测性能的研究

### 项目简介
随着高分辨率遥感图像在多个应用领域的需求增长，传统的目标检测模型在复杂场景中的检测精度和效率面临挑战。该项目通过在YOLOv8模型中引入Cross-Stage Feature Enhancement with Modified Bottleneck Convolution（C2FEMBC）和Generalized Feature Pyramid Network（GFPN）两项创新增强技术，优化特征提取和多尺度表示，尤其在检测小物体和复杂模式上表现突出。实验表明，相较于基线YOLOv8模型，修改后的模型在mAP50:95指标上提高了59.4%，显著提升了检测性能。

### 主要功能
1. **C2FEMBC改进**：改进YOLOv8骨干网络，增强了小物体检测能力，特别是复杂背景下的细节处理。
2. **GFPN改进**：优化了特征金字塔网络，提升了不同尺度特征的融合，增强了小物体在复杂背景下的检测效果。
3. **多尺度目标检测**：结合C2FEMBC与GFPN，提升了在高分辨率遥感图像中的目标检测精度。

### 技术栈
- **YOLOv8**：最新的目标检测框架，已做性能增强。
- **PyTorch**：用于深度学习模型的训练与推理。
- **C2FEMBC与GFPN**：两项核心创新技术，用于增强目标检测性能。

### 数据集
本研究使用了NWPU提供的高分辨率遥感图像数据集，数据集包含10类地理物体（如飞机、船只、油罐、桥梁和车辆等），共计800张图像。每张图像至少包含一个目标，数据集适合高分辨率遥感图像中的目标检测。

### 安装与依赖
```bash
pip install torch torchvision
```

### 使用方法
1. 克隆此仓库：
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```
2. 数据集预处理：
   将数据集格式化为适用于YOLOv8训练的格式。
3. 训练模型：
   ```bash
   python train.py --epochs 200 --batch_size 16
   ```
4. 进行推理：
   ```bash
   python infer.py --input <image_path> --output <output_path>
   ```

### 实验与结果
在高分辨率遥感图像数据集上进行的实验表明，结合C2FEMBC和GFPN的YOLOv8模型在精度、召回率以及mAP指标上均超过了原始YOLOv8模型，尤其在复杂背景下的小物体检测上有显著提升。

### 贡献
- **Zhijiang Li**: 主要贡献者，负责模型设计与实验。
- **团队成员**: 提供了数据集、模型改进和实验支持。

### 许可证
本项目采用MIT许可证，详细信息请参见 [LICENSE](LICENSE) 文件。

### 联系方式
如有任何问题，欢迎通过以下方式联系：
- 邮箱：20030902@bupt.edu.cn
