# 自然语言处理任务

本仓库包含了自然语言处理基于深度学习的文本分类任务中的几个重要模型实现，包括基于PyTorch构建的前馈神经网络训练器以及针对情感分析的增强模型。这些模型包括循环神经网络（RNN）、卷积神经网络（CNN）和Transformer等组件。

## 目录
- [GitHub地址](#github地址)
- [项目结构](#项目结构)
- [安装](#安装)
- [模型](#模型)
  - [前馈神经网络训练器](#前馈神经网络训练器-1)
  - [增强的情感分析模型](#增强的情感分析模型)
- [贡献指南](#贡献指南)

## GitHub地址

详情请查看项目主页：[https://github.com/scottzwells/NLP_task](https://github.com/scottzwells/NLP_task)

## 项目结构

```
NLP_task/
|-- main.py
|-- README.md
|-- data/
|   |-- glove.6B.zip
|   |-- sentiment-analysis-on-movie-reviews/
|   |   |-- test.tev.zip
|   |   |-- train.tsv.zip
```

- `main.py`: 主要的源代码文件。
- `README.md`: 当前文档。
- `data/`: 数据集目录。

## 安装

为了运行此项目，请确保已安装以下依赖项：
确保已安装以下依赖库：
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

## 模型

### 前馈神经网络训练器

`NetTrainer` 是一个简易的训练器，用于训练前馈神经网络。它提供了训练和评估功能。

### 增强的情感分析模型

增强的情感分析模型包括以下几个关键组件：
- **GRU**：用于捕获序列信息并处理变长序列。双向GRU有助于捕获上下文信息。
- **残差CNN**：CNN层用于提取局部特征，残差连接有助于保留特征并缓解梯度消失问题。
- **Transformer**：Transformer层增强了全局依赖关系的建模能力。Transformer在处理自注意力机制时能够捕捉到更长距离的依赖关系，补充了GRU和CNN对局部特征的捕捉。
- **全连接层**：最后通过全连接层将特征映射到情感分类任务的5个类别上。

## 贡献指南

欢迎贡献！如果您想参与改进本项目，请按照以下步骤操作：
1. Fork 该项目。
2. 在您的仓库中创建一个新的分支。
3. 实现您的更改。
4. 提交更改并推送到您的仓库。
5. 向本项目提交 Pull Request。
