# 《手推到狗跑》一只狗怎么站起来？控制和学习各管什么

配套 B 站视频：《手推到狗跑》一只狗怎么站起来？学出来的更准，为什么先翻的是它（链接见视频简介）

Lite3 机器狗站在一块会晃的板子上（MuJoCo 仿真），对比四种做法：关掉电机、只有关节 PD、手推平衡层、学出来的平衡层；再看加上延迟以后谁先翻。

## 怎么用

1. **先看练习单**：`练习单.pdf`，这一期不强制手推，但五道题能帮你吃透腿长几何和“快与稳的取舍”；答案在 `练习单_答案.pdf`。
2. **再跑 notebook**：`验证.ipynb`，从上往下运行，复现视频里的数字；最后三格是作业。

## 运行环境

Python 3.10+：

```
pip install -r requirements.txt
jupyter notebook 验证.ipynb
```

不想装环境：直接在网页上看 notebook（图已经跑好，作业格子留空）。

## 文件

- `balance_sim.py`：仿真代码（notebook 开头也内嵌了一份，单独下载 notebook 也能跑，但需要同目录的 `lite3/` 模型文件夹）
- `lite3/`：云深处 Lite3 的 MuJoCo 模型，来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model)，BSD-3-Clause 许可（见 `lite3/LICENSE.txt`）

## 说明

- 平衡层是教学用的简化做法，不是 Lite3 官方控制器。
- 学习部分是 4 个参数的线性策略 + 随机搜索，属于强化学习的入门实验，不是深度强化学习。
- 所有结果都是仿真，参数为教学设置，不代表实机。
