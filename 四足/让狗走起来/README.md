# 《手推到狗跑》让狗走起来，要解决哪几件事？

配套 B 站视频：《手推到狗跑》让狗走起来，要解决哪几件事？（链接见视频简介）

同一只 Lite3 机器狗（MuJoCo 仿真）、同一个关节 PD，换三种“上层”比一比：可皮的凑合（四条腿一起摆）、可迪的手写小跑（脚的轨迹 + 逆运动学）、云深处官方开源的走路策略。再用侧推测试看谁更稳、谁会突然垮掉。

## 怎么用

1. **先看练习单**：`练习单.pdf`。第 1、2 题是视频里的“手推挑战”（一条腿的正 / 逆运动学），推一遍你就能写自己的步态；答案在 `练习单_答案.pdf`。
2. **再跑 notebook**：`验证.ipynb`，从上往下运行，复现视频里的数字；最后三格是作业。

## 运行环境

Python 3.10+：

```
pip install -r requirements.txt
jupyter notebook 验证.ipynb
```

不想装环境：直接在网页上看 notebook（图已经跑好，作业格子留空）。

## 文件

- `walk_sim.py`：仿真代码（notebook 开头也内嵌了一份；单独下载 notebook 时，缺的模型和策略文件会自动从本仓库下载）
- `lite3/`：云深处 Lite3 的 MuJoCo 模型，来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model)，BSD-3-Clause 许可（见 `lite3/LICENSE.txt`）
- `lite3_policy.npz`：官方走路策略的网络权重，从 [DeepRoboticsLab/Lite3_rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy) 的 `policy/ppo/policy.onnx` 原样读出（`onnx_to_npz.py`），BSD-3-Clause 许可（见 `LICENSE_Lite3_rl_deploy.txt`）。这样不用装 onnxruntime，numpy 就能跑。

## 说明

- 官方策略只是搬进 MuJoCo 运行，没有重新训练；观测、动作缩放、关节 PD（Kp=30、Kd=1）、12 ms 的控制周期都和官方部署代码一致。
- 手写小跑是教学用的最简单写法，不是 Lite3 官方步态。
- 所有结果都是仿真，参数为教学设置，不代表实机。
