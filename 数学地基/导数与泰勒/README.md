# 《手推到狗跑》导数到底有什么用？手推泰勒：放大到最后，曲线都是直线

配套 B 站视频：《手推到狗跑》导数到底有什么用？手推泰勒：放大到最后，曲线都是直线（链接见视频简介）

这一期是数学地基的第一期：导数、一阶泰勒、偏导和雅可比。**先手推，再用代码验证。**

## 怎么用

1. **先手推**：打印 `练习单.pdf`，五道题都给了起点和终点，中间自己写。
2. **再对答案**：`练习单_答案.pdf`。
3. **最后用代码验证**：`jupyter lab 验证.ipynb`，从上往下运行。前五节只要 numpy 和 matplotlib；最后一节“射门”要 mujoco（用上一期建好的 `shoutui` 环境即可），notebook 里会直接放出踢球动画。
4. **想转着看**：终端里运行 `python sim.py --view`，弹出 MuJoCo 的 3D 窗口回放“射门”（鼠标拖动旋转、滚轮缩放）。macOS 用 `mjpython sim.py --view`。

## 文件

- `练习单.pdf` / `练习单_答案.pdf`：x² 的导数、一阶泰勒、sinθ ≈ θ 的范围、Lite3 前腿的雅可比、最优差分步长。
- `验证.ipynb`：割线 → sinθ ≈ θ → 雅可比（手推 vs 差分）→ 直线预测什么时候失效 → 步长的 V 字 → MuJoCo 踢球 → 作业。
- `sim.py`：notebook 用到的全部代码（notebook 开头也内嵌了一份）。
- `lite3/`：云深处 Lite3 的 MuJoCo 模型，来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model)，BSD-3-Clause 许可（见 `lite3/LICENSE.txt`）。

## 说明

- 腿的几何取自官方模型：大腿 0.20 m、小腿 0.21012 m，站姿髋 −0.8、膝 1.6 弧度。
- 踢球仿真里机身固定在检修架上，只有前右腿髋关节转 60°，其他关节用 PD（Kp=60、Kd=1.5）锁在站姿；参数为教学设置，不代表实机。
