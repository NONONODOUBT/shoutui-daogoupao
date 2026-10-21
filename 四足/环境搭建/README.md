# 《手推到狗跑》明明装了 mujoco，为什么说找不到？

配套 B 站视频：《手推到狗跑》明明装了 mujoco，为什么说找不到？（链接见视频简介）

这一期不跑新算法，只做一件事：把课程环境装对，并且**证明它装对了**。

## 三步

1. **建环境**（装好 [Miniforge](https://github.com/conda-forge/miniforge) 或 Anaconda 之后，在本文件夹里打开终端）：

   ```
   conda env create -f environment.yml
   conda activate shoutui
   python -m ipykernel install --user --name shoutui --display-name "shoutui"
   ```

   作者实测：约 3–4 分钟，环境占 1.9 GB。

2. **体检**：`jupyter lab 体检.ipynb`，右上角内核选 `shoutui`，从上往下运行。6 层全是 ✓ 就可以开始后面的课了；哪一层出 ✗，就先修那一层。
   不想开 Jupyter：`python p03_check.py` 结果一样。

3. **做体检清单**：`练习单.pdf`（答案在 `练习单_答案.pdf`），最后三道是作业。

## 文件

- `environment.yml`：课程环境（Python 3.11、mujoco 3.16、numpy、matplotlib、Jupyter）。**只写英文**：中文版 Windows 上 conda 可能按 GBK 读这个文件。
- `p03_check.py`：体检脚本（notebook 开头也内嵌了一份）。
- `体检.ipynb`：一键体检 + 拆开的每一层 + 作业。输出没有预先跑：体检结果只对你自己的电脑有意义。
- `指南_缓存搬家与镜像源.md`：pip / conda / 模型缓存怎么搬出 C 盘，国内镜像源怎么配，什么时候需要 Ubuntu。
- `lite3/`：云深处 Lite3 的 MuJoCo 模型，来自 [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model)，BSD-3-Clause 许可（见 `lite3/LICENSE.txt`）。

## 说明

- 体检第 6 层“Lite3 站起来”只用关节 PD（Kp=60、Kd=1.5）锁站姿；参数为教学设置，不代表实机。
- 以后的显卡训练要另建一个环境，别往 `shoutui` 里装。
