"""让狗走起来，要解决哪几件事？三种“走路”放在同一套测试里比（MuJoCo，Lite3）。

    kepi   可皮的凑合：四条腿的目标角一起按正弦来回摆（没有步态时序、没有落脚点、没有平衡）
    trot   手写小跑：对角两条腿一组、轮流抬腿往前迈（有步态时序；落脚点、平衡都是固定的）
    rl     官方 RL 策略：DeepRoboticsLab/Lite3_rl_deploy 的 policy.onnx（PPO，BSD-3-Clause）

三种做法底下是同一个关节 PD（和官方部署代码一致：Kp=30、Kd=1，1000 Hz）；
上层每 12 ms 给一次 12 个关节的目标角——官方策略的输出也正是这个。
模型：DeepRoboticsLab/deep_robotics_model 的 Lite3 MJCF（BSD-3-Clause）。参数为教学设置，不代表实机。
运行环境：pip install mujoco（测试版本 3.16）；直接运行本文件 = 跑三种做法的“走 10 秒 + 侧推”测试。
"""
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                  # 在 notebook 格子里运行时没有 __file__
    _HERE = Path.cwd()
# 发布包里模型放在同文件夹的 lite3/；找不到时退回作者本地的模型路径
MJCF = next((p for p in [_HERE / "lite3", _HERE.parents[1] / "assets" / "deep_robotics_model-main" / "Lite3" / "mjcf"]
             if (p / "Lite3.xml").exists()), _HERE / "lite3")
POLICY = _HERE / "lite3_policy.npz"

# ---------------- 和官方部署代码一致的常数（Lite3_rl_deploy/run_policy） ----------------
DT = 0.001                                      # 物理步长 1 ms
DECIM = 12                                      # 上层每 12 步（12 ms）给一次目标角
KP, KD, TAU_MAX = 30.0, 1.0, 30.0               # 关节 PD
Q0 = np.tile([0.0, -0.65, 1.30], 4)             # 默认站姿：每条腿 [髋侧摆, 髋前后, 膝]，腿序 FL FR HL HR
ACT_SCALE = np.tile([0.125, 0.25, 0.25], 4)     # 策略输出 × 这个 = 目标角相对站姿的偏移
CMD_MAX = 0.8                                   # 速度指令上限（m/s、rad/s），训练时见过的范围
LEGS = ["FL", "FR", "HL", "HR"]


REPO = ["https://gitee.com/nononodoubt/shoutui-daogoupao/raw/main/四足/让狗走起来/",
        "https://raw.githubusercontent.com/NONONODOUBT/shoutui-daogoupao/main/四足/让狗走起来/"]
LITE3_FILES = ["lite3/Lite3.xml", "lite3/LICENSE.txt"] + [f"lite3/meshes/{n}.STL" for n in ("torso", "hip", "thigh", "shank")]


def download(files, dst):
    """只拿到 notebook、旁边缺文件时，从配套仓库下载（国内先走 Gitee）。files 是相对仓库这一期文件夹的路径。"""
    from urllib.parse import quote
    from urllib.request import urlopen
    for base in REPO:
        try:
            for f in files:
                data = urlopen(quote(base + f, safe=":/"), timeout=30).read()
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                (dst / f).write_bytes(data)
            print(f"已下载 {', '.join(Path(f).name for f in files)} 到 {dst}")
            return
        except OSError as e:
            print(f"从 {base.split('/')[2]} 下载失败（{e}），换一个地址")
    raise FileNotFoundError(f"缺少 {files}：请从配套仓库下载，放到 {dst}")


def load_model():
    """MuJoCo 在 Windows 上打不开中文路径：自己读 XML 和网格，以内容形式传进去。"""
    if not (MJCF / "Lite3.xml").exists():
        download(LITE3_FILES, MJCF.parent)
    xml = (MJCF / "Lite3.xml").read_text(encoding="utf-8")
    assets = {f"./meshes/{p.name}": p.read_bytes() for p in (MJCF / "meshes").iterdir()}
    spec = mujoco.MjSpec.from_string(xml, assets)
    spec.assets = {k[2:]: v for k, v in assets.items()}
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.76, 0.92], rgb2=[0.09, 0.13, 0.23], width=512, height=3072)
    for t in spec.textures:
        if t.name == "checker":
            t.rgb1, t.rgb2 = [0.70, 0.75, 0.82], [0.86, 0.89, 0.93]
    m = spec.compile()
    m.opt.timestep = DT
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    vis = (m.geom_bodyid > 0) & (m.geom_group == 2) & (m.geom_matid < 0)
    m.geom_rgba[vis] = [0.58, 0.60, 0.64, 1]
    return m


# ---------------- 上层 1：官方 RL 策略 ----------------
class RLPolicy:
    """45 维观测 → 12 维动作的小网络（45→512→256→128→12，ELU）。权重由 onnx_to_npz.py 从 policy.onnx 读出。"""

    def __init__(self, path=POLICY):
        if not Path(path).exists():
            download([Path(path).name], Path(path).parent)
        w = np.load(path)
        self.layers = [(w[f"mlp_{i}_weight"], w[f"mlp_{i}_bias"]) for i in (0, 2, 4, 6)]
        self.last = np.zeros(12)

    def obs(self, d, cmd):
        """观测（顺序和缩放与官方 runner 一致）：机身角速度×0.25、重力在机身里的方向、速度指令、
        关节角 − 站姿、关节角速度×0.05、上一次的动作。注意：没有“机身速度”，它得自己从别的量里猜。"""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        grav = R.reshape(3, 3).T @ np.array([0, 0, -1.0])
        return np.concatenate([d.qvel[3:6] * 0.25, grav, np.clip(cmd, -CMD_MAX, CMD_MAX),
                               d.qpos[7:19] - Q0, d.qvel[6:18] * 0.05, self.last])

    def __call__(self, d, t, cmd):
        x = self.obs(d, cmd)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < 3:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))      # ELU
        self.last = x
        return Q0 + x * ACT_SCALE


# ---------------- 上层 2、3：手写 ----------------
def kepi_targets(amp=0.2, freq=2.5):
    """可皮：四条腿一起蹲起——髋前后和膝按同一个正弦摆（膝摆两倍，脚大致在髋正下方上下动）。"""
    def f(d, t, cmd):
        s = amp * np.sin(2 * np.pi * freq * t)
        return Q0 + np.tile([0.0, -s, 2 * s], 4)
    return f


L1, L2 = 0.20, 0.21012          # 大腿、小腿长（m）
H0 = 0.3265                     # 默认站姿下脚在髋下方的深度（m）


def leg_ik(x, z):
    """脚在髋坐标系里的位置（x 往前、z 往上，单位 m）→ (髋前后, 膝)。
    正运动学：x = L1·sin(髋) + L2·sin(髋+膝)，z = −L1·cos(髋) − L2·cos(髋+膝)。"""
    D2 = x * x + z * z
    knee = np.arccos(np.clip((D2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1))
    hip = np.arctan2(x, -z) - np.arctan2(L2 * np.sin(knee), L1 + L2 * np.cos(knee))
    return hip, knee


def trot_targets(freq=3.0, step=0.30, lift=0.16, duty=0.5, height=0.23):
    """手写小跑：FL+HR 一组、FR+HL 一组，相位差半个周期。
    每只脚的轨迹是写死的：着地时贴着地面从前往后匀速蹬 step 米，离地时抬高 lift 米再迈回前面；
    再用 leg_ik 反解出关节角。脚落在哪儿、身子歪了怎么办——都没写，所以一推就……"""
    phase = np.array([0.0, 0.5, 0.5, 0.0])
    def f(d, t, cmd):
        q = Q0.copy()
        p = (freq * t + phase) % 1.0
        stance = p < duty
        u = np.where(stance, p / duty, (p - duty) / (1 - duty))        # 本阶段进度 0→1
        x = np.where(stance, step / 2 - step * u, -step / 2 * np.cos(np.pi * u))
        z = -height + np.where(stance, 0.0, lift * np.sin(np.pi * u))
        hip, knee = leg_ik(x, z)
        q[1::3], q[2::3] = hip, knee
        return q
    return f


# ---------------- 测试 ----------------
@dataclass
class Case:
    mode: str = "rl"                     # kepi / trot / rl
    T: float = 10.0
    cmd: tuple = (0.5, 0.0, 0.0)         # 速度指令（只有 rl 用）：前进 m/s、侧移 m/s、转向 rad/s
    pushes: list = field(default_factory=list)   # [(开始秒, 侧向力 N, 持续秒)]
    settle: float = 0.5                  # 开始前先按站姿站稳
    kw: dict = field(default_factory=dict)       # 传给手写步态的参数
    upper: object = None                 # 自定义上层（工厂函数，返回 f(d, t, cmd) → 12 个目标角），给 mode="custom" 用


def make_upper(c):
    if c.mode == "custom":
        return c.upper()
    return {"rl": lambda: RLPolicy(), "kepi": lambda: kepi_targets(**c.kw),
            "trot": lambda: trot_targets(**c.kw)}[c.mode]()


def simulate(c: Case, m=None, traj=None, fps=30):
    """返回日志：t、机身位置 xyz、横滚/俯仰/偏航、四只脚是否着地、脚相对机身的左右位置 foot_y（按机身朝向算，m）、
    关节跟踪误差 track（目标角 − 实际角的均方根，rad）；traj 给列表时按 fps 存 qpos（渲染用）。"""
    m = m or load_model()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.36
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = Q0
    upper = make_upper(c)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_SHANK_collision") for l in LEGS]   # 小腿碰撞网格包住了脚，着地的是它
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    n_settle, n = int(c.settle / DT), int((c.settle + c.T) / DT)
    log = {k: [] for k in ("t", "x", "y", "z", "roll", "pitch", "yaw", "contact", "foot_x", "foot_y", "track")}
    foot_b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_FOOT") for l in LEGS]
    q_des = Q0.copy()
    every = max(1, int(round(1 / (fps * DT))))
    for k in range(n):
        t = (k - n_settle) * DT
        if k >= n_settle and (k - n_settle) % DECIM == 0:
            q_des = upper(d, t, np.array(c.cmd))
        d.xfrc_applied[torso] = 0
        for t0, fy, dur in c.pushes:
            if t0 <= t < t0 + dur:
                d.xfrc_applied[torso, 1] = fy
        tau = KP * (q_des - d.qpos[7:19]) - KD * d.qvel[6:18]
        d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
        mujoco.mj_step(m, d)
        if k >= n_settle and (k - n_settle) % 10 == 0:              # 日志 100 Hz
            w, x, y, z = d.qpos[3:7]
            log["t"].append(t)
            log["x"].append(d.qpos[0]); log["y"].append(d.qpos[1]); log["z"].append(d.qpos[2])
            log["roll"].append(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            log["pitch"].append(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
            log["yaw"].append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            on = [False] * 4
            for ci in range(d.ncon):
                g = {d.contact[ci].geom1, d.contact[ci].geom2}
                for i, fg in enumerate(feet):
                    if fg in g and floor in g:
                        on[i] = True
            log["contact"].append(on)
            yaw = log["yaw"][-1]
            side = np.array([-np.sin(yaw), np.cos(yaw)])                  # 机身朝向的“左”方向（水平面内）
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            log["foot_x"].append([float((d.xpos[b][:2] - d.qpos[:2]) @ fwd) for b in foot_b])
            log["foot_y"].append([float((d.xpos[b][:2] - d.qpos[:2]) @ side) for b in foot_b])
            log["track"].append(float(np.sqrt(np.mean((q_des - d.qpos[7:19]) ** 2))))
        if traj is not None and k % every == 0:
            traj.append(d.qpos.copy())
    return {k: np.array(v) for k, v in log.items()}


def fell(log):
    tilt = np.hypot(log["roll"], log["pitch"])
    bad = np.nonzero((tilt > np.radians(60)) | (log["z"] < 0.12))[0]
    return float(log["t"][bad[0]]) if len(bad) else None


def report(log):
    t_fall = fell(log)
    end = len(log["t"]) - 1 if t_fall is None else int(np.searchsorted(log["t"], t_fall))
    return {"前进_m": round(float(log["x"][end] - log["x"][0]), 2),
            "侧偏_m": round(float(log["y"][end] - log["y"][0]), 2),
            "偏航_deg": round(float(np.degrees(np.unwrap(log["yaw"])[end] - log["yaw"][0])), 1),   # 展开，转过半圈也不会绕回
            "最大倾角_deg": round(float(np.degrees(np.hypot(log["roll"], log["pitch"])[:end + 1].max())), 1),
            "翻倒时刻_s": t_fall}


PUSHES = [(5.0, 80.0, 0.2), (7.5, -80.0, 0.2)]      # 走到第 5 秒从左边推一下、7.5 秒从右边推一下（80 N 推 0.2 s）

if __name__ == "__main__":
    m = load_model()
    for mode in ("kepi", "trot", "rl"):
        print(mode, "不推:", report(simulate(Case(mode=mode), m)))
        print(mode, "侧推:", report(simulate(Case(mode=mode, pushes=PUSHES), m)))
