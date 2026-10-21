"""一只狗怎么站起来：Lite3 站在会倾斜的板子上，对比四种“控制器”。

    off      关掉电机：力矩为 0
    pd       只有关节 PD：把 12 个关节锁在站姿（底层，500 Hz）
    geom     PD + 手推平衡层：用腿长几何 h = 2L·cosθ 把“板子歪了多少”算出来，调左右/前后腿长（上层，50 Hz）
    learned  PD + 学出来的平衡层：同一个接口，参数用随机搜索（ARS 思路）从仿真回报里学

两层结构：上层（平衡层）只决定“每条腿多长”，底层关节 PD 负责“跟上”。
模型：DeepRoboticsLab/deep_robotics_model 的 Lite3 MJCF（BSD-3-Clause）。参数为教学设置，不代表实机。
运行环境：pip install mujoco（测试版本 3.16）。直接运行本文件 = 跑四种情况并打印指标。
"""
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                  # 在 notebook 格子里运行时没有 __file__
    _HERE = Path.cwd()
# ---------------- 可视化（notebook 动画 / 3D 窗口）----------------
"""notebook 可视化公共件：MuJoCo 画面在 notebook 里放动画、几个画面并排对比、弹出 3D 窗口回放。
发布时由 make_release.py 原样并进每期的仿真文件（release.viz: true），notebook 仍然单文件能跑。

    frames = mj_frames(m, traj, cam=dict(lookat=[0, 0, 0.3], distance=1.3, azimuth=145, elevation=-14))
    show(frames)                                  # notebook 里放动画（能播放、暂停、拖动）
    show([frames_a, frames_b], titles=["只有 PD", "手推"])     # 并排对比，同步播放
    mj_view(m, traj, cam=...)                      # 弹出 MuJoCo 3D 窗口回放（鼠标拖动旋转、滚轮缩放）

traj 是 qpos 列表，或 (qpos, mocap_quat) 的列表（带会动的板子时）。
"""
import numpy as _np


def style_spec(spec):
    """画面和视频一致：渐变天空、淡色地面格子（compile 之前调用）。"""
    import mujoco
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.76, 0.92], rgb2=[0.09, 0.13, 0.23], width=512, height=3072)
    for t in spec.textures:
        if t.name == "checker":
            t.rgb1, t.rgb2 = [0.70, 0.75, 0.82], [0.86, 0.89, 0.93]
    return spec


def style_model(m):
    """机身改成中性灰（compile 之后调用），和蓝天分得开。"""
    dog = (m.geom_bodyid > 0) & (m.geom_group == 2) & (m.geom_matid < 0)
    m.geom_rgba[dog] = [0.58, 0.60, 0.64, 1]
    return m


def _set_cam(cam, spec):
    for k in ("distance", "azimuth", "elevation"):
        if k in spec:
            setattr(cam, k, spec[k])
    if "lookat" in spec:
        cam.lookat[:] = spec["lookat"]
    return cam


def _apply(d, item):
    """traj 的一项写进 MjData：qpos，或 (qpos, 板子姿态四元数)。"""
    if isinstance(item, tuple):
        d.qpos[:] = item[0]
        d.mocap_quat[0] = item[1]
    else:
        d.qpos[:] = item


def mj_frames(m, traj, cam=None, size=(480, 270), every=1, follow=None, overlay=None):
    """离屏渲染 traj（不弹窗口），返回 RGB 帧列表。
    follow=k：镜头跟着机身慢慢走（k 是每帧追上的比例，如 0.08）；overlay(scene, data)：每帧往场景里加东西（虚影、轨迹）。"""
    import mujoco
    w, h = size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, w), max(m.vis.global_.offheight, h)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w, max_geom=4000)
    c = _set_cam(mujoco.MjvCamera(), cam or {})
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0                      # 不画碰撞网格（半透明蓝壳）
    base = _np.array(c.lookat)
    look, frames = None, []
    for i, item in enumerate(traj[::every]):
        _apply(d, item)
        mujoco.mj_forward(m, d)
        if follow:
            tgt = _np.array([d.qpos[0], d.qpos[1], base[2]])
            look = tgt if look is None else look + follow * (tgt - look)
            c.lookat[:] = look
        r.update_scene(d, c, opt)
        if overlay:
            overlay(r.scene, d)
        frames.append(r.render().copy())
    r.close()
    return frames


def caption(frames, text, color=(255, 255, 255)):
    """在每帧左上角写一行字（中文可用）。"""
    from PIL import Image, ImageDraw, ImageFont
    fo, px = None, max(14, frames[0].shape[0] // 14)
    for name in ("msyh.ttc", "simhei.ttf", "PingFang.ttc", "/System/Library/Fonts/PingFang.ttc",
                 "NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "wqy-microhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        try:
            fo = ImageFont.truetype(name, px)          # Windows / macOS / Linux 常见中文字体，找到哪个用哪个
            break
        except OSError:
            continue
    fo = fo or ImageFont.load_default()
    out = []
    for f in frames:
        im = Image.fromarray(f)
        ImageDraw.Draw(im).text((10, 8), text, font=fo, fill=color, stroke_width=3, stroke_fill=(20, 24, 40))
        out.append(_np.asarray(im))
    return out


def show(frames, titles=None, fps=15, width=None):
    """在 notebook 里放动画。frames 可以是一组帧，也可以是几组帧的列表（并排同步播放，短的停在最后一帧）。
    width=None：按帧的原始像素嵌入（文件最小、不糊）；给英寸数则缩放到这个宽度。"""
    import matplotlib.pyplot as plt
    from matplotlib import animation, rcParams
    from IPython.display import HTML
    groups = frames if isinstance(frames[0], list) else [frames]
    n = max(len(g) for g in groups)
    h, w = groups[0][0].shape[:2]
    dpi = 80
    width = width or len(groups) * w / dpi * 1.01
    fig, axes = plt.subplots(1, len(groups), figsize=(width, width / len(groups) * h / w + (0.35 if titles else 0)), dpi=dpi)
    axes = _np.atleast_1d(axes)
    ims = []
    for k, (ax, g) in enumerate(zip(axes, groups)):
        ax.axis("off")
        ims.append(ax.imshow(g[0]))
        if titles:
            ax.set_title(titles[k], fontsize=11)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 if not titles else 1 - 0.3 / fig.get_figheight(), wspace=0.02)

    def step(i):
        for im, g in zip(ims, groups):
            im.set_data(g[min(i, len(g) - 1)])
        return ims
    anim = animation.FuncAnimation(fig, step, frames=n, interval=1000 / fps)
    plt.close(fig)
    rcParams["animation.frame_format"] = "jpeg"           # 比 png 小很多，GitHub 网页上也能直接看
    rcParams["animation.embed_limit"] = 60
    return HTML(anim.to_jshtml(fps=fps, default_mode="loop"))


def mj_view(m, traj, cam=None, fps=30, repeat=3, overlay=None):
    """弹出 MuJoCo 自带的 3D 窗口回放 traj：鼠标左键拖动旋转、右键平移、滚轮缩放；播完 repeat 遍自动关闭。
    macOS 上要用 mjpython 运行；远程服务器、没有显示器的环境请用 show()。"""
    import time
    import mujoco
    import mujoco.viewer
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as v:
        _set_cam(v.cam, cam or {})
        v.opt.geomgroup[1] = 0
        for _ in range(repeat):
            for item in traj:
                if not v.is_running():
                    return
                _apply(d, item)
                mujoco.mj_forward(m, d)
                if overlay:
                    with v.lock():
                        v.user_scn.ngeom = 0
                        overlay(v.user_scn, d)
                v.sync()
                time.sleep(1 / fps)
            time.sleep(1.0)
# 发布包里模型放在同文件夹的 lite3/；找不到时退回作者本地的模型路径
MJCF = next((p for p in [_HERE / "lite3", _HERE.parents[1] / "assets" / "deep_robotics_model-main" / "Lite3" / "mjcf"]
             if (p / "Lite3.xml").exists()), _HERE / "lite3")

# ---------------- 模型 ----------------
L = 0.2051                 # 大腿 = 小腿长度（m），由站姿正运动学反推：0.286 = 2L·cos0.8
H0 = 2 * L * np.cos(0.8)   # 站立腿长 ≈ 0.286 m（髋到足心）
H_MIN, H_MAX = 0.20, 0.37
W = 0.318                  # 左右足距（m）
LX = 0.349                 # 前后足距（m）
LEG_Y = np.array([1, -1, 1, -1])   # FL FR HL HR：左 +1 / 右 -1
LEG_X = np.array([1, 1, -1, -1])   # 前 +1 / 后 -1
DT = 0.002                 # 物理与关节 PD 步长 2 ms（500 Hz）
BAL_EVERY = 10             # 平衡层每 10 步更新一次（50 Hz）
KP, KD, TAU_MAX = 60.0, 1.5, 30.0   # KP=40 时机身有约 1 Hz 的前后晃动，上层一叠加就失控（见翻车记录）
BOARD_Z = 0.30             # 板子转轴高度


LITE3_FILES = ["Lite3.xml", "LICENSE.txt"] + [f"meshes/{n}.STL" for n in ("torso", "hip", "thigh", "shank")]
LITE3_URLS = ["https://gitee.com/nononodoubt/shoutui-daogoupao/raw/main/四足/一只狗怎么站起来/lite3/",
              "https://fastly.jsdelivr.net/gh/NONONODOUBT/shoutui-daogoupao@main/四足/一只狗怎么站起来/lite3/",
              "https://raw.githubusercontent.com/NONONODOUBT/shoutui-daogoupao/main/四足/一只狗怎么站起来/lite3/"]


def download_lite3(dst):
    """只拿到 notebook、旁边没有 lite3/ 时，从配套仓库下载模型（约 3 MB，国内先走 Gitee）。"""
    from urllib.parse import quote
    from urllib.request import urlopen
    for base in LITE3_URLS:
        try:
            for f in LITE3_FILES:
                data = urlopen(quote(base + f, safe=":/@"), timeout=30).read()
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                (dst / f).write_bytes(data)
            print(f"已下载 Lite3 模型到 {dst}")
            return
        except OSError as e:
            print(f"从 {base.split('/')[2]} 下载失败（{e}），换一个地址")
    raise FileNotFoundError(f"找不到 Lite3 模型：请把配套仓库里的 lite3/ 文件夹放到 {dst.parent}")


def load_model():
    """MuJoCo 在 Windows 上打不开中文路径：自己读 XML 和网格，以内容形式传进去。"""
    if not (MJCF / "Lite3.xml").exists():
        download_lite3(MJCF)
    xml = (MJCF / "Lite3.xml").read_text(encoding="utf-8")
    assets = {f"./meshes/{p.name}": p.read_bytes() for p in (MJCF / "meshes").iterdir()}
    spec = mujoco.MjSpec.from_string(xml, assets)
    spec.assets = {k[2:]: v for k, v in assets.items()}     # compile 时按 meshes/xxx 查找
    # 画面：渐变天空（和讲解画布同色系）、淡一点的地面格子
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.76, 0.92], rgb2=[0.09, 0.13, 0.23], width=512, height=3072)
    for t in spec.textures:
        if t.name == "checker":
            t.rgb1, t.rgb2 = [0.70, 0.75, 0.82], [0.86, 0.89, 0.93]
    board = spec.worldbody.add_body(name="board", pos=[0, 0, BOARD_Z], mocap=True)
    board.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.45, 0.02], pos=[0, 0, -0.02],
                   rgba=[0.85, 0.62, 0.38, 1], friction=[1.0, 0.01, 0.01])
    m = spec.compile()
    m.opt.timestep = DT
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080     # 离屏渲染用 1080p
    dog = m.geom_bodyid > 0
    vis = dog & (m.geom_group == 2) & (m.geom_matid < 0)                 # 机身改成中性灰，和蓝天分得开
    m.geom_rgba[vis] = [0.58, 0.60, 0.64, 1]
    return m


def leg_angles(h):
    """腿长 → 关节角（足在髋正下方）：h = 2L·cosθ，髋 = -θ，膝 = 2θ。"""
    th = np.arccos(np.clip(h, H_MIN, H_MAX) / (2 * L))
    return -th, 2 * th


def quat_to_rp(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def rp_to_quat(roll, pitch):
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])


# ---------------- 上层：平衡层 ----------------
# 观测 o = [机身横滚 φ, 机身俯仰 θ, 横滚角速度, 俯仰角速度, 当前左右腿长差 Δy, 当前前后腿长差 Δx]
# 输出 = 新的 (Δy, Δx)；各腿长 h_i = H0 + LEG_Y·Δy/2 + LEG_X·Δx/2

ALPHA = 0.1                # 平衡层每次只往目标走 10%：一步改到位会让底层跟不上，来回振荡直到翻倒


def geom_policy(o, alpha=ALPHA):
    """手推：机身倾角 = 板子倾角 + 腿长差造成的倾角 ⇒ 反推出板子倾角，再算出让机身水平所需的腿长差。"""
    roll, pitch, _, _, dy, dx = o
    board_roll = roll - np.arctan2(dy, W)          # 左腿长 → 机身向右倾（横滚为负）
    board_pitch = pitch + np.arctan2(dx, LX)       # 前腿长 → 机身抬头（俯仰为负）
    ty, tx = -W * np.tan(board_roll), LX * np.tan(board_pitch)
    return dy + alpha * (ty - dy), dx + alpha * (tx - dx)


def linear_policy(w):
    """学习用的线性策略：每 20 ms 把腿长差改一点，改多少 = 参数 · 观测。w 共 4 个数。"""
    def pol(o):
        roll, pitch, wr, wp, dy, dx = o
        return dy + (w[0] * roll + w[1] * wr) * 0.02, dx + (w[2] * pitch + w[3] * wp) * 0.02
    return pol


@dataclass
class Case:
    mode: str = "pd"                    # off / pd / geom / learned
    w: np.ndarray = field(default=None)  # learned 的参数
    T: float = 12.0
    roll_amp: float = np.radians(15)
    pitch_amp: float = np.radians(10)
    tilt_start: float = 1.0
    seed: int = 0
    phase: tuple = (0.0, 0.0)            # 板子摆动相位（训练时随机化）
    period: tuple = (4.0, 5.3)
    board: str = "sine"                  # sine：来回晃；step：突然歪一下就停住
    step_time: float = 0.3               # step 时用多久歪过去（秒）
    imu_noise: float = 0.0               # 倾角测量噪声标准差（rad），角速度噪声取其 10 倍（rad/s）
    delay: int = 0                       # 观测延迟（平衡层拍数，1 拍 = 20 ms）
    policy_start: float = 0.0            # 平衡层从这一刻起才接管（之前只有关节 PD 锁站姿）
    alpha: float = ALPHA                 # 手推平衡层的旋钮：每次往目标走多少


def board_angles(c, t):
    if c.board == "step":                           # 作业 3：tilt_start 起在 step_time 秒内歪到 (roll_amp, pitch_amp)，之后不动
        s = np.clip((t - c.tilt_start) / max(c.step_time, 1e-6), 0, 1)   # 不能一帧瞬移：脚会嵌进板子，冲击力不真实
        return (c.roll_amp * s, c.pitch_amp * s)
    s = np.clip((t - c.tilt_start) / 2.0, 0, 1)    # 2 s 内把幅度加满
    r = c.roll_amp * s * np.sin(2 * np.pi * (t - c.tilt_start) / c.period[0] + c.phase[0])
    p = c.pitch_amp * s * np.sin(2 * np.pi * (t - c.tilt_start) / c.period[1] + c.phase[1])
    return (r, p) if t > c.tilt_start else (0.0, 0.0)


def simulate(c: Case, m=None, traj=None, fps=30):
    """返回时间序列 dict；traj 传入一个 list 时，按 fps 存 (qpos, 板子姿态)，供 p01_render.py 出画面。"""
    m = m or load_model()
    d = mujoco.MjData(m)
    d.qpos[2] = BOARD_Z + H0 + 0.035
    d.qpos[3] = 1
    hy, kn = leg_angles(H0)
    for i in range(4):
        d.qpos[7 + 3 * i: 10 + 3 * i] = [0, hy, kn]
    mujoco.mj_forward(m, d)
    policy = (lambda o: geom_policy(o, c.alpha)) if c.mode == "geom" else linear_policy(c.w) if c.mode == "learned" else None
    dy = dx = 0.0
    rng = np.random.default_rng(c.seed)
    obs_q = []                           # 观测队列：模拟传感器和通信的延迟
    n = int(c.T / DT)
    log = {k: np.zeros(n) for k in ["t", "roll", "pitch", "board_roll", "board_pitch", "height", "dy", "dx", "tau_max"]}
    for k in range(n):
        t = k * DT
        br, bp = board_angles(c, t)
        d.mocap_quat[0] = rp_to_quat(br, bp)
        roll, pitch = quat_to_rp(d.qpos[3:7])
        if policy and t >= c.policy_start and k % BAL_EVERY == 0:
            meas = np.array([roll, pitch, d.qvel[3], d.qvel[4]])
            meas += rng.standard_normal(4) * c.imu_noise * np.array([1, 1, 10, 10])
            obs_q.append(meas)
            meas = obs_q.pop(0) if len(obs_q) > c.delay else obs_q[0]
            dy, dx = policy(np.concatenate([meas, [dy, dx]]))
            dy = float(np.clip(dy, -(H_MAX - H_MIN), H_MAX - H_MIN))
            dx = float(np.clip(dx, -(H_MAX - H_MIN), H_MAX - H_MIN))
        # 底层：关节 PD 跟踪各腿目标
        tau = np.zeros(12)
        if c.mode != "off":
            for i in range(4):
                h = H0 + LEG_Y[i] * dy / 2 + LEG_X[i] * dx / 2
                a_hy, a_kn = leg_angles(h)
                q_ref = np.array([0.0, a_hy, a_kn])
                q, qd = d.qpos[7 + 3 * i: 10 + 3 * i], d.qvel[6 + 3 * i: 9 + 3 * i]
                tau[3 * i: 3 * i + 3] = KP * (q_ref - q) - KD * qd
        d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
        mujoco.mj_step(m, d)
        for key, v in zip(log, [t, roll, pitch, br, bp, d.qpos[2] - BOARD_Z, dy, dx, np.abs(d.ctrl).max()]):
            log[key][k] = v
        if traj is not None and int(t * fps) >= len(traj):
            traj.append((d.qpos.copy(), d.mocap_quat[0].copy()))
    return log


def score(log, t0=3.0):
    """机身倾角（相对水平）的均方根，单位度；以及最后机身高度（m），太低算趴下。"""
    sel = log["t"] >= min(t0, log["t"][-1] / 2)
    tilt = np.degrees(np.sqrt(log["roll"][sel] ** 2 + log["pitch"][sel] ** 2))
    return {"倾角RMS_deg": float(np.sqrt(np.mean(tilt ** 2))), "最大倾角_deg": float(tilt.max()),
            "最后高度_m": float(log["height"][-1]), "趴下": bool(log["height"][-1] < 0.15)}


# ---------------- 学习：随机搜索（ARS 思路）----------------
def train_cases(rng, n, randomize=False):
    """训练用的板子：幅度、周期、相位都随机，避免策略只记住一条固定轨迹。
    randomize=True 时再随机加传感器噪声（0–1°）和延迟（0–40 ms）：域随机化。"""
    return [Case(mode="learned", T=6.0, roll_amp=np.radians(rng.uniform(3, 15)), pitch_amp=np.radians(rng.uniform(3, 12)),
                 phase=tuple(rng.uniform(0, 2 * np.pi, 2)), period=tuple(rng.uniform(3, 6, 2)),
                 imu_noise=np.radians(rng.uniform(0, 1)) if randomize else 0.0,
                 delay=int(rng.integers(0, 3)) if randomize else 0,
                 seed=int(rng.integers(1 << 30)) if randomize else 0)    # 不随机化时不多抽随机数，保证结果可复现
            for _ in range(n)]


def rollout_return(w, cases, m):
    """回报 = −(机身倾角的均方，单位 度²)；趴下或掉下板子再扣 1000。训练器只看到这个数。"""
    total = 0.0
    for c in cases:
        c.w = w
        s = score(simulate(c, m), t0=1.5)
        total += -s["倾角RMS_deg"] ** 2 - (1000 if s["趴下"] else 0)
    return total / len(cases)


def train(iters=40, n_dir=8, top=4, nu=0.5, step=0.4, seed=0, randomize=False, m=None, log=print):
    """从 w = 0 出发。每轮：随机挑 n_dir 个方向，各试 w+νδ 和 w−νδ，用回报差决定往哪边走。
    不用板子角度、不用手推公式、不对物理求导。返回每轮的参数与回报。"""
    m = m or load_model()
    rng = np.random.default_rng(seed)
    w = np.zeros(4)
    hist = []
    for it in range(iters):
        cases = train_cases(rng, 2, randomize)
        deltas = rng.standard_normal((n_dir, 4))
        rp = np.array([rollout_return(w + nu * dl, cases, m) for dl in deltas])
        rn = np.array([rollout_return(w - nu * dl, cases, m) for dl in deltas])
        best = np.argsort(-np.maximum(rp, rn))[:top]
        sig = np.concatenate([rp[best], rn[best]]).std()
        if sig > 1e-8:
            w = w + step / (top * sig) * ((rp[best] - rn[best])[:, None] * deltas[best]).sum(0)
        ret = rollout_return(w, train_cases(np.random.default_rng(999), 4, randomize), m)   # 固定的 4 块板子做验证
        hist.append((it, w.copy(), ret))
        log(f"第 {it:2d} 轮  w = {np.round(w, 2)}  验证回报 {ret:9.1f}")
    return w, hist


# ---------------- 看画面 ----------------
CAM = dict(lookat=[0, 0, BOARD_Z + 0.12], distance=1.3, azimuth=145, elevation=-14)


def movie(c: Case, m=None, title=None, size=(400, 225), fps=10):
    """跑一遍 c，渲染成帧（不弹窗口）；title 写在左上角，最后加上“平均歪多少”。"""
    m = m or load_model()
    traj = []
    s = score(simulate(c, m, traj=traj, fps=fps))
    frames = mj_frames(m, traj, cam=CAM, size=size, follow=0.3)
    tag = "翻了！" if s["趴下"] else f"平均歪 {s['倾角RMS_deg']:.1f}°"
    return caption(frames, f"{title}　{tag}" if title else tag)


def show_cases(cases, m=None):
    """几种情况并排同步播放：cases = {"只有关节 PD": Case(mode="pd"), "手推平衡层": Case(mode="geom")}。"""
    m = m or load_model()
    return show([movie(c, m, name) for name, c in cases.items()], fps=10)


def view_case(c: Case):
    """弹出 MuJoCo 的 3D 窗口看一种情况（鼠标拖动旋转、滚轮缩放）。macOS 用 mjpython 运行。"""
    m = load_model()
    traj = []
    simulate(c, m, traj=traj, fps=30)
    mj_view(m, traj, cam=CAM, fps=30)


if __name__ == "__main__":
    import sys
    if "--view" in sys.argv:              # python balance_sim.py --view：弹出 3D 窗口看手推平衡层
        view_case(Case(mode="geom"))
        sys.exit()
    m = load_model()
    for mode in ["off", "pd", "geom"]:
        print(mode, score(simulate(Case(mode=mode), m)))
