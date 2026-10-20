"""环境体检：按“验收顺序”一层层检查，哪一层出问题就停在哪一层说清楚。

    1 哪个 Python    现在跑代码的是哪个解释器、哪个 conda 环境；pip 和它是不是同一个；Jupyter 内核指向哪
    2 包版本         numpy / matplotlib / mujoco 装没装、是不是 environment.yml 里锁的版本
    3 缓存在哪       pip 缓存、conda 包缓存和环境目录、模型缓存（HF_HOME / TORCH_HOME）在哪个盘、那个盘还剩多少
    4 走一步         MuJoCo 建一个最小模型，mj_step 一次
    5 跑一段         一个方块自由落体 1 秒，看结果是否符合物理
    6 Lite3 站起来   读入 Lite3 模型，关节 PD 锁站姿 3 秒，看机身高度

运行：python p03_check.py（或在 notebook 里 from p03_check import *; checkup()）。
模型：DeepRoboticsLab/deep_robotics_model 的 Lite3 MJCF（BSD-3-Clause），缺的话自动从配套仓库下载。
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                  # 在 notebook 格子里运行时没有 __file__
    _HERE = Path.cwd()
_UP2 = _HERE.parents[1] if len(_HERE.parents) > 1 else _HERE      # 在盘根目录附近运行时没有两级上级目录
# 模型放在同文件夹的 lite3/；找不到时自动下载
MJCF = next((p for p in [_HERE / "lite3", _UP2 / "assets" / "deep_robotics_model-main" / "Lite3" / "mjcf"]
             if (p / "Lite3.xml").exists()), _HERE / "lite3")

WANT = {"numpy": "2.4", "matplotlib": "3.11", "mujoco": "3.16"}      # 和 environment.yml 一致（只比前两段）
LOW_GB = 20                                                          # 系统盘剩余低于这个数就提醒

REPO = ["https://gitee.com/nononodoubt/shoutui-daogoupao/raw/main/四足/环境搭建/",
        "https://fastly.jsdelivr.net/gh/NONONODOUBT/shoutui-daogoupao@main/四足/环境搭建/",
        "https://raw.githubusercontent.com/NONONODOUBT/shoutui-daogoupao/main/四足/环境搭建/"]
LITE3_FILES = ["lite3/Lite3.xml", "lite3/LICENSE.txt"] + [f"lite3/meshes/{n}.STL" for n in ("torso", "hip", "thigh", "shank")]

OK, WARN, BAD = "✓", "⚠", "✗"


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _drive(path):
    """路径所在的盘（Windows 是 C:，Linux/macOS 是挂载点）和剩余空间（GB）。"""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free / 1e9
    return (p.drive or p.anchor or "/"), free


def _system_drive():
    return os.environ.get("SystemDrive", "/") if os.name == "nt" else "/"


# ---------------- 1 哪个 Python ----------------
def which_python():
    rows = []
    env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    in_base = (Path(sys.prefix) / "condabin").exists()        # 只有 base 里有 condabin（conda 本体装在 base）
    rows.append(("解释器", sys.executable, OK))
    rows.append(("conda 环境", "base（别往这里装课程的包）" if in_base else env, WARN if in_base else OK))
    if "ipykernel" in sys.modules:
        rows.append(("Jupyter 内核", f"这个 notebook 的内核 = 上面那个解释器", OK))
    # 终端里敲 pip 时用的是哪个：和当前解释器不是同一个，就会出现“装了还说找不到”
    pip_here = _run([sys.executable, "-m", "pip", "--version"])
    pip_path = shutil.which("pip")
    pip_cli = _run([pip_path, "--version"]) if pip_path else ""
    here_site = pip_here.split(" from ")[-1].split(" (")[0] if " from " in pip_here else ""
    cli_site = pip_cli.split(" from ")[-1].split(" (")[0] if " from " in pip_cli else ""
    if not pip_cli:
        rows.append(("命令行的 pip", "PATH 里找不到 pip：用 python -m pip 装包", WARN))
    elif Path(here_site).parents[1] == Path(cli_site).parents[1]:
        rows.append(("命令行的 pip", "和当前解释器是同一个", OK))
    else:
        rows.append(("命令行的 pip", f"装到别处去了：{cli_site}（用 python -m pip 装包）", BAD))
    return rows


# ---------------- 2 包版本 ----------------
def packages():
    from importlib.metadata import PackageNotFoundError, version
    rows = []
    for pkg, want in WANT.items():
        try:
            v = version(pkg)
            good = v.split(".")[:2] == want.split(".")
            rows.append((pkg, v + ("" if good else f"（课程锁的是 {want}）"), OK if good else WARN))
        except PackageNotFoundError:
            rows.append((pkg, f"没装进这个环境：{sys.executable} -m pip install {pkg}", BAD))
    return rows


# ---------------- 3 缓存在哪 ----------------
def cache_dirs():
    home = Path.home()
    out = {}
    out["pip 缓存"] = _run([sys.executable, "-m", "pip", "cache", "dir"]) or os.environ.get("PIP_CACHE_DIR", "")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:                                    # 没激活环境直接跑解释器时，PATH 里没有 conda：到 base 里找
        base = Path(os.environ.get("CONDA_ROOT", "")) if os.environ.get("CONDA_ROOT") else None
        cands = [Path(sys.prefix) / "Scripts" / "conda.exe", Path(sys.prefix) / "bin" / "conda",
                 Path(sys.prefix).parents[1] / "Scripts" / "conda.exe", Path(sys.prefix).parents[1] / "bin" / "conda"]
        if base:
            cands.insert(0, base / "Scripts" / "conda.exe")
        conda = next((str(c) for c in cands if c.exists()), None)
    if conda:
        try:
            cfg = json.loads(_run([conda, "config", "--show", "pkgs_dirs", "envs_dirs", "--json"]) or "{}")
            if cfg.get("pkgs_dirs"):
                out["conda 包缓存"] = cfg["pkgs_dirs"][0]
            if cfg.get("envs_dirs"):
                out["conda 环境目录"] = cfg["envs_dirs"][0]
        except json.JSONDecodeError:
            pass
    out["HF_HOME（模型下载）"] = os.environ.get("HF_HOME", str(home / ".cache" / "huggingface"))
    out["TORCH_HOME（模型下载）"] = os.environ.get("TORCH_HOME", str(home / ".cache" / "torch"))
    return out


def caches():
    rows = []
    sysd = _system_drive()
    for name, path in cache_dirs().items():
        if not path:
            continue
        drv, free = _drive(path)
        on_sys = drv.rstrip("\\/").upper() == sysd.rstrip("\\/").upper()
        msg = f"{path}（{drv} 盘剩 {free:.0f} GB）"
        rows.append((name, msg, WARN if on_sys and free < LOW_GB else OK))
    return rows


# ---------------- 4–6 MuJoCo ----------------
def mujoco_step():
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    return [("MuJoCo 走一步", f"mj_step 一次，t = {d.time * 1000:.0f} ms", OK)]


def mujoco_rollout():
    """方块从 1 m 高自由落体 0.4 s：z 应该 ≈ 1 − ½·g·t² = 0.215 m（没碰地）。"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><option timestep='0.001'/><worldbody><body pos='0 0 1'><freejoint/><geom type='box' size='0.05 0.05 0.05'/>"
        "</body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    while d.time < 0.4 - 1e-9:
        mujoco.mj_step(m, d)
    z, want = d.qpos[2], 1 - 0.5 * 9.81 * 0.4 ** 2
    good = abs(z - want) < 0.01
    return [("跑一段（自由落体 0.4 s）", f"z = {z:.3f} m，理论 {want:.3f} m", OK if good else BAD)]


def download(files, dst):
    """缺模型时从配套仓库下载（国内先走 Gitee，再走 jsDelivr）。"""
    from urllib.parse import quote
    from urllib.request import urlopen
    for base in REPO:
        try:
            for f in files:
                data = urlopen(quote(base + f, safe=":/@"), timeout=30).read()
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                (dst / f).write_bytes(data)
            print(f"已下载 Lite3 模型到 {dst / 'lite3'}")
            return
        except OSError as e:
            print(f"从 {base.split('/')[2]} 下载失败（{e}），换一个地址")
    raise FileNotFoundError(f"缺少 Lite3 模型：请从配套仓库下载 lite3/ 文件夹，放到 {dst}")


def load_lite3():
    """MuJoCo 在 Windows 上打不开中文路径：自己读 XML 和网格，以内容形式传进去。"""
    import mujoco
    if not (MJCF / "Lite3.xml").exists():
        download(LITE3_FILES, MJCF.parent)
    xml = (MJCF / "Lite3.xml").read_text(encoding="utf-8")
    assets = {f"./meshes/{p.name}": p.read_bytes() for p in (MJCF / "meshes").iterdir()}
    spec = mujoco.MjSpec.from_string(xml, assets)
    spec.assets = {k[2:]: v for k, v in assets.items()}
    m = spec.compile()
    m.opt.timestep = 0.001
    return m


Q0 = [0.0, -0.65, 1.30] * 4          # 站姿：每条腿 [髋侧摆, 髋前后, 膝]（和官方部署代码一致）


KP, KD = 60.0, 1.5                   # 只靠关节 PD 锁站姿要硬一点：Kp=30（官方 RL 部署值）时狗会慢慢往后坐到 −20°，那是靠策略一直在调


def lite3_stand(T=3.0, traj=None, fps=30):
    """关节 PD 把 12 个关节锁在站姿，从 0.36 m 高放下，站 T 秒；最后 1 秒机身够高、不歪才算站住。"""
    import mujoco
    import numpy as np
    m = load_lite3()
    d = mujoco.MjData(m)
    q0 = np.array(Q0)
    d.qpos[2], d.qpos[3:7], d.qpos[7:19] = 0.36, [1, 0, 0, 0], q0
    every = int(round(1 / (fps * m.opt.timestep)))
    zs = []
    for k in range(int(T / m.opt.timestep)):
        d.ctrl[:] = np.clip(KP * (q0 - d.qpos[7:19]) - KD * d.qvel[6:18], -30, 30)
        mujoco.mj_step(m, d)
        w, x, y, z = d.qpos[3:7]
        zs.append((d.qpos[2], np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))))
        if traj is not None and k % every == 0:
            traj.append(d.qpos.copy())
    last = np.array(zs[-1000:])
    z, pitch = float(last[:, 0].mean()), float(np.abs(last[:, 1]).max())
    good = z > 0.30 and pitch < 5
    return [("Lite3 站起来", f"站 {T:.0f} 秒后机身高 {z:.3f} m，前后歪 {pitch:.1f}°" + ("" if good else "（没站住）"),
             OK if good else BAD)]


LAYERS = [("1 哪个 Python", which_python), ("2 包版本", packages), ("3 缓存在哪", caches),
          ("4 走一步", mujoco_step), ("5 跑一段", mujoco_rollout), ("6 Lite3 站起来", lite3_stand)]


def checkup(show=True):
    """按顺序体检；某一层出现 ✗ 就停下（下面几层的结果没有意义）。返回 [(层, 项目, 结果, 状态)]。"""
    out = []
    for layer, fn in LAYERS:
        try:
            rows = fn()
        except Exception as e:                       # noqa: BLE001  体检要把任何报错都翻译成人话
            rows = [(layer, f"{type(e).__name__}: {e}", BAD)]
        for name, msg, st in rows:
            out.append((layer, name, msg, st))
            if show:
                print(f"{st} [{layer}] {name}：{msg}")
        if any(st == BAD for _, _, st in rows):
            if show:
                print(f"\n停在第 {layer} 层：先把上面的 ✗ 修好，再往下查。")
            break
    else:
        if show:
            print("\n六层全部通过：可以开始训练了。" if all(r[3] == OK for r in out) else "\n能跑，但有 ⚠ 提醒，建议处理。")
    return out


if __name__ == "__main__":
    checkup()
