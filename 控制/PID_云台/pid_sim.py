"""仿真：RM 云台俯仰轴（单个转动轴 + 枪管重力矩），比较 P / PI / PD / PID。

对象：
    J dv/dt + b v = Kt u - tau_load,   dq/dt = v
    J=0.01 kg·m², b=0.02 N·m·s/rad, Kt=0.5 N·m/单位命令, u ∈ [-1, 1]
控制器周期 Ts=1 ms，零阶保持；对象用 0.1 ms 子步 RK4 推进。
tau_load 是本样片新增的枪管重力矩，属于教学假设：
重力矩 m g d cos q 在俯仰角变化不大时近似为常数。参数是教学用的数，不是某台真实云台的参数。
D 项对测量微分（避免目标阶跃造成微分冲击）。
I 项用条件积分：只暂停把输出继续推向所触上下限的那部分积分（A07 §6）。
"""
from dataclasses import dataclass, field
import numpy as np

J, B, KT = 0.01, 0.02, 0.5
TS = 1e-3
SUBSTEPS = 10
U_MAX = 1.0


@dataclass
class Case:
    name: str
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    tau_f: float = 0.0          # 微分低通时间常数，0 表示不滤波
    noise_sigma: float = 0.0    # 位置测量噪声标准差 rad
    seed: int = 20261015
    r: float = 0.2              # 目标角 rad
    t_step: float = 0.1         # 目标阶跃时刻
    tau_load: float = 0.0       # 负载 N·m
    t_load: float = 1.5         # 负载加上的时刻
    t_end: float = 4.0
    extra: dict = field(default_factory=dict)


def _deriv(q, v, u, tl):
    return v, (KT * u - B * v - tl) / J


def simulate(c: Case):
    n = int(round(c.t_end / TS))
    rng = np.random.default_rng(c.seed)
    t = np.arange(n) * TS
    q_log, u_log, ref_log, meas_log, i_log, d_log = (np.zeros(n) for _ in range(6))
    q = v = 0.0
    integ = 0.0
    prev_meas = None
    d_filt = 0.0
    h = TS / SUBSTEPS
    for k in range(n):
        tk = t[k]
        r = c.r if tk >= c.t_step else 0.0
        tl = c.tau_load if tk >= c.t_load else 0.0
        meas = q + (rng.normal(0, c.noise_sigma) if c.noise_sigma > 0 else 0.0)
        e = r - meas
        # D：对测量微分，取负号
        raw_rate = 0.0 if prev_meas is None else (meas - prev_meas) / TS
        if c.tau_f > 0:
            a = c.tau_f / (c.tau_f + TS)
            d_filt = a * d_filt + (1 - a) * raw_rate
            rate = d_filt
        else:
            rate = raw_rate
        prev_meas = meas
        p_term = c.kp * e
        d_term = -c.kd * rate
        raw = p_term + integ + d_term
        u = min(max(raw, -U_MAX), U_MAX)
        # 条件积分：本拍算完输出后再更新 I
        inc = c.ki * e * TS
        pushes_high = raw >= U_MAX and inc > 0
        pushes_low = raw <= -U_MAX and inc < 0
        if not (pushes_high or pushes_low):
            integ += inc
        q_log[k], u_log[k], ref_log[k], meas_log[k] = q, u, r, meas
        i_log[k], d_log[k] = integ, d_term
        # 对象推进一个控制周期（零阶保持）
        for _ in range(SUBSTEPS):
            k1 = _deriv(q, v, u, tl)
            k2 = _deriv(q + h / 2 * k1[0], v + h / 2 * k1[1], u, tl)
            k3 = _deriv(q + h / 2 * k2[0], v + h / 2 * k2[1], u, tl)
            k4 = _deriv(q + h * k3[0], v + h * k3[1], u, tl)
            q += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return {"t": t, "q": q_log, "u": u_log, "r": ref_log, "meas": meas_log, "I": i_log, "D": d_log}


def overshoot_pct(res, c: Case, t_until=None):
    t_until = t_until if t_until is not None else c.t_load
    m = (res["t"] >= c.t_step) & (res["t"] < t_until)
    return (res["q"][m].max() - c.r) / c.r * 100


def steady_error(res, c: Case, window=0.3):
    m = res["t"] >= c.t_end - window
    return float(np.mean(c.r - res["q"][m]))


def zeta_pd(kp, kd):
    """连续时间、忽略采样/滤波/噪声/限幅时的阻尼比。"""
    return (B + KT * kd) / (2 * np.sqrt(J * KT * kp))


def omega_n(kp):
    return np.sqrt(KT * kp / J)


def overshoot_formula(z):
    return 100 * np.exp(-np.pi * z / np.sqrt(1 - z * z)) if z < 1 else 0.0


# ---------------- 看画面：云台动画 ----------------
def show_gimbal(cases, t_max=4.0, fps=12, speed=0.5):
    """几种控制器的云台并排做成动画（notebook 里播放）：上面是枪管实际转到哪、虚线是目标；
    装枪管那一刻出现红色箭头（重力矩）；下面是 q(t) 曲线跟着画出来。speed=0.5：半速慢放。
    cases = [("只用 P", Case("P", kp=2, tau_load=0.05)), ...]"""
    import matplotlib.pyplot as plt
    from matplotlib import animation, rcParams
    from IPython.display import HTML
    runs = [(lab, c, simulate(c)) for lab, c in cases]
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 4.8), gridspec_kw={"height_ratios": [1.25, 1]}, squeeze=False, dpi=70)
    arts = []
    for j, (lab, c, r) in enumerate(runs):
        a = axes[0, j]
        a.set_aspect("equal"); a.set_xlim(-0.45, 1.25); a.set_ylim(-0.5, 0.75); a.axis("off"); a.set_title(lab, fontsize=12)
        a.add_patch(plt.Rectangle((-0.22, -0.48), 0.44, 0.3, color="#5a6070"))
        a.plot([0, 0], [-0.18, 0], color="#5a6070", lw=8)
        a.plot([0, 1.15 * np.cos(c.r)], [0, 1.15 * np.sin(c.r)], "k--", lw=1)
        a.text(1.15 * np.cos(c.r), 1.15 * np.sin(c.r) + 0.04, "目标", fontsize=9, ha="center")
        barrel, = a.plot([], [], lw=11, color="tab:orange", solid_capstyle="round")
        a.plot(0, 0, "o", color="#222", ms=11)
        load = a.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2.5))
        lt = a.text(0, 0, "负载", color="tab:red", fontsize=9)
        txt = a.text(-0.42, 0.66, "", fontsize=9, family="monospace")
        b = axes[1, j]
        sel = r["t"] <= t_max
        b.plot(r["t"][sel], r["r"][sel], "k--", lw=1)
        line, = b.plot([], [], color="tab:blue", lw=2)
        if c.tau_load > 0 and c.t_load < t_max:
            b.axvline(c.t_load, color="tab:red", lw=0.8, ls=":")
        b.set_xlim(0, t_max); b.set_ylim(-0.02, max(0.3, r["q"][sel].max() * 1.12))
        b.set_xlabel("t / s"); b.set_ylabel("q / rad")
        arts.append((c, r, barrel, load, lt, txt, line))
    fig.tight_layout(pad=0.4)

    def step(i):
        t = i * speed / fps
        out = []
        for c, r, barrel, load, lt, txt, line in arts:
            k = min(int(t / TS), len(r["t"]) - 1)
            q = r["q"][k]
            barrel.set_data([-0.25 * np.cos(q), 1.0 * np.cos(q)], [-0.25 * np.sin(q), 1.0 * np.sin(q)])
            on = c.tau_load > 0 and t >= c.t_load
            tip = (0.85 * np.cos(q), 0.85 * np.sin(q))
            load.xyann, load.xy = tip, (tip[0], tip[1] - 0.32)
            lt.set_position((tip[0] + 0.05, tip[1] - 0.3))
            load.set_visible(on); lt.set_visible(on)
            txt.set_text(f"t = {t:4.2f} s   q = {np.degrees(q):5.1f}°")
            line.set_data(r["t"][:k + 1], r["q"][:k + 1])
            out += [barrel, load, lt, txt, line]
        return out
    anim = animation.FuncAnimation(fig, step, frames=int(t_max / speed * fps) + 1, interval=1000 / fps)
    plt.close(fig)
    rcParams["animation.frame_format"] = "jpeg"
    rcParams["animation.embed_limit"] = 60
    return HTML(anim.to_jshtml(fps=fps, default_mode="loop"))


if __name__ == "__main__":
    # 直接运行本文件时的演示：打印本期的几个关键数，并画出 P / PI / PID 三条曲线。
    # 平时是被 notebook 用 from pid_sim import * 调用的。
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cases = [Case("只用 P（Kp=2）", kp=2, tau_load=0.05, t_end=10),
             Case("PI（Kp=2, Ki=5，没有 D）", kp=2, ki=5, tau_load=0.05, t_end=6),
             Case("PID（Kp=2, Ki=5, Kd=0.3）", kp=2, ki=5, kd=0.3, tau_load=0.05, t_end=6)]
    print("稳态误差（只用 P）：", round(steady_error(simulate(cases[0]), cases[0]) * 1000, 2), "mrad，公式 τL/(Kt·Kp) =",
          round(0.05 / (KT * 2) * 1000, 2), "mrad")
    print("阻尼比：Kd=0 时", round(zeta_pd(2, 0), 3), "；Kd=0.3 时", round(zeta_pd(2, 0.3), 3))
    print("没有 D 时 Ki 的上限：", round(B * 2 / J, 2), "；Kd=0.3 时：", round((B + KT * 0.3) * 2 / J, 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in cases:
        r = simulate(c)
        ax.plot(r["t"], r["q"], label=c.name)
    ax.axhline(0.2, color="k", ls="--", lw=1, label="目标")
    ax.axvline(1.5, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.1, 0.5)
    ax.set_xlabel("时间 t / s")
    ax.set_ylabel("俯仰角 q / rad")
    ax.set_title("云台俯仰轴：只用 P / 加积分 / 再加微分（1.5 s 装上枪管）")
    ax.legend(loc="upper right")
    plt.show()
