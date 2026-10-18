"""把官方 Lite3 策略（DeepRoboticsLab/Lite3_rl_deploy 的 policy/ppo/policy.onnx，BSD-3-Clause）的权重读成 numpy。
不依赖 onnx 包：只解析 protobuf 里 graph.initializer 的几个字段（dims、data_type、name、raw_data / float_data）。
网络结构：45 → 512 → 256 → 128 → 12，ELU 激活（rsl_rl 的 actor）。
运行：python onnx_to_npz.py policy.onnx lite3_policy.npz
"""
import struct
import sys

import numpy as np


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def fields(b):
    """逐个产出 (字段号, 线型, 值)；长度分隔字段的值是 bytes。"""
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        elif wt == 2:
            n, i = varint(b, i)
            v, i = b[i:i + n], i + n
        else:
            raise ValueError(f"不支持的线型 {wt}")
        yield num, wt, v


def tensor(b):
    dims, name, raw, floats = [], "", None, []
    for num, wt, v in fields(b):
        if num == 1:
            if wt == 2:                       # packed
                j = 0
                while j < len(v):
                    d, j = varint(v, j)
                    dims.append(d)
            else:
                dims.append(v)
        elif num == 2 and v != 1:
            raise ValueError("只支持 float32")
        elif num == 8:
            name = v.decode()
        elif num == 9:
            raw = v
        elif num == 4:
            floats.extend(struct.unpack(f"<{len(v) // 4}f", v) if wt == 2 else [struct.unpack("<f", v)[0]])
    a = np.frombuffer(raw, "<f4") if raw is not None else np.array(floats, np.float32)
    return name, a.reshape(dims)


def load_onnx(path):
    b = open(path, "rb").read()
    graph = next(v for num, wt, v in fields(b) if num == 7)
    return dict(tensor(v) for num, wt, v in fields(graph) if num == 5)


if __name__ == "__main__":
    w = load_onnx(sys.argv[1])
    for k, v in w.items():
        print(k, v.shape)
    np.savez(sys.argv[2], **{k.replace(".", "_"): v for k, v in w.items()})
