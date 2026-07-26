# -*- coding: utf-8 -*-
# grid.py
'''
import random
import numpy as np

# ====== 6 bộ bạn đã chạy ======
done = {
    "1.00e+00 8.68e-02 1.00e-02",
    "1.16e+00 0.00e+00 0.00e+00",
    "1.26e+00 0.00e+00 0.00e+00",
    "1.79e+00 0.00e+00 0.00e+00",
    "9.00e-01 1.50e-01 1.00e-02",
    "9.00e-01 8.68e-02 1.00e-02",
}

lines = []

for tau in [0.9] + list(np.linspace(1.0, 2.0, 20)):
    if tau <= 1.0:
        eps = 1e-2
        for b in np.linspace(0, 15, 20):
            lam = b * eps
            s = f"{tau:.2e} {lam:.2e} {eps:.2e}"
            if s not in done:
                lines.append(s)
    else:
        s = f"{tau:.2e} 0.00e+00 0.00e+00"
        if s not in done:
            lines.append(s)

# Shuffle để chọn ngẫu nhiên
random.seed(42)
random.shuffle(lines)

# Chỉ lấy 4 bộ mới
new_jobs = lines[:4] + [
    "1.00e+00 8.68e-02 1.00e-02",
    "1.16e+00 0.00e+00 0.00e+00",
    "1.26e+00 0.00e+00 0.00e+00",
    "1.79e+00 0.00e+00 0.00e+00",
    "9.00e-01 1.50e-01 1.00e-02",
    "9.00e-01 8.68e-02 1.00e-02",
]

# In ra 4 bộ cần chạy
print("\n".join(new_jobs))
'''

# -*- coding: utf-8 -*-
# grid.py

# ====== CHỈ CHẠY ĐÚNG 10 BỘ SAU ======
jobs = [
    "1.00e+00 8.68e-02 1.00e-02",
    "1.16e+00 0.00e+00 0.00e+00",
    "1.26e+00 0.00e+00 0.00e+00",
    # "1.79e+00 0.00e+00 0.00e+00",
    "9.00e-01 1.50e-01 1.00e-02",
    "9.00e-01 8.68e-02 1.00e-02",

    # 4 bộ mới
    "9.00e-01 7.11e-02 1.00e-02",
    "1.00e+00 3.95e-02 1.00e-02",
    "1.00e+00 5.53e-02 1.00e-02",
    "9.00e-01 2.37e-02 1.00e-02",
]

print("\n".join(jobs))

