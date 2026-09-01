"""Generate a tiny synthetic WESAD-shaped dataset for smoke-testing the pipeline.
Not for results. Baseline: HR ~65, low EDA. Stress: HR ~90, higher EDA with SCRs."""
import pickle, numpy as np
from pathlib import Path
rng = np.random.default_rng(0)
def seg(cond, secs):
    hr = 65 if cond != 2 else 90
    t64 = np.arange(secs*64)/64
    bvp = np.sin(2*np.pi*(hr/60)*t64 + rng.normal(0,0.05,len(t64)).cumsum()*0.1) + rng.normal(0,0.2,len(t64))
    t4 = np.arange(secs*4)/4
    eda = (2.0 if cond!=2 else 4.0) + 0.002*t4 + rng.normal(0,0.02,len(t4))
    if cond == 2:
        for s in rng.integers(0, secs*4-20, size=secs//15):
            eda[s:s+20] += 0.3*np.exp(-np.arange(20)/6)
    temp = 33 + rng.normal(0,0.05,len(t4))
    acc = np.c_[rng.normal(0,0.02,secs*32), rng.normal(0,0.02,secs*32), 1+rng.normal(0,0.02,secs*32)]*64
    lab = np.full(secs*700, cond, dtype=np.int16)
    return bvp, eda, temp, acc, lab
def subject(sid, root):
    parts = [seg(1, 300), seg(2, 300), seg(3, 200)]
    d = {"signal": {"wrist": {"BVP": np.concatenate([p[0] for p in parts])[:,None],
                               "EDA": np.concatenate([p[1] for p in parts])[:,None],
                               "TEMP": np.concatenate([p[2] for p in parts])[:,None],
                               "ACC": np.concatenate([p[3] for p in parts])}},
         "label": np.concatenate([p[4] for p in parts]), "subject": sid}
    (Path(root)/sid).mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(Path(root)/sid/f"{sid}.pkl","wb"))
if __name__ == "__main__":
    import sys; root = sys.argv[1]
    for s in [f"S{i}" for i in range(2,18) if i!=12]: subject(s, root)
    print("fake WESAD written to", root)
