"""
test_sampler.py -- verifies the sampling-control surface WITHOUT network or torch
CUDA. It fakes a StreamingMDCATH catalog (self.idx) and checks group_indices_by()
and PerSequenceBatchSampler behavior. Needs numpy (+ torch for Sampler base).
"""
import numpy as np
import mdcath_streaming as m

fail = 0
def check(c, msg):
    global fail
    print(("ok:   " if c else "FAIL: ") + msg)
    if not c: fail += 1


class FakeDS(m.StreamingMDCATH):
    # Bypass __init__/network: build a synthetic flat catalog directly.
    def __init__(self, idx):
        self.idx = idx
        self.skip_frames = 1
    # group_indices_by / build_catalog are inherited and only touch self.idx.


def main():
    # catalog entries: (pdb, file, temp, replica, conf_idx)
    idx = []
    # domain A: temp 348 x 10 frames, temp 320 x 4 frames
    idx += [("A", "A.h5", "348", "0", i) for i in range(10)]
    idx += [("A", "A.h5", "320", "0", i) for i in range(4)]
    # domain B: temp 348 x 6 frames
    idx += [("B", "B.h5", "348", "0", i) for i in range(6)]
    ds = FakeDS(idx)

    check(len(ds.build_catalog()) == 20, "catalog length == 20")

    by_dom = ds.group_indices_by("domain")
    check(set(by_dom) == {"A", "B"}, "grouped by domain -> {A,B}")
    check(len(by_dom["A"]) == 14 and len(by_dom["B"]) == 6, "domain sizes 14/6")

    by_dt = ds.group_indices_by("domain_temp")
    check(set(by_dt) == {("A", "348"), ("A", "320"), ("B", "348")},
          "grouped by domain_temp -> 3 groups")
    check(len(by_dt[("A", "348")]) == 10, "(A,348) has 10 frames")

    # Per-sequence batches: every batch stays within one domain.
    s = m.PerSequenceBatchSampler(ds, batch_size=4, group_key="domain",
                                  shuffle=True, seed=3)
    batches = list(iter(s))
    within = True
    for b in batches:
        doms = {idx[i][0] for i in b}
        if len(doms) != 1:
            within = False
    check(within, "every batch contains exactly one domain")
    total = sum(len(b) for b in batches)
    check(total == 20, "batches cover all 20 frames (drop_last=False)")
    check(len(s) == len(batches), "len(sampler) matches produced batches")

    # domain_temp grouping keeps temperature constant too.
    s2 = m.PerSequenceBatchSampler(ds, batch_size=4, group_key="domain_temp",
                                   shuffle=False, seed=0)
    ok_dt = True
    for b in iter(s2):
        keys = {(idx[i][0], idx[i][2]) for i in b}
        if len(keys) != 1:
            ok_dt = False
    check(ok_dt, "domain_temp batches keep (domain,temp) constant")

    # drop_last drops the ragged tail.
    s3 = m.PerSequenceBatchSampler(ds, batch_size=4, group_key="domain",
                                   shuffle=False, drop_last=True)
    for b in iter(s3):
        check_len = (len(b) == 4)
        if not check_len:
            check(False, "drop_last batch not full")
            break
    else:
        check(True, "drop_last yields only full batches")

    # max_frames_per_group caps per-domain frames.
    s4 = m.PerSequenceBatchSampler(ds, batch_size=4, group_key="domain",
                                   shuffle=False, max_frames_per_group=8)
    capped = sum(len(b) for b in iter(s4))
    check(capped == 8 + 6, "max_frames_per_group caps A to 8 (A=8 + B=6 = 14)")

    # determinism.
    a = list(iter(m.PerSequenceBatchSampler(ds, 4, "domain", True, seed=9)))
    b = list(iter(m.PerSequenceBatchSampler(ds, 4, "domain", True, seed=9)))
    check(a == b, "same seed -> identical batch plan")

    print(f"\n{'TESTS FAILED' if fail else 'ALL TESTS PASSED'} ({fail} failures)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
