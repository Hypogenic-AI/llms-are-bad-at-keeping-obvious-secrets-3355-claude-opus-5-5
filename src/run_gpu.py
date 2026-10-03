"""Run all GPU stages in one process (model loading from the slow disk takes minutes)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import e1_generate, e2_generate, mech_m1, mech_m2, mech_m3  # noqa

STAGES = {"e1": e1_generate.run, "e2": e2_generate.run, "m1": mech_m1.m1, "m1b": mech_m1.m1b,
          "m3": lambda: mech_m3.run(["abl_secret", "abl_other", "abl_random", "abl_secret_sub", "abl_other_sub",
                                     "abl_secret_late", "abl_secret_early", "ko_sentence"]),
          "m2": mech_m2.main,
          "steer": lambda: mech_m3.run(["steer_L24_c1", "steer_L24_c3", "steer_L32_c1"])}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(STAGES)):
        t = time.time()
        print(f"=== stage {s} start", flush=True)
        STAGES[s]()
        print(f"=== stage {s} done in {time.time() - t:.0f}s", flush=True)
