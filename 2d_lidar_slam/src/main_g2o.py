import os
import sys

from load import data_loader

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "../data/input_INTEL_g2o.g2o"
    name = os.path.splitext(os.path.split(data_file)[-1])[0]
    if "_" in name:
        # breakpoint()
        name = name.split("_")[-2]

    g = data_loader(data_file)

    g.plot(title=f"./../results/Before_{name}")

    print("Error before optimization: ", g.calc_chi2())
    g.optimize()
    g.plot(title=f"./../results/After_{name}")
