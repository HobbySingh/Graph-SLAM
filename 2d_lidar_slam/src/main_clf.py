from load import data_loader
import sys
import os

import frontend

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "../data/intel.clf"
    name = os.path.splitext(os.path.split(data_file)[-1])[0]
    if "_" in name:
        # breakpoint()
        name = name.split("_")[-2]

    final_graph = frontend.run(data_file, name, save_gif=True, plot_every=100)
    # g = data_loader(data_file)

    # g.plot(title=f"Before_{name}")

    # print("Error before optimization: ", g.calc_chi2())
    # g.optimize()
    final_graph.plot(title=f"After_{name}")
