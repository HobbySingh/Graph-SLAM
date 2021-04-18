from load import data_loader

if __name__ == "__main__":

    data_file = "../data/input_INTEL_g2o.g2o"
    g = data_loader(data_file)

    g.plot()

    print("Error before optimization: ", g.calc_chi2())