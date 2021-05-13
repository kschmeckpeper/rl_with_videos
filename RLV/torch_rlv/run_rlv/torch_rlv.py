from RLV.torch_rlv.executor.run_experiment import run_experiment


def main():
    run_experiment("acrobot", "sac", 2500, 1000, 256, 256, 0.0003)


if __name__ == "__main__":
    main()
