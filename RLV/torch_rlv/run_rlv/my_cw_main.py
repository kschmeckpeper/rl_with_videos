from cw2.cw_data import cw_logging
from cw2 import experiment, cluster_work, cw_error
from torch_rlv import run_torch_rlv
#import wandb

class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        #wandb.init(project="rlv_on_slurm")
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        my_config = config.get("params")  # move into custom config. This is now everything that you specified
        #wandb.log(my_config)
        run_torch_rlv(config=my_config)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()
