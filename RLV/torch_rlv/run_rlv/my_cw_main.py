from cw2.cw_data import cw_logging
from cw2 import experiment, cluster_work
import gc
from torch_rlv import run_torch_rlv


class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        run_torch_rlv(config)

    def finalize(self, crash: bool = False):  # called after the current repetition finishes or crashes
        if crash:
            # run failed. You can do some error handling here.
            pass
        self._logger.info("Finishing current run. \n")
        # clean up
        gc.collect()  # make sure there is no residual stuff left
        while self._logger.hasHandlers():
            self._logger.removeHandler(self._logger.handlers[0])


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()
