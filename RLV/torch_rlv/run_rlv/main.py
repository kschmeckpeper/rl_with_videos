from cw2.cw_data import cw_logging
from cw2 import cluster_work, experiment
from RLV.torch_rlv.executor.run_experiment import run_experiment

import gc
import logging
import sys
import numpy as np
import torch
from typing import Dict, Any


class CustomExperiment(experiment.AbstractExperiment):

    def initialize(self, config: Dict[str, Any], rep: int, logger: cw_logging.AbstractLogger) -> None:
        # called before starting a new repetition

        # todo I still use my custom recording for legacy reasons, but there should be examples in the cw2 code
        #   on how to use the built-in recorder. It also has a built-in wandb logger.
        self._logger = logging.getLogger("MAIN")
        self._logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)  # log everything but DEBUG logs
        self._logger.addHandler(stream_handler)
        runname = config["_experiment_name"] + "_" + str(rep)
        self._logger.info("Current Run: {}. Repetition: {}".format(runname, rep))

    def run(self, config: dict, rep: int, logger) -> None:
        # called after initialize to actually start a new repetition

        # have (global) random seed depend on repetition to make things both reproducible and
        # different for different reps
        torch.manual_seed(seed=rep)
        np.random.seed(seed=rep)

        experiment_name = config.get("_experiment_name")
        my_config = config.get("params")  # move into custom config. This is now everything that you specified
        run_experiment(config=my_config, experiment_name=experiment_name)

    def _run_algorithm(self, config: Dict[str, Any], experiment_name: str):
        # import your algorithm here and run it
        self._logger.info("Logging config for experiment '{}'".format(experiment_name))
        import pprint
        self._logger.info(pprint.pformat(config))


    def finalize(self, surrender=None, crash=False):  # called after the current repetition finishes or crashes
        if crash:
            # run failed. You can do some error handling here.
            # Note that the error messages are automatically saved in the clusterwork logs
            pass
        self._logger.info("Finishing current run. \n")
        # clean up
        gc.collect()  # make sure there is no residual stuff left
        while self._logger.hasHandlers():
            self._logger.removeHandler(self._logger.handlers[0])


if __name__ == "__main__":
    # you call this via "python main.py config.yaml -e my_run -o".
    # The -e is for specifying an experiment from your config,
    # -o is for "overwrite logging if it already exists"
    cw = cluster_work.ClusterWork(CustomExperiment)
    cw.run()
    # you can add your loggers here
