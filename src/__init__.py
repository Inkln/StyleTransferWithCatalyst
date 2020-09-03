import os
from time import strftime, gmtime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ConsoleLogger:
    def __init__(self):
        pass

    def _message(self, text, level):
        call_time = strftime("%H:%M:%S", gmtime())
        result = f"{call_time}: {text}"
        print(result)

    def debug(self, message):
        self._message(message, "DEBUG")

    def info(self, message):
        self._message(message, "INFO")


logger = ConsoleLogger()

from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from .trainable_models import *
from .loss_models import *
