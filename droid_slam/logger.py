
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


SUM_FREQ = 100

class Logger:
    def __init__(self, name, scheduler):
        self.total_steps = 0
        self.running_loss = {}
        self.running_count = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter('runs/%s' % self.name)
            logger.info(f"Tracking metrics: {[k for k in self.running_loss]}")

        lr = self.scheduler.get_lr().pop()
        # average each key over the pushes that actually contained it: with
        # alternating labeled/gt-free steps, each domain's keys only appear in a
        # fraction of the pushes, so dividing by SUM_FREQ would understate them
        metrics_data = [self.running_loss[k]/self.running_count[k] for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, lr)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)

        logger.info(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / self.running_count[key]
            self.writer.add_scalar(key, val, self.total_steps)

    def push(self, metrics):

        for key in metrics:
            self.running_loss[key] = self.running_loss.get(key, 0.0) + metrics[key]
            self.running_count[key] = self.running_count.get(key, 0) + 1

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}
            self.running_count = {}

        self.total_steps += 1

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

