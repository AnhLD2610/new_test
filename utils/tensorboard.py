# Based on https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf  # for tensorboard logging only

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def log_scalar(self, tag, value, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()