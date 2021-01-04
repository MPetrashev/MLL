import os
import json

import tensorflow as tf
import mnist

per_worker_batch_size = 64
# tf_config = json.loads(os.environ['TF_CONFIG'])
tf_config = {
    'cluster': {
        'worker': ['localhost:12345']
        # 'worker': ['localhost:12345','desktop-VG8BID2:12345']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CONFIG'] = json.dumps(tf_config)

num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)