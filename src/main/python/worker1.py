import os
import json

tf_config = {
    'cluster': {
        # 'worker': ['akmaserver:12345','akmaserver:23456']
        'worker': ['akmaserver:12345','desktop-VG8BID2:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CONFIG'] = json.dumps(tf_config)

import main