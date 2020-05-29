# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:13:16 2019

@author: thomas
"""

from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile
sys.path.append('../utils')



def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


dev_set = read_data(gen_config, dev_query, dev_answer)
















