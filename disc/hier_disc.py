import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
from .hier_rnn_model import Hier_rnn_model
import utils.data_utils as data_utils

from tensorflow.python.platform import gfile
import sys

sys.path.append("../utils")


def evaluate(session, model, config, evl_inputs, evl_labels, evl_masks):
    total_num = len(evl_inputs[0])

    fetches = [model.correct_num, model.prediction, model.logits, model.target]
    feed_dict = {}
    for i in xrange(config.max_len):
        feed_dict[model.input_data[i].name] = evl_inputs[i]
    feed_dict[model.target.name] = evl_labels
    feed_dict[model.mask_x.name] = evl_masks
    correct_num, prediction, logits, target = session.run(fetches, feed_dict)

    print("total_num: ", total_num)
    print("correct_num: ", correct_num)
    print("prediction: ", prediction)
    print("target: ", target)

    accuracy = float(correct_num) / total_num
    return accuracy


def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [data_utils.PAD_ID] * (query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set


def hier_get_batch(config, max_set, query_set, answer_set, gen_set):
    
    """
    train_query  = [[True Q0],[True Q0],[True Q1],[True Q1].....[True Q256]] 
    train_answer = [[True A0],[Fake A0],[True A1],[Fake A1].....[Fake A256]] 
    train_labels = [[   1   ],[   0   ],[   1   ],[   0   ].....[    0    ]]
    
    """
    
    batch_size = config.batch_size # 256
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = int(batch_size / 2) # 128 除二是要 True和Fake都要相對長度
    for _ in range(half_size):
        index = random.randint(0, max_set) # 0 ~ 255 random int
        train_query.append(query_set[index]) # True Q
        train_answer.append(answer_set[index]) # True A
        train_labels.append(1) # True lebel
        train_query.append(query_set[index]) # True Q
        train_answer.append(gen_set[index]) # Fake A
        train_labels.append(0) # False label
         
    return train_query, train_answer, train_labels


def create_model(sess, config, name_scope, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = Hier_rnn_model(config=config, name_scope=name_scope)
        disc_ckpt_dir = os.path.abspath(os.path.join(config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model


def prepare_data(config):
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
        data_utils.hier_prepare_disc_data(config.train_dir, vocab, config.vocab_size)
    query_set, answer_set, gen_set = hier_read_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def hier_train(config_disc, config_evl):
    """
    1. prepare data
    2. Use train_set bucket to gain train_bucket_sizes,train_total_size,train_buckets_scale 
    3. create_model
    4. bucket_id : Based on Distribution [0~1] , we pick random bucket(QA)
    5. Get b_query(True Q), b_answer(True A), b_gen(Fake A) (current sentence)
    6. hier_get_batch( batch_size , b_query, b_answer, b_gen) = train_query, train_answer, train_labels
    7. training
    
    ------------- 2019/1/19 --------------------
    summary
    
    1. Get Question(256 * 20) & Answer(256 * 25) as input (batch_size = 256)
    2. Question Through LSTM(2 layer) with embedding(random) , Answer through as well , get q_state,a_state
    3. concate two state and through LSTM(2 layer), get top_state (256*512)
    4. Through 'spare' softmax with cross_entrophy to get a logit(prediction , 0 is Fake , 1 is True)( 256 * 512 * 512 * 2  = 256 * 2)
 
    """
    
    config_evl.keep_prob = 1.0

    print("begin training")

    with tf.Session() as session:

        print("prepare_data")
        query_set, answer_set, gen_set = prepare_data(config_disc)

        train_bucket_sizes = [len(query_set[b]) for b in xrange(len(config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        #dev_query_set, dev_answer_set, dev_gen_set = hier_read_data(dev_query_path, dev_answer_path, dev_gen_path)
        for set in query_set:
            print("set length: ", len(set))
        
        
        # ------------------------------------------ 2019/1/15 --------------------------
        
        model = create_model(session, config_disc, name_scope=config_disc.name_model)

        step_time, loss = 0.0, 0.0
        current_step = 0
        #previous_losses = []
        step_loss_summary = tf.Summary()
        disc_writer = tf.summary.FileWriter(config_disc.tensorboard_dir, session.graph)

        while True:
            
            
            
            # Get a random sample input
            # -------------------------------------------------------
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()

            b_query, b_answer, b_gen = query_set[bucket_id], answer_set[bucket_id], gen_set[bucket_id]

            train_query, train_answer, train_labels = hier_get_batch(config_disc, len(b_query)-1, b_query, b_answer, b_gen)

            train_query = np.transpose(train_query) # (20 , 256)
            train_answer = np.transpose(train_answer) # ( 25 , 256)
            
            # 
            # 得到一個隨機的 query & answer(True or Fake)
            # ------------------------------------------ 2019/1/15 --------------------------
            
            # 塞入到 model
            feed_dict = {}
            for i in xrange(config_disc.buckets[bucket_id][0]):
                feed_dict[model.query[i].name] = train_query[i]
                
            for i in xrange(config_disc.buckets[bucket_id][1]):
                feed_dict[model.answer[i].name] = train_answer[i]
                
            feed_dict[model.target.name] = train_labels
            
            fetches = [model.b_train_op[bucket_id], model.b_logits[bucket_id], model.b_loss[bucket_id], model.target]
            # 跑 model
            train_op, logits, step_loss, target = session.run(fetches, feed_dict) # target.shape = (256,1) 

            step_time += (time.time() - start_time) / config_disc.steps_per_checkpoint
            loss += step_loss /config_disc.steps_per_checkpoint
            current_step += 1

            if current_step % config_disc.steps_per_checkpoint == 0:

                disc_loss_value = step_loss_summary.value.add()
                disc_loss_value.tag = config_disc.name_loss
                disc_loss_value.simple_value = float(loss)

                disc_writer.add_summary(step_loss_summary, int(session.run(model.global_step)))
                
                # softmax operation
                # softmax 是為了要讓預測結果都變成 positive prob
                logits = np.transpose(softmax(np.transpose(logits)))
                
                                         
                # reward 
                # reward 在 0.5 左右是正常的
                reward = 0.0
                for logit, label in zip(logits, train_labels):
                    reward += logit[1]  # only for True Answer probility
                reward = reward / len(train_labels)
                print("reward: ", reward)


                print("current_step: %d, step_loss: %.4f" %(current_step, step_loss))


                if current_step % (config_disc.steps_per_checkpoint * 3) == 0:
                    print("current_step: %d, save_model" % (current_step))
                    disc_ckpt_dir = os.path.abspath(os.path.join(config_disc.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    model.saver.save(session, disc_model_path, global_step=model.global_step)


                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

