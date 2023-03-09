
'''
 Common configurations

'''


import flags

#conf = flag
#conf = flags.conf

from absl import flags
conf = flags.FLAGS


import os

# GPU setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,4"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
import tensorflow as tf

#
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import matplotlib
matplotlib.use('TkAgg')

import re

#
import utils


class Config():
    def __init__(self):
        #
        self.set_model_dataset()

        #
        self.set_train()
        self.set_eager_mode()

        self.set_batch_size()

        #
        self.set_file_path()

        #
        self.set_hp_tune()

        #
        self.set_load_model()

        #
        self.set_metric()

    def set_model_dataset(self):
        self.model_name = conf.model
        self.dataset_name = conf.dataset

    def set_train(self):
        self.train = (conf.mode=='train') or (conf.mode=='load_and_train')
        self.train_type = conf.train_type
        self.train_epoch = conf.train_epoch

    def set_eager_mode(self):
        # eager mode
        if self.train:
            eager_mode=False
        else:
            if conf.f_write_stat:
                eager_mode=True
                #eager_mode=False
            else:
                eager_mode=False

        if conf.debug_mode:
            # TODO: parameterize - debug mode
            eager_mode=True

        self.eager_mode = eager_mode

    def set_batch_size(self):

        # TODO: batch size calculation unification
        #batch_size_inference = batch_size_inference_sel.get(model_name,256)
        batch_size_train = conf.batch_size
        if self.train:
            batch_size_inference = batch_size_train
        else:
            if conf.full_test:
                batch_size_inference = conf.batch_size_inf
            else:
                if conf.batch_size_inf > conf.num_test_data:
                    batch_size_inference = conf.num_test_data
                else:
                    batch_size_inference = conf.batch_size_inf
            #batch_size_train = batch_size_train_sel.get(model_name,256)

        if not conf.full_test:
            assert (conf.num_test_data%batch_size_inference)==0

        #
        if self.train:
            batch_size = batch_size_train
        else:
            batch_size = batch_size_inference

        self.batch_size = batch_size
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference

    def set_file_path(self):
        self.filepath_save, self.filepath_load, self.config_name = utils.set_file_path(self.batch_size_train)

        # TODO: check
        if conf.path_stat_root=='':
            path_stat_root = self.filepath_load
        else:
            path_stat_root = conf.path_stat_root
        self.path_stat = os.path.join(path_stat_root,conf.path_stat_dir)


    def set_hp_tune(self):
        self.f_hp_tune_train = self.train and conf.hp_tune
        self.f_hp_tune_load = (not self.train) and conf.hp_tune

    def set_load_model(self):
        self.load_model = (conf.mode=='inference') or (conf.mode=='load_and_train')

        # TODO: integration - ImageNet
        if self.load_model and (not self.f_hp_tune_load):
            #if conf.dataset == 'ImageNet':
            if False:
                # ImageNet pretrained model
                load_weight = 'imagenet'
                include_top = True
                add_top = False
            else:
                # get latest saved model
                #latest_model = utils.get_latest_saved_model(filepath)

                latest_model = utils.get_latest_saved_model(self.filepath_load)
                load_weight = os.path.join(self.filepath_load, latest_model)
                #print('load weight: '+load_weight)
                #pre_model = tf.keras.models.load_model(load_weight)

                #latest_model = lib_snn.util.get_latest_saved_model(filepath)
                #load_weight = os.path.join(filepath, latest_model)


                if not latest_model.startswith('ep-'):
                    #assert False, 'the name of latest model should start with ''ep-'''
                    print('the name of latest model should start with ep-')

                    load_weight = tf.train.latest_checkpoint(self.filepath_load)

                    # TODO:
                    init_epoch = 0
                else:
                    print('load weight: '+load_weight)

                    if conf.mode=='inference':
                        init_epoch = int(re.split('-|\.',latest_model)[1])
                    elif conf.mode=='load_and_train':
                        init_epoch = 0
                    else:
                        assert False

                include_top = True
                add_top = False
        else:
            if self.train_type == 'transfer':
                load_weight = 'imagenet'
                include_top = False
                add_top = True
            elif self.train_type == 'scratch':
                load_weight = None
                include_top = True
                add_top = False
            else:
                assert False

            init_epoch = 0

        self.include_top = include_top
        self.add_top = add_top
        self.load_weight = load_weight
        self.init_epoch = init_epoch


    def set_metric(self):
        self.metric_name_acc='acc'
        self.metric_name_acc_top5='acc-5'
        self.monitor_cri='val_'+self.metric_name_acc





config = Config()
