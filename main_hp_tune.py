



import re
import datetime
import shutil

from functools import partial

import os

# TF logging setup
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.engine import data_adapter

#
from absl import app
from absl import flags

# HP tune
#import kerastuner as kt
import keras_tuner as kt
#import tensorboard.plugins.hparams import api as hp


#
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#
#import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
#np.set_printoptions(precision=4)
#np.set_printoptions(linewidth=np.inf)
#import tensorflow.experimental.numpy as tnp
#tnp.set_printoptions(linewidth=np.inf)

#import argparse
#import cv2

# configuration
import config
from config import conf

# snn library
import lib_snn

#
import datasets
#global input_size
#global input_size_pre_crop_ratio
import collections


# TODO: check use
global model_name


#
from lib_snn.sim import glb_plot
from lib_snn.sim import glb_plot_1
from lib_snn.sim import glb_plot_2

from lib_snn.sim import glb_ig_attributions
from lib_snn.sim import glb_rand_vth
from lib_snn.sim import glb_vth_search_err
from lib_snn.sim import glb_vth_init
from lib_snn.sim import glb_bias_comp

#
#conf = flags.FLAGS

########################################
# configuration
########################################

# logging - ignore warning
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# GPU setting
#
GPU_NUMBER=1

GPU_PARALLEL_RUN = 1
#GPU_PARALLEL_RUN = 2
#GPU_PARALLEL_RUN = 3

# RTX3090
if GPU_PARALLEL_RUN == 1:
    gpu_mem = -1
    NUM_PARALLEL_CALL = 15
elif GPU_PARALLEL_RUN == 2:
    gpu_mem = 10240
    NUM_PARALLEL_CALL = 7
elif GPU_PARALLEL_RUN == 3:
    gpu_mem = 6144
    NUM_PARALLEL_CALL = 5
else:
    assert False



# exp set name
#exp_set_name = 'HPTune-RS'
#exp_set_name = 'HPTune-TEST'
#exp_set_name = 'HPTune-GRID'
#exp_set_name = 'HPTune-GRID'
#exp_set_name = 'CODE_TEST'
#exp_set_name = 'Train_SC'
exp_set_name = 'Train_SC_resnet'
#exp_set_name = 'DNN-to-SNN'

# hyperparamter tune mode
#hp_tune = True
hp_tune = False


#
#train=True
#train=False
train=conf.train

# TODO: parameterize
load_model=True
#load_model=False

#
#save_model = False
save_model = True

#
#overwrite_train_model =True
overwrite_train_model=False

#
overwrite_tensorboard = True

#epoch = 20000
#epoch = 20472
#train_epoch = 300
train_epoch = 1000
#train_epoch = 1


# learning rate schedule - step_decay
#step_decay_epoch = 100
step_decay_epoch = 200


# TODO: move to config
#
root_hp_tune = './hp_tune'

#
#root_model = './models_trained'
root_model = './models_trained_resnet_relu_debug'

# model
#model_name = 'VGG16'
#model_name = 'ResNet18'
#model_name = 'ResNet20'
#model_name = 'ResNet32'
#model_name = 'ResNet34'
#model_name = 'ResNet50'
#model_name = 'ResNet18V2'
#model_name = 'ResNet20V2'
model_name = conf.model

# dataset
#dataset_name = 'CIFAR10'
#dataset_name = 'CIFAR100'
#dataset_name='ImageNet'
dataset_name = conf.dataset

#
#learning_rate = 0.2
#learning_rate = 0.01
learning_rate = conf.learning_rate

#
opt='SGD'

#
#lr_schedule = 'COS'     # COSine
#lr_schedule = 'COSR'    # COSine with Restart
lr_schedule = 'STEP'    # STEP wise
#lr_schedule = 'STEP_WUP'    # STEP wise, warmup


#
#root_tensorboard = './tensorboard/'
root_tensorboard = conf.root_tensorboard





# lr schedule



# models
#from models.vgg16 import VGG16
from models.vgg16_keras_toh5 import VGG16 as VGG16_KERAS

#from models.vgg16_tr import VGG16_TR
#from models.vgg16 import VGG16
#from models.resnet import ResNet18
#from models.resnet import ResNet20
#from models.resnet import ResNet32
#from models.resnet import ResNet34
#from models.resnet import ResNet50
#from models.resnet import ResNet101
#from models.resnet import ResNet152
#from models.resnet import ResNet18V2
#from models.resnet import ResNet20V2

from models.models import model_sel

#
#from lib_snn.hp_tune_model import model_builder




#
#import test
#import train

#
#import models.input_preprocessor as preprocessor

#
#tf.config.functions_run_eagerly()


os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_NUMBER)

# TODO: gpu mem usage - parameterize
# GPU mem usage
#if False:
#if False:
#if True:
if gpu_mem != -1:
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpu[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_mem)])
        except RuntimeError as e:
            print(e)


# training types
#train_type='finetuning' # not supported yet
#train_type='transfer'
train_type='scratch'


#
#model_name='Xception'
#model_name='VGG16'
#model_name='VGG19'
#model_name='ResNet50'
#model_name='ResNet101'
#model_name='ResNet152'
#model_name='ResNet50V2'
#model_name='ResNet101V2'
#model_name='ResNet152V2'
#model_name='InceptionV3'
#model_name='InceptionResNetV2'
#model_name='MobileNet'
#model_name='MobileNetV2'
#model_name='DenseNet121'
#model_name='DenseNet169'
#model_name='DenseNet201'
#model_name='NASNetMobile'
#model_name='NASNetLarge'
#model_name='EfficientNetB0'
#model_name='EfficientNetB1'
#model_name='EfficientNetB2'
#model_name='EfficientNetB3'
#model_name='EfficientNetB4'
#model_name='EfficientNetB5'
#model_name='EfficientNetB6'
#model_name='EfficientNetB7'



#
assert conf.data_format == 'channels_last', 'not support "{}", only support channels_last'.format(conf.data_format)

########################################
# DO NOT TOUCH
########################################

#
f_hp_tune = train and hp_tune

# data augmentation - mix




# l2-norm
#lmb = 1.0E-10
lmb = conf.lmb




GPU = 'RTX_3090'
# NVIDIA TITAN V (12GB)
if GPU=='NVIDIA_TITAN_V':
    batch_size_inference_sel ={
        'NASNetLarge': 128,
        'EfficientNetB4': 128,
        'EfficientNetB5': 128,
        'EfficientNetB6': 64,
        'EfficientNetB7': 64,
    }


batch_size_inference_sel ={
    'NASNetLarge': 128,
    'EfficientNetB4': 128,
    'EfficientNetB5': 128,
    'EfficientNetB6': 64,
    'EfficientNetB7': 64,
}

batch_size_train_sel = {
    #'VGG16': 256,
    'VGG16': 512,
    #'VGG16': 1024,
    #'VGG16': 2048,
}

# TODO:
dataset_sel = {
    'ImageNet': datasets.imagenet,
    'CIFAR10': datasets.cifar,
    'CIFAR100': datasets.cifar,
}


#
input_size_default = {
    'ImageNet': 244,
    'CIFAR10': 32,
    'CIFAR100': 32,
}

#
input_sizes_imagenet = {
    'Xception': 299,
    'InceptionV3': 299,
    'InceptionResNetV2': 299,
    'NASNetLarge': 331,
    'EfficientNetB1': 240,
    'EfficientNetB2': 260,
    'EfficientNetB4': 380,
    'EfficientNetB5': 456,
    'EfficientNetB6': 528,
    'EfficientNetB7': 600,
}

input_sizes_cifar = {
    'VGG16': 32,
}

#
input_size_sel ={
    'ImageNet': input_sizes_imagenet,
    'CIFAR10': input_sizes_cifar,
    'CIFAR100': input_sizes_cifar,
}

# TODO: integrate input size selector
input_size = input_size_sel[dataset_name].get(model_name,input_size_default[dataset_name])
#input_size = input_sizes.get(model_name,224)


#
initial_channels_sel= {
    'VGG16': 64,
}
initial_channels = initial_channels_sel.get(model_name,64)


# TODO: batch size calulation unification
#batch_size_inference = batch_size_inference_sel.get(model_name,256)
batch_size_train = conf.batch_size
if train:
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


#
if train:
    batch_size = batch_size_train
else:
    batch_size = batch_size_inference


#
image_shape = (input_size, input_size, 3)


# dataset load
#dataset = dataset_sel[dataset_name]
#train_ds, valid_ds, test_ds = dataset.load(dataset_name,input_size,input_size_pre_crop_ratio,num_class,train,NUM_PARALLEL_CALL,conf,input_prec_mode)
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class =\
    datasets.datasets.load(dataset_name,batch_size,input_size,train_type,train,conf,NUM_PARALLEL_CALL)


# data-based weight normalization (DNN-to-SNN conversion)
if conf.f_write_stat and conf.f_stat_train_mode:
    test_ds = train_ds

#assert False
train_steps_per_epoch = train_ds.cardinality().numpy()




# model compile
metric_accuracy = tf.keras.metrics.categorical_accuracy
metric_accuracy_top5 = tf.keras.metrics.top_k_categorical_accuracy

# TODO: move to configuration

metric_name_acc = 'acc'
metric_name_acc_top5 = 'acc-5'
monitor_cri = 'val_' + metric_name_acc

metric_accuracy.name = metric_name_acc
metric_accuracy_top5.name = metric_name_acc_top5

#model.compile(optimizer='adam',
              #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              ## metrics=['accuracy'])
              #metrics=[metric_accuracy, metric_accuracy_top5])

################
# name set
################

#
if conf.load_best_model:
    root_model = conf.root_model_best


# TODO: configuration & file naming
#exp_set_name = model_name + '_' + dataset_name
model_dataset_name = model_name + '_' + dataset_name

# path_model = './'+exp_set_name
#path_model = os.path.join(root_model, exp_set_name)
path_model = os.path.join(root_model, model_dataset_name)


# hyperparameter tune name
#hp_tune_name = exp_set_name+'_'+model_dataset_name+'_ep-'+str(train_epoch)
hp_tune_name = exp_set_name

# TODO: functionalize
# file_name='checkpoint-epoch-{}-batch-{}.h5'.format(epoch,batch_size)
# config_name='ep-{epoch:04d}_bat-{}_lmb-{:.1E}'.format(batch_size,lmb)
# config_name='bat-{}_lmb-{:.1E}'.format(batch_size,lmb)

#config_name = 'bat-{}_opt-{}_lr-{:.0E}_lmb-{:.0E}'.format(batch_size,opt,learning_rate,lmb)
config_name = 'ep-{}_bat-{}_opt-{}_lr-{}-{:.0E}_lmb-{:.0E}'.format(train_epoch,batch_size_train,opt,lr_schedule,learning_rate,lmb)

#config_name = 'bat-{}_lmb-{:.0E}'.format(batch_size, lmb)
#config_name = 'bat-512_lmb-{:.1E}'.format(lmb)

if train_type=='transfer':
    config_name += '_tr'
elif train_type=='scratch':
    config_name += '_sc'
    #if n_dim_classifier is not None:
        #if model_name == 'VGG16':
            #config_name = config_name+'-'+str(n_dim_classifier[0])+'-'+str(n_dim_classifier[1])
else:
    assert False

if conf.data_aug_mix == 'mixup':
    en_mixup = True
    en_cutmix = False
elif conf.data_aug_mix == 'cutmix':
    en_mixup = False
    en_cutmix = True
else:
    en_mixup = False
    en_cutmix = False

if en_mixup:
    config_name += '_mu'
elif en_cutmix:
    config_name += '_cm'

#
if train:
    filepath = os.path.join(path_model, config_name)
else:
    if conf.load_best_model:
        filepath = path_model
    else:
        filepath = os.path.join(path_model, config_name)




########################################
#
########################################

model_top = model_sel(model_name,train_type)

if load_model:
    # get latest saved model
    #latest_model = lib_snn.util.get_latest_saved_model(filepath)

    latest_model = lib_snn.util.get_latest_saved_model(filepath)
    load_weight = os.path.join(filepath, latest_model)
    print('load weight: '+load_weight)
    #pre_model = tf.keras.models.load_model(load_weight)

    #latest_model = lib_snn.util.get_latest_saved_model(filepath)
    #load_weight = os.path.join(filepath, latest_model)


    if not latest_model.startswith('ep-'):
        assert False, 'the name of latest model should start with ''ep-'''
    init_epoch = int(re.split('-|\.',latest_model)[1])

    include_top = True
    add_top = False

    #if train_type == 'transfer':
        #model_top = model_sel_tr[model_name]
    #elif train_type == 'scratch':
        #model_top = model_sel_sc[model_name]
    #else:
        #assert False
else:
    if train_type == 'transfer':
        load_weight = 'imagenet'
        include_top = False
        add_top = True

        #model_top = model_sel_tr[model_name]

    elif train_type == 'scratch':
        load_weight = None
        include_top = True
        add_top = False

        #model_top = model_sel_sc[model_name]
    else:
        assert False

    init_epoch = 0



# TODO: move to parameter
# eager mode
if train:
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


# for HP tune
model_top_glb = model_top

#
# model builder
if f_hp_tune:

    # TODO: move to config.py
    #hp_model_builder = model_builder
    hps = collections.OrderedDict()
    hps['dataset'] = [dataset_name]
    hps['model'] = [model_name]
    hps['opt'] = [opt]
    hps['lr_schedule'] = [lr_schedule]
    hps['train_epoch'] = [train_epoch]
    hps['step_decay_epoch'] = [step_decay_epoch]

    # main to hp_tune, need to seperate configuration
    hp_tune_args = collections.OrderedDict()
    hp_tune_args['model_top'] = model_top
    hp_tune_args['batch_size'] = batch_size
    hp_tune_args['image_shape'] = image_shape
    hp_tune_args['conf'] = conf
    hp_tune_args['include_top'] = include_top
    hp_tune_args['load_weight'] = load_weight
    hp_tune_args['num_class'] = num_class
    hp_tune_args['metric_accuracy'] = metric_accuracy
    hp_tune_args['metric_accuracy_top_5'] = metric_accuracy_top5
    hp_tune_args['train_steps_per_epoch'] = train_steps_per_epoch

    #hp_model_builder = partial(model_builder, hp, hps)
    hp_model_builder = lib_snn.hp_tune_model.CustomHyperModel(hp_tune_args, hps)


    #tuner = kt.Hyperband(model_builder,
    #tuner=kt.RandomSearch(model_builder,
    tuner=lib_snn.hp_tune.GridSearch(hp_model_builder,
                         objective='val_acc',
                         #max_trials=12,
                         #max_epochs = 300,
                         #factor=3,
                         overwrite=True,
                         directory=root_hp_tune,
                         project_name=hp_tune_name,
                         #directory='test_hp_dir',
                         #project_name='test_hp')
                          )

    #tuner.results_summary()
    #assert False
else:
    model = lib_snn.model_builder.model_builder(
        eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)

#
if conf.nn_mode=='SNN' and conf.dnn_to_snn:
    print('DNN-to-SNN mode')
    nn_mode_ori = conf.nn_mode
    conf.nn_mode='ANN'
    model_ann = lib_snn.model_builder.model_builder(
        eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
        train_epoch, train_steps_per_epoch,
        opt, learning_rate,
        lr_schedule, step_decay_epoch,
        metric_accuracy, metric_accuracy_top5)
    conf.nn_mode=nn_mode_ori

    #model_ann.set_en_snn('ANN')

    model_ann.load_weights(load_weight)

    print('-- model_ann - load done')
    model.load_weights_dnn_to_snn(model_ann)

    #del(model_ann)


elif load_model:
    model.load_weights(load_weight)
    #model.load_weights(load_weight, by_name=True, skip_mismatch=True)
    #model.load_weights_custom(load_weight)
    #model.load_weights(load_weight, by_name=True)
    # model.load_weights(load_weight,by_name=

if conf.nn_mode=='ANN':
    model_ann=None


#ann_kernel={}
#snn_kernel={}
#
#ann_bias={}
#snn_bias={}
#
#ann_bn={}
#snn_bn={}
#
#print('loaded kernel')
#for layer in model_ann.layers:
#    if hasattr(layer,'kernel'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.kernel)))
#        ann_kernel[layer.name] = tf.reduce_sum(layer.kernel)
#
#
#print('loaded bias')
#for layer in model_ann.layers:
#    if hasattr(layer,'bias'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.bias)))
#        ann_bias[layer.name] = tf.reduce_sum(layer.bias)
#
#print('loaded bn')
#for layer in model_ann.layers:
#    if hasattr(layer, 'bn') and layer.bn is not None:
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.beta)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.gamma)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_mean)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_variance)))
#        ann_bn[layer.name] = tf.reduce_sum(layer.bn.beta)
#
#
#print('loaded kernel')
#for layer in model.layers:
#    if hasattr(layer,'kernel'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.kernel)))
#        snn_kernel[layer.name] = tf.reduce_sum(layer.kernel)
#
#
#print('loaded bias')
#for layer in model.layers:
#    if hasattr(layer,'bias'):
#        print('{} - {}'.format(layer.name,tf.reduce_sum(layer.bias)))
#        snn_bias[layer.name] = tf.reduce_sum(layer.bias)
#
#print('loaded bn')
#for layer in model.layers:
#    if hasattr(layer, 'bn') and layer.bn is not None:
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.beta)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.gamma)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_mean)))
#        print('{} - {}'.format(layer.name, tf.reduce_sum(layer.bn.moving_variance)))
#        snn_bn[layer.name] = tf.reduce_sum(layer.bn.beta)
#
#
##for layer in model.layers:
#for layer_name in snn_kernel.keys():
#    assert snn_kernel[layer_name]==ann_kernel[layer_name]
#
#for layer_name in snn_bias.keys():
#    assert snn_bias[layer_name]==ann_bias[layer_name]
#
#for layer_name in snn_bn.keys():
#    assert snn_bn[layer_name]==ann_bn[layer_name]

#assert False


#
#model.make_test_function = lib_snn.training.make_test_function(model)

#
#if train:
    #print('Train mode')
# remove dir - train model
if not load_model:
    if overwrite_train_model:
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)

# path_tensorboard = root_tensorboard+exp_set_name
# path_tensorboard = root_tensorboard+filepath

if f_hp_tune:
    path_tensorboard = os.path.join(root_tensorboard, hp_tune_name)

else:
    path_tensorboard = os.path.join(root_tensorboard, exp_set_name)
    path_tensorboard = os.path.join(path_tensorboard, model_dataset_name)
    path_tensorboard = os.path.join(path_tensorboard, config_name)

if not overwrite_tensorboard:
    if os.path.isdir(path_tensorboard):
        date_cur = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
        path_dest_tensorboard = path_tensorboard + '_' + date_cur
        print('tensorboard data already exists')
        print('move {} to {}'.format(path_tensorboard, path_dest_tensorboard))

        shutil.move(path_tensorboard, path_dest_tensorboard)



########
# Callbacks
########

#
if train and load_model:
    print('Evaluate pretrained model')
    assert monitor_cri == 'val_acc', 'currently only consider monitor criterion - val_acc'
    result = model.evaluate(valid_ds)
    idx_monitor_cri = model.metrics_names.index('acc')
    best = result[idx_monitor_cri]
    print('previous best result - {}'.format(best))
else:
    best = None

# model checkpoint save and resume
cb_model_checkpoint = lib_snn.callbacks.ModelCheckpointResume(
    # filepath=filepath + '/ep-{epoch:04d}',
    # filepath=filepath + '/ep-{epoch:04d}.ckpt',
    filepath=filepath + '/ep-{epoch:04d}.hdf5',
    save_weight_only=True,
    save_best_only=True,
    monitor=monitor_cri,
    verbose=1,
    best=best,
    log_dir=path_tensorboard,
    # tensorboard_writer=cb_tensorboard._writers['train']
)
cb_manage_saved_model = lib_snn.callbacks.ManageSavedModels(filepath=filepath)
cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboard, update_freq='epoch')

#cb_dnntosnn = lib_snn.callbacks.DNNtoSNN()
cb_libsnn = lib_snn.callbacks.SNNLIB(conf,path_model,test_ds_num,model_ann)
cb_libsnn_ann = lib_snn.callbacks.SNNLIB(conf,path_model,test_ds_num)

#
callbacks_train = [cb_tensorboard]
if save_model:
    callbacks_train.append(cb_model_checkpoint)
    callbacks_train.append(cb_manage_saved_model)

callbacks_test = []
# TODO: move to parameters
#dnn_to_snn = True
#if dnn_to_snn:
    #callbacks_test.append(cb_dnntosnn)

callbacks_test = [cb_libsnn]
callbacks_test_ann = [cb_libsnn_ann]

#
if train:
    if hp_tune:
        print('HP Tune mode')

        #callbacks = [cb_tensorboard]

        tuner.search(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                     callbacks=callbacks_train)
    else:
        print('Train mode')

        #callbacks = [
            #cb_model_checkpoint,
            #cb_manage_saved_model,
            #cb_tensorboard
        #]

        model.summary()

        train_histories = model.fit(train_ds, epochs=train_epoch, initial_epoch=init_epoch, validation_data=valid_ds,
                                    callbacks=callbacks_train)
else:
    print('Test mode')

    #
    #dnn_snn_compare=True
    #dnn_snn_compare=False

    compare_control_snn = False
    #compare_control_snn = True

    act_based_calibration = conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21 or conf.calibration_weight_act_based
    #if (conf.nn_mode=='SNN') and (dnn_snn_compare or conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21) :
    #if (conf.nn_mode == 'SNN') and (dnn_snn_compare or act_based_calibration):
    if (conf.nn_mode == 'SNN') and (compare_control_snn or act_based_calibration):

        cb_libsnn_ann.run_for_calibration = True

        #
        #compare_control_snn = True
        if (not conf.full_test) and compare_control_snn and conf.verbose_visual:
            #cb_libsnn_ann.run_for_compare_post_calib = True
            lib_snn.sim.set_for_visual_debug(True)
            model_ann.evaluate(test_ds, callbacks=callbacks_test_ann)
            #cb_libsnn_ann.run_for_compare_post_calib=False
            lib_snn.sim.set_for_visual_debug(False)


        #if conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21:
        #if act_based_calibration:
        if True:
            # random sampling
            #test_ds_one_batch = tf.data.experimental.get_single_element(test_ds)
            #test_ds_one_batch = tf.data.Dataset.from_tensors(test_ds_one_batch)
            images_one_batch, labels_one_batch = next(iter(train_ds))
            #images_one_batch, labels_one_batch = next(iter(test_ds))
            #print(tf.reduce_mean(images_one_batch))
        #else:
            #images_one_batch, labels_one_batch = next(iter(test_ds))

        ds_one_batch = tf.data.Dataset.from_tensors((images_one_batch,labels_one_batch))
        ds_ann = ds_one_batch

        #
        #result = model.evaluate(ds_one_batch, callbacks=callbacks_test)

        #
        nn_mode_ori = conf.nn_mode
        conf.nn_mode = 'ANN'

        # run for init
        for data in ds_ann:
            x = data[0]
            y = data[1]
            x = x[0]
            y = y[0]
            break

        x = tf.expand_dims(x,axis=0)
        y = tf.expand_dims(y,axis=0)
        ds_one_sample = tf.data.Dataset.from_tensors((x,y))

        #result_ann = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)
        result_ann = model_ann.evaluate(ds_one_sample, callbacks=callbacks_test_ann)

        #
        calib_ori = cb_libsnn_ann.calibration
        cb_libsnn_ann.calibration = False

        #
        gradient_test_tmp = False
        #gradient_test_tmp = True
        if gradient_test_tmp:
            with tf.GradientTape(persistent=True) as tape:
            #with tf.GradientTape(watch_accessed_variables=False) as tape:
                for data in ds_ann:
                    x = data[0]
                    y = data[1]
                    y_decoded = tf.argmax(y,axis=1)

                    print(x)
                    print(y)

                    tape.watch(x)
                    #tf.compat.v1.disable_eager_execution()
                    #y_pred = model_ann(x, training=True)
                    y_pred = model_ann(x, training=False)
                    #loss = model_ann.compiled_loss(y,y_pred)

                    print(y_pred)
                    #top_class = y_pred[:, y_decoded]

                    top_class = tf.gather(y_pred,y_decoded)
                    print(top_class)


                    #tape.gradient(loss,x)
                    grads_pred_act = collections.OrderedDict()

                    for l in model_ann.layers_w_kernel:
                        act = l.record_output
                        grads_pred_act[l.name] = tape.gradient(top_class,act)

                    #grads = tape.gradient(top_class,x)
                    #print(grads)


                    print([var.name for var in tape.watched_variables()])
                    assert False


                assert False
                #if False:
                #for epoch, iterator in data_handler.enumerate_epochs():
                    #y_pred = model_ann(ds_ann, callbacks=callba)
                    #data = data_adapter.expand_1d(ds_ann)
                    #x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                    #print(x)
                    #print(y)

            assert False

        #
        # integrated gradients
        #
        num_range = 1000
        for l in model_ann.layers_w_kernel:
            #glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.0, maxval=1, dtype=tf.float32)
            #glb_rand_vth[l.name] = tf.range(1 / num_range, 1 + 1 / num_range, 1 / num_range, dtype=tf.float32)
            glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.0, maxval=1, dtype=tf.float32)
            #glb_rand_vth[l.name] = tf.random.uniform(shape=[num_range], minval=0.5, maxval=1, dtype=tf.float32)
            glb_vth_search_err[l.name] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)


        #num_batch_for_vth_search = 10
        num_batch_for_vth_search = 1

        vth_search = True
        #vth_search = False

        if vth_search:
            for idx_batch in range(num_batch_for_vth_search):
                images_one_batch, labels_one_batch = next(iter(train_ds))
                ds_ann_vth_search = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch))

                if conf.vth_search_ig:
                    one_image=True
                else:
                    one_image=False


                if conf.vth_search_ig:
                    #glb_ig_attributions = []
                    ig_attributions = collections.OrderedDict()

                    m_steps = 50
                    #m_steps = 250
                    # preprocessing
                    for l in model_ann.layers_w_kernel:
                        #glb_ig_attributions[l.name] = tf.TensorArray(tf.float32, size=m_steps+1)
                        ig_attributions[l.name] = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

                    for data in ds_ann_vth_search:
                    #for data in ds_one_batch:
                        #x = data[0]
                        #y = data[1]

                        #y_decoded = tf.argmax(y, axis=1)

                        #if one_image:
                        #    x = x[1]
                        #    y = y[1]
                        #    y_decoded = tf.argmax(y)
                        images = data[0]
                        labels = data[1]

                        for idx, image in enumerate(images):
                            print(idx)
                            label = labels[idx]

                            #image = images[0]
                            #label = labels[0]

                            label_decoded = tf.argmax(label)

                            baseline = tf.zeros(shape=image.shape)

                            ig_attribution = lib_snn.xai.integrated_gradients(model=model_ann,
                                                                             baseline=baseline,
                                                                             images=image,
                                                                             target_class_idxs=label_decoded,
                                                                             m_steps=m_steps)
                            #glb_ig_attributions.append(ig_attribution)

                            for l in model_ann.layers_w_kernel:
                                ig_attributions[l.name] = ig_attributions[l.name].write(idx,ig_attribution[l.name])

                            #assert False

                        for l in model_ann.layers_w_kernel:
                            ig_attr = ig_attributions[l.name].stack()
                            #ig_attr = tf.reduce_mean(ig_attr,axis=0)
                            glb_ig_attributions[l.name] = tf.reduce_sum(tf.math.abs(ig_attr),axis=1) # sum - alphas


                    #cmap = plt.cm.inferno
                    ##cmap = plt.cm.viridis
                    #lib_snn.xai.plot_image_attributions(baseline,image,ig_attribution,cmap=cmap)

                #
                result_ann = model_ann.evaluate(ds_ann_vth_search, callbacks=callbacks_test_ann)

                # calculate and accumulate errors
                #error_level = 'layer'
                error_level = 'channel'

                for l in model_ann.layers_w_kernel:

                    if error_level == 'layer':
                        if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                            axis = [0, 1, 2, 3]
                        elif isinstance(l, lib_snn.layers.Dense):
                            axis = [0, 1]
                        else:
                            assert False
                    elif error_level == 'channel':
                        if isinstance(l, lib_snn.layers.Conv2D) or isinstance(l, lib_snn.layers.InputGenLayer):
                            axis = [0, 1, 2]
                        elif isinstance(l, lib_snn.layers.Dense):
                            axis = [0]
                        else:
                            assert False
                    else:
                        assert False

                    dnn_act = model_ann.get_layer(l.name).record_output

                    if conf.f_w_norm_data:
                        stat_max = tf.reduce_max(dnn_act, axis=axis)
                        stat_max = tfp.stats.percentile(dnn_act, 99.9, axis=axis)
                        stat_max = tf.ones(stat_max.shape)
                        # stat_max = tf.reduce_max(dnn_act, axis=axis)
                        # stat_max = read_stat(self, l, 'max')
                        # stat_max = stat_max / self.norm_b[l.name]
                        # stat_max = tf.expand_dims(stat_max,axis=0)
                        # stat_max = tf.reduce_max(stat_max,axis=axis)
                    else:
                        stat_max = lib_snn.calibration.read_stat(cb_libsnn, l, 'max')
                        stat_max = tf.expand_dims(stat_max, axis=0)
                        stat_max = tf.reduce_max(stat_max, axis=axis)

                    #
                    #if error_level == 'layer':
                        #errs = tf.zeros([num_range])
                    #elif error_level == 'channel':
                        #errs = tf.zeros([num_range, stat_max.shape[0]])
                    #else:
                        #assert False

                    #errs = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

                    # assert False
                    # errs = []

                    if conf.bias_control:
                        time = tf.cast(conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
                    else:
                        time = conf.time_step
                    # time = self.conf.time_step

                    #
                    for idx, vth_scale in enumerate(glb_rand_vth[l.name]):

                        vth = vth_scale * stat_max

                        # clip_max = vth*self.conf.time_step
                        # clip_max = vth*time
                        clip_max = time

                        # dnn_act_clip_floor = tf.math.floor(dnn_act/vth)
                        dnn_act_clip_floor = tf.math.floor(dnn_act / vth * time)
                        dnn_act_clip_floor = tf.clip_by_value(dnn_act_clip_floor, 0, clip_max)
                        dnn_act_clip_floor = dnn_act_clip_floor * vth / time

                        # print(vth)
                        err = dnn_act - dnn_act_clip_floor
                        err = tf.math.square(err)
                        # err = tf.math.square(err)*1/dnn_act

                        # integrated gradients
                        if conf.vth_search_ig:
                            # if isinstance(l, lib_snn.layers.Conv2D):
                            # ig_attributions = tf.reduce_mean(glb_ig_attributions[l.name],axis=[0,1])
                            # elif isinstance(l, lib_snn.layers.Dense):
                            # ig_attributions = glb_ig_attributions[l.name]
                            # else:
                            # assert False
                            # print(err.shape)
                            ig_attributions = glb_ig_attributions[l.name]
                            eps = 0.01
                            # print(err.shape)
                            # print(ig_attributions.shape)
                            # err = err*(1+ig_attributions)
                            # err = err*(10+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(2+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(1+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(0.5+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(0.1+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(0.01+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                            # err = err*(0.001+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(tf.reduce_min(ig_attributions)+ig_attributions)
                            # alpha = 0.5
                            # err = alpha*err/tf.reduce_max(err,axis=axis)+(1-alpha)*1/(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                            # err = err*ig_attributions
                            # err = err*(ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                            # err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions,axis=axis))
                            #err = err * (eps + ig_attributions / tf.reduce_max(ig_attributions))
                            # err = err*(tf.reduce_min(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(tf.reduce_mean(ig_attributions)+ig_attributions/tf.reduce_max(ig_attributions))
                            # err = err*(eps+ig_attributions/tf.reduce_max(ig_attributions))    # old best?
                            err = err*(ig_attributions/tf.reduce_max(ig_attributions))
                            # print(err.shape)

                        # err = err * snn_act
                        # print(err)

                        # err = tf.math.abs(err)

                        # err = err * snn_act
                        # print(err)
                        # err = tf.math.square(err)
                        err = tf.reduce_mean(err, axis=axis)
                        # print(err)

                        #if error_level == 'layer':
                            #errs = tf.tensor_scatter_nd_update(errs, [[idx]], [err])
                        #elif error_level == 'channel':
                            #errs = tf.tensor_scatter_nd_update(errs, [[idx]], [err])

                        #errs[l.name] = errs[l.name].write(idx, err)

                        vth_err_arr = glb_vth_search_err[l.name]
                        #if vth_err_arr.size() > 1:
                        if idx_batch==0:
                            glb_vth_search_err[l.name] = vth_err_arr.write(idx, err)
                        else:
                            glb_vth_search_err[l.name] = vth_err_arr.write(idx, vth_err_arr.read(idx) + err)

                    assert False

        if False:
        #if vth_search:
            for l in model_ann.layers_w_kernel:

                #ig_attr = ig_attributions[l.name].stack()
                # ig_attr = tf.reduce_mean(ig_attr,axis=0)
                #glb_ig_attributions[l.name] = tf.reduce_sum(tf.math.abs(ig_attr), axis=1)  # sum - alphas

                #errs [l.name] = ig_attributions[l.name].write(idx,ig_attribution[l.name])
                errs_layer = glb_vth_search_err[l.name].stack()
                #print(errs_layer.shape)

                vth_idx_min_err = tf.math.argmin(errs_layer)
                print(vth_idx_min_err)

                # vth_min_err = tf.gather(range_vth,vth_idx_min_err)
                vth_min_err_scale = tf.gather(glb_rand_vth[l.name], vth_idx_min_err)
                #vth_min_err = vth_min_err_scale * stat_max
                vth_min_err = vth_min_err_scale
                vth_init = vth_min_err

                glb_vth_init[l.name]=vth_init
                print(vth_init)

                #kssert False

        assert False


        cb_libsnn_ann.calibration = calib_ori
        result_ann = model_ann.evaluate(ds_ann, callbacks=callbacks_test_ann)

        #lib_snn.calibration.weight_calibration_act_based(cb_libsnn)

        conf.nn_mode = nn_mode_ori

        model.evaluate(ds_ann, callbacks=callbacks_test)

        print('here')

        if vth_search:
            lib_snn.calibration.vth_set_and_norm(cb_libsnn)

        #assert False

    #
    # calibration with activations
    # calibration ICML-21
    #
    #if (conf.nn_mode == 'SNN') and (conf.calibration_bias_ICML_21 or conf.calibration_vmem_ICML_21):
    #if (conf.nn_mode == 'SNN') and (act_based_calibration):
    if False:
        # pre
        cb_libsnn.run_for_calibration = True
        glb_plot.mark='ro'
        glb_plot_1.mark='ro'
        glb_plot_2.mark='ro'

        #
        compare_control_snn=True
        if (not conf.full_test) and compare_control_snn and conf.verbose_visual:
            lib_snn.sim.set_for_visual_debug(True)
            model.evaluate(test_ds, callbacks=callbacks_test)
            lib_snn.sim.set_for_visual_debug(False)

        # run
        if conf.calibration_weight:
            print('run for calibration - static')
            model.evaluate(ds_one_batch, callbacks=callbacks_test)

        # run
        if conf.calibration_weight_act_based:
            print('run for calibration - act based')
            model.evaluate(ds_one_batch, callbacks=callbacks_test)

        # run
        if conf.calibration_bias_ICML_21:
            print('run for calibration - post')
            model.evaluate(ds_one_batch, callbacks=callbacks_test)

        # post
        cb_libsnn.run_for_calibration = False
        glb_plot.mark = 'bo'
        glb_plot_1.mark = 'bo'
        glb_plot_2.mark = 'bo'


    # new start
    #num_batch_for_vth_search = 10
    #num_batch_for_vth_search = 4
    #num_batch_for_vth_search = 3
    #num_batch_for_vth_search = 2
    #num_batch_for_vth_search = 1


    #train_ds = test_ds

    #vth_search = True
    vth_search = False


    if vth_search:
        for idx_batch, (x, y) in enumerate(train_ds):
            # ds_one_batch_ann = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch)).take(1).cache()
            # ds_one_batch_snn = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch)).take(1).cache()
            ds_one_batch = tf.data.Dataset.from_tensors((x, y))


            #
            if not model_ann.layers_w_kernel:
                result = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)

            # run - ann
            # callbacks_test[0].model_ann = model_ann
            #result = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)

            # ds_one_batch = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch))
            # run - snn
            #result = model.evaluate(ds_one_batch, callbacks=callbacks_test)


            if conf.vth_search_ig:
                # glb_ig_attributions = []
                ig_attributions = collections.OrderedDict()

                m_steps = 50
                # m_steps = 250
                # preprocessing
                for l in model_ann.layers_w_kernel:
                    # glb_ig_attributions[l.name] = tf.TensorArray(tf.float32, size=m_steps+1)
                    ig_attributions[l.name] = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                                             clear_after_read=False)

                for data in ds_one_batch:
                    # for data in ds_one_batch:
                    # x = data[0]
                    # y = data[1]

                    # y_decoded = tf.argmax(y, axis=1)

                    # if one_image:
                    #    x = x[1]
                    #    y = y[1]
                    #    y_decoded = tf.argmax(y)
                    images = data[0]
                    labels = data[1]

                    for idx, image in enumerate(images):
                        print(idx)
                        label = labels[idx]

                        # image = images[0]
                        # label = labels[0]

                        label_decoded = tf.argmax(label)

                        baseline = tf.zeros(shape=image.shape)

                        ig_attribution = lib_snn.xai.integrated_gradients(model=model_ann,
                                                                          baseline=baseline,
                                                                          images=image,
                                                                          target_class_idxs=label_decoded,
                                                                          m_steps=m_steps)
                        # glb_ig_attributions.append(ig_attribution)

                        for l in model_ann.layers_w_kernel:
                            ig_attributions[l.name] = ig_attributions[l.name].write(idx, ig_attribution[l.name])

                        # assert False

                    for l in model_ann.layers_w_kernel:
                        ig_attr = ig_attributions[l.name].stack()
                        # ig_attr = tf.reduce_mean(ig_attr,axis=0)
                        glb_ig_attributions[l.name] = tf.reduce_sum(tf.math.abs(ig_attr), axis=1)  # sum - alphas
            #assert False

            callbacks_test_ann[0].run_for_vth_search = True
            result = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)
            callbacks_test_ann[0].run_for_vth_search = False

            # idx_batch = idx_batch+1
            # if idx_batch == num_batch_for_vth_search-1:
            if idx_batch == conf.calibration_num_batch - 1:
                break

        result = model.evaluate(ds_one_batch, callbacks=callbacks_test)

        cb_libsnn_ann.f_vth_set_and_norm=True
        result = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)
        cb_libsnn_ann.f_vth_set_and_norm=False

        cb_libsnn.f_vth_set_and_norm=True
        result = model.evaluate(ds_one_batch, callbacks=callbacks_test)
        cb_libsnn.f_vth_set_and_norm=False
        #cb_lib_snn.calibration.vth_set_and_norm(cb_libsnn)


    #calibration_ML=True
    calibration_ML=False

    if calibration_ML:

        last_batch = False
        #for idx_batch in range(num_batch_for_vth_search):
        #    images_one_batch, labels_one_batch = next(iter(train_ds))
        for idx_batch, (x,y) in enumerate(train_ds):

            if idx_batch == conf.calibration_num_batch - 1:
                last_batch = True

            #ds_one_batch_ann = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch)).take(1).cache()
            #ds_one_batch_snn = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch)).take(1).cache()
            ds_one_batch = tf.data.Dataset.from_tensors((x, y))

            # run - ann
            #callbacks_test[0].model_ann = model_ann
            result = model_ann.evaluate(ds_one_batch, callbacks=callbacks_test_ann)

            #ds_one_batch = tf.data.Dataset.from_tensors((images_one_batch, labels_one_batch))

            # run - snn
            callbacks_test[0].run_for_calibration_ML = True
            if last_batch:
                callbacks_test[0].calibration_bias = True

            result = model.evaluate(ds_one_batch, callbacks=callbacks_test)

            callbacks_test[0].run_for_calibration_ML = False
            callbacks_test[0].calibration_bias = False

            #idx_batch = idx_batch+1
            #if idx_batch == num_batch_for_vth_search-1:
            if last_batch:
                break

            #assert False


        #cb_lib_snn.calibration.vth_set_and_norm(cb_libsnn)

        #
        #if not vth_search:
        #lib_snn.calibration.vth_set_and_norm(cb_libsnn)
        #cb_libsnn.calibration.calibration_bias_set(cb_libsnn)

    ####
    # run - test dataset
    #
    #lib_snn.sim.set_for_visual_debug(True)
    result = model.evaluate(test_ds, callbacks=callbacks_test)
    #lib_snn.sim.set_for_visual_debug(False)

    #result = model.evaluate(test_ds)
    # result = model.predict(test_ds)

    print(result)

    #
    exp_act_dist=True

    if exp_act_dist and conf.nn_mode=='ANN':
        fig = glb_plot

        for layer in model.layers_w_kernel:

            axe = fig.axes.flatten()[layer.depth]
            act = layer.record_output
            act = tf.reshape(act,-1)


            #(n, bins, patches) = axe.hist(act,bins=bins)
            (n, bins, patches) = axe.hist(act)
            axe.axvline(x=np.max(act), color='b')
            #axe.set_ylim([0,n[10]])
            axe.set_title(layer.name)

        plt.show()


    #
#    ## compare control model
#    compare_control_snn_model = True
#    if compare_control_snn_model:
#
#        cb_libsnn_ctrl = lib_snn.callbacks.SNNLIB(conf, path_model, test_ds_num)
#        cb_libsnn_ctrl.run_for_calibration = True
#
#
#
#        #if dnn_snn_compare:
#        model_ann.evaluate(test_ds)
#
#        lib_snn.proc.dnn_snn_compare_func(cb_libsnn)

    #
    #for layer in model.layers_w_neuron:
    #    print('{} - {}'.format(layer.name,tf.reduce_sum(layer.act.spike_count_int)))

    # ANN for comparison

    #print(result_ann)

#
if False:
    zeros_input = tf.zeros([1,32,32,3])
    zeros_output = tf.zeros([1,10])
    #zeros_output = tf.constant([0,0,0,0,0,0,0,0,1,0])
    result = model.evaluate(x=zeros_input,y=zeros_output,callbacks=callbacks_test)
    #result = model.evaluate(test_ds,callbacks=callbacks_test)
    print(result)

    #for layer in model.layers:
    #    if hasattr(layer,'record_output'):
    #        print('{} - {}'.format(layer.name,tf.reduce_mean(layer.record_output)))

# debug - compare activation
if False:
    result_ann = model_ann.evaluate(test_ds, callbacks=callbacks_test_ann)

    for layer in model.layers:
        if hasattr(layer,'record_output'):
            snn = layer.record_output
            ann = model_ann.get_layer(layer.name).record_output

            #assert snn == ann
            if snn is not None:
                if tf.reduce_mean(snn) != tf.reduce_mean(ann):
                    print(ann)
                    print(layer.name)
                    print(tf.reduce_mean(snn))
                    print(tf.reduce_mean(ann))
                    assert False



#if __name__=="__main__":
#    # logging.set_verbosity(logging.INfO)
#    config.configurations()
#    app.run(main)