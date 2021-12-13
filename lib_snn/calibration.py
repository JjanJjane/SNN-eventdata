
import numpy as np
import collections
import csv
import os

import tensorflow as tf

import tensorflow_probability as tfp

import matplotlib.pyplot as plt

import lib_snn

#
def vth_calibration_stat(self):
    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    #stat = 'max'
    stat = 'max_999'
    #stat = 'max_99'
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        #print(self.dict_stat_r[l.name])
        #print(np.median(self.dict_stat_r[l.name]))
        #stat = self.dict_stat_r[l.name]

        #represent_stat = f_norm(self.dict_stat_r[l.name])
        represent_stat = self.norm_b[l.name]

        #
        #stat_r = np.where(stat_r==0, )
        #vth_cal = represent_stat / (stat_r+ 1e-10)
        #vth_cal = represent_stat / (stat_r)
        #self.vth_cal = stat_r / represent_stat
        vth_cal = stat_r / represent_stat
        #vth_cal = stat_r / represent_stat
        vth_cal = np.where(vth_cal==0, 1, vth_cal)
        #vth_cal = np.where(vth_cal>1, 1, vth_cal)
        #vth_cal = 0.1*(vth_cal*l.act.vth) + 0.9*(l.act.vth)
        vth_cal = 0.2*(vth_cal) + 0.8*(tf.reduce_mean(l.act.vth,axis=0))
        #vth_cal = 0.3*(vth_cal) + 0.7
        vth_cal_one_batch = vth_cal

        #vth_cal = np.expand_dims(vth_cal, axis=0)
        #vth_cal = np.broadcast_to(vth_cal, l.act.dim)
        vth_cal = tf.expand_dims(vth_cal, axis=0)
        vth_cal = tf.broadcast_to(vth_cal, l.act.dim)
        l.act.set_vth_init(vth_cal)


        if isinstance(l, lib_snn.layers.Conv2D):
            axis = [0, 1, 2]
        elif isinstance(l, lib_snn.layers.Dense):
            axis = [0]
        else:
            assert False

        #vth_cal_w_comp = np.mean(vth_cal,axis=axis)
        vth_cal_w_comp = tf.reduce_mean(vth_cal_one_batch,axis=axis)

        #
        #vth_cal_w_comp = np.mean(stat_r,axis=[0,1])/represent_stat
        #vth_cal_w_comp = np.mean(stat_r,axis=(0, 1))/represent_stat
        # TODO: tmp
        #vth_cal_w_comp = np.mean(stat_r)/np.mean(represent_stat)
        #print(vth_cal_w_comp)
        #print(vth_cal_w_comp.shape)

        # weight compensation
        if idx_l != 0:
            #scale = prev_vth_cal
            l.kernel = l.kernel*scale_next_layer

        #prev_vth_cal = vth_cal_w_comp
        scale_next_layer = vth_cal_w_comp


#
#def vth_calibration(self,f_norm, stat):
def vth_calibration_manual(self):

    vth_cal = collections.OrderedDict()
    const = 1/1.3
    if False:
        vth_cal['conv1'] = 0.5
        vth_cal['conv1_1'] = 0.5
        vth_cal['conv2'] = 0.8
        vth_cal['conv2_1'] = 1.0
        vth_cal['conv3'] = 0.5
        vth_cal['conv3_1'] = 0.7
        vth_cal['conv3_2'] = 0.5
        vth_cal['conv4'] = 0.8
        vth_cal['conv4_1'] = 0.5
        vth_cal['conv4_2'] = 0.5
        vth_cal['conv5'] = 0.7
        vth_cal['conv5_1'] = 0.7
        vth_cal['conv5_2'] = 0.7
        vth_cal['fc1'] = 0.7
        vth_cal['fc2'] = 0.7
        vth_cal['predictions'] = 0.5
    else:
        vth_cal['conv1'] = const
        vth_cal['conv1_1'] = const
        vth_cal['conv2'] = const
        vth_cal['conv2_1'] = const
        vth_cal['conv3'] = const
        vth_cal['conv3_1'] = const
        vth_cal['conv3_2'] = const
        vth_cal['conv4'] = const
        vth_cal['conv4_1'] = const
        vth_cal['conv4_2'] = const
        vth_cal['conv5'] = const
        vth_cal['conv5_1'] = const
        vth_cal['conv5_2'] = const
        vth_cal['fc1'] = const
        vth_cal['fc2'] = const
        vth_cal['predictions'] = const

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        l.act.set_vth_init(vth_cal[l.name])


    # scale - vth
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if idx_l != 0:
            scale = prev_vth_cal
            l.kernel = l.kernel*scale

        prev_vth_cal = vth_cal[l.name]



# TODO: move
def read_stat(self,layer,stat):

    path_stat = os.path.join(self.path_model,self.conf.path_stat)

    key = layer.name + '_' + stat

    # f_name_stat = f_name_stat_pre+'_'+key
    f_name_stat = key
    f_name = os.path.join(path_stat, f_name_stat)
    f_stat = open(f_name, 'r')
    r_stat = csv.reader(f_stat)

    for row in r_stat:
        # self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
        stat_r = np.asarray(row, dtype=np.float32).reshape(layer.output_shape_fixed_batch[1:])

    return stat_r

#
def vth_toggle(self):

    stat = 'max_999'
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        #

        #
        # simple toggle
        #if True:
        if False:
            vth_toggle_init = self.conf.vth_toggle_init
            #vth_schedule = tf.stack([self.conf.vth_toggle_init, 2-self.conf.vth_toggle_init])

            # vth schedule update
            #l.act.vth_schedule = vth_schedule
            #l.act.set_vth_init(vth_schedule[0])

            if isinstance(l, lib_snn.layers.Conv2D):
                shape = l.act.vth.shape[1:4]
            elif isinstance(l, lib_snn.layers.Dense):
                shape = l.act.vth.shape[1]
            else:
                assert False

            a = tf.constant(vth_toggle_init,shape=shape)
            #b = tf.constant(2-vth_toggle_init,shape=shape)
            b = a/(2*a-1)       # harmonic mean

        #
        # stat based toggle
        #if False:
        if True:
            self.stat_r = read_stat(self,l,stat)
            stat_r = self.stat_r

            #represent_stat = f_norm(self.dict_stat_r[l.name])
            represent_stat = self.norm_b[l.name]

            vth_toggle_init = stat_r/self.norm_b[l.name]
            #vth_schedule = [vth_toggle_init, 2-vth_toggle_init]
            #a = vth_toggle_init*1.1

            #alpha = 0.9
            alpha = self.conf.vth_toggle_init
            a = (1-alpha)*vth_toggle_init+alpha
            b = a/(2*a-1)       # harmonic mean
            #b = 2-vth_toggle_init

            #b = vth_toggle_init
            #a = 2-vth_toggle_init

        #vth_schedule = np.stack([a,b],axis=-1)
        vth_schedule = tf.stack([a,b],axis=-1)
        vth_schedule = tf.reshape(vth_schedule,shape=[-1,2])


        # batch
        vth_schedule = tf.tile(vth_schedule,[l.act.vth.shape[0],1])

        # vth schedule update
        l.act.vth_schedule = vth_schedule
        #l.act.set_vth_init(vth_schedule[:,0])
        l.act.vth_toggle_init = vth_schedule[:,0]
        l.act.set_vth_init(l.act.vth_toggle_init)

        print('{} - vth toggle set done '.format(l.name))

        #assert False
        #vth_schedule_init = tf.tile(vth_schedule,[l.act.vth.shape[0],1])[:,0]
        #l.act.set_vth_init(vth_schedule_init)


        #l.act.set_vth_init(l.act.vth*self.conf.vth_toggle_init)
        #l.act.set_vth_init(l.act.vth*0.3)



#
def vth_calibration_old(self,f_norm, stat):

    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    stat = 'max'
    stat = 'max_999'
    #stat = 'max_99'
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        #print(self.dict_stat_r[l.name])
        #print(np.median(self.dict_stat_r[l.name]))
        #stat = self.dict_stat_r[l.name]

        #represent_stat = f_norm(self.dict_stat_r[l.name])
        represent_stat = self.norm_b[l.name]

        #
        vth_cal = represent_stat / (stat_r+ 1e-10)
        vth_cal = np.where(vth_cal>1, 1, vth_cal)
        vth_cal = np.expand_dims(vth_cal, axis=0)
        vth_cal = np.broadcast_to(vth_cal, l.act.dim)

        l.act.set_vth_init(vth_cal)


    #assert False

#
def bias_calibration(self):

    bias_cal = collections.OrderedDict()

    const = 1.3

    bias_cal['conv1'] = const
    bias_cal['conv1_1'] = const
    bias_cal['conv2'] = const
    bias_cal['conv2_1'] = const
    bias_cal['conv3'] = const
    bias_cal['conv3_1'] = const
    bias_cal['conv3_2'] = const
    bias_cal['conv4'] = const
    bias_cal['conv4_1'] = const
    bias_cal['conv4_2'] = const
    bias_cal['conv5'] = const
    bias_cal['conv5_1'] = const
    bias_cal['conv5_2'] = const
    bias_cal['fc1'] = const
    bias_cal['fc2'] = const
    bias_cal['predictions'] = const

    #
    for layer in self.model.layers_w_kernel:
        layer.bias = layer.bias*bias_cal[layer.name]


# weight calibration - resolve information bottleneck
def weight_calibration(self):
    #
    stat = None

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    #norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    #const = 0.95
    #const = 0.5
    #const = 0.6
    const = 0.7

    #
    self.cal=collections.OrderedDict()

    # layer-wise norm, max_90
    #if stat=='max_90':
    #if True:
    if False:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.3
        norm['conv2']   = 0.75
        norm['conv2_1'] = 0.9
        norm['conv3']   = 1.0
        norm['conv3_1'] = 0.9
        norm['conv3_2'] = 1.0
        norm['conv4']   = 1.0
        norm['conv4_1'] = 1.0
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 0.9
        norm['conv5_2'] = 0.4
        norm['fc1']     = 1.0
        norm['fc2']     = 0.4
        norm['predictions'] = 0.1

    # stat=='max_99', channel-wise
    #elif True:
    elif False:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.9
        norm['conv2']   = 0.8
        norm['conv2_1'] = 0.8
        norm['conv3']   = 1.0
        norm['conv3_1'] = 1.0
        norm['conv3_2'] = 0.9
        norm['conv4']   = 0.9
        norm['conv4_1'] = 0.9
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 1.0
        norm['conv5_2'] = 1.0
        norm['fc1']     = 1.0
        norm['fc2']     = 1.0
        norm['predictions'] = 1.0

    elif False:
    #elif True:
        #norm['conv1']   = const
        norm['conv1']   = 0.95
        #norm['conv1']   = 0.9
        #norm['conv1']   = [0.7]*64
        #norm['conv1'][5] = 0.5
        norm['conv1_1'] = const
        #norm['conv1_1'] = 0.5
        norm['conv2']   = const
        #norm['conv2']   = 0.5
        #norm['conv2_1'] = const
        norm['conv2_1'] = 0.9
        #norm['conv3']   = const
        norm['conv3']   = 0.8
        #norm['conv3_1'] = const
        norm['conv3_1'] = 0.8
        norm['conv3_2'] = const
        norm['conv4']   = const
        norm['conv4_1'] = const
        norm['conv4_2'] = const
        norm['conv5']   = const
        norm['conv5_1'] = const
        norm['conv5_2'] = const
        norm['fc1']     = const
        norm['fc2']     = const
        norm['predictions'] = const


    elif False:
    #elif True:
        #norm['conv1']   = const
        norm['conv1']   = 0.95
        #norm['conv1']   = 0.7
        #norm['conv1']   = [0.7]*64
        #norm['conv1'][5] = 0.5
        norm['conv1_1'] = const
        #norm['conv1_1'] = 0.5
        norm['conv2']   = const
        #norm['conv2']   = 0.5
        #norm['conv2_1'] = const
        norm['conv2_1'] = 0.9
        #norm['conv3']   = const
        norm['conv3']   = 0.8
        #norm['conv3_1'] = const
        norm['conv3_1'] = 0.8
        norm['conv3_2'] = const
        norm['conv4']   = const
        norm['conv4_1'] = const
        norm['conv4_2'] = const
        norm['conv5']   = const
        norm['conv5_1'] = const
        norm['conv5_2'] = const
        norm['fc1']     = const
        norm['fc2']     = const
        norm['predictions'] = const

    # current best - 1208
    #elif False:
    elif True:
        #for idx_l, l in enumerate(self.model.layers_w_kernel):
            #norm[l.name]=0.8


        depth_l = len(self.model.layers_w_kernel)
        a = 0.5
        #a = 0.8
        b = 1.0
        #b = 0.8
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            norm[l.name] = a + (1 - a) * (depth_l - idx_l) / (depth_l)
            norm[l.name] *= b
            # norm[l.name]=a*(depth_l-idx_l)/(depth_l)
    #elif True:



    elif False:
    #elif True:
        norm['conv1']   = 0.3
        norm['conv1_1'] = 0.9
        norm['conv2']   = 1.0
        norm['conv2_1'] = 0.8
        norm['conv3']   = 1.0
        norm['conv3_1'] = 1.0
        norm['conv3_2'] = 0.9
        norm['conv4']   = 0.9
        norm['conv4_1'] = 0.9
        norm['conv4_2'] = 1.0
        norm['conv5']   = 1.0
        norm['conv5_1'] = 1.0
        norm['conv5_2'] = 1.0
        norm['fc1']     = 1.0
        norm['fc2']     = 1.0
        norm['predictions'] = 1.0

    else:
        #
        path_stat = os.path.join(self.path_model,self.conf.path_stat)
        #stat = 'max_999'
        #stat = 'max'
        #stat = 'max_75'
        stat = 'median'
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            #print(l.name)
            key=l.name+'_'+stat

            #f_name_stat = f_name_stat_pre+'_'+key
            f_name_stat = key
            f_name=os.path.join(path_stat,f_name_stat)
            f_stat=open(f_name,'r')
            r_stat=csv.reader(f_stat)

            for row in r_stat:
                #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
                stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

            #norm[l.name] = np.median(stat_r)
            norm[l.name] = 1/np.max(stat_r)
            print(norm[l.name])

    #
    #norm_wc['conv1'] = norm[0]
    #norm_b_wc['conv1'] = norm[0]
    #
    #    norm_wc['conv1_1'] = norm[1]/norm[0]
    #    norm_b_wc['conv1_1'] = norm[1]
    #

    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            if idx_l == 0:
                norm_wc[l.name] = norm[l.name]
            else:
                #norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]
                norm_wc[l.name] = norm[l.name] / np.expand_dims(norm_b_wc[prev_layer_name],axis=0).T

            prev_layer_name = l.name
            norm_b_wc[l.name] = norm[l.name]

    for layer in self.model.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        if layer.name in norm_wc.keys():
            layer.kernel = layer.kernel / norm_wc[layer.name]
        if layer.name in norm_b_wc.keys():
            layer.bias = layer.bias / norm_b_wc[layer.name]

    for layer in self.model.layers_w_kernel:
        print(layer.name)
        print(norm_wc[layer.name])


# weight calibration - resolve information bottleneck
def weight_calibration_post(self):
    #
    stat = None

    norm_wc = collections.OrderedDict()
    norm_b_wc = collections.OrderedDict()

    #norm = [0.5, 0.5, 0.5, ]
    norm = collections.OrderedDict()

    #const = 0.95
    #const = 0.5
    #const = 0.6
    const = 0.7

    #
    self.cal=collections.OrderedDict()


    #if True:
    if False:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            norm[l.name]=1.0

        #norm['conv1']=0.9
        #norm['conv1_1']=0.8

    # act (DNN) / act (SNN)
    #elif True:
    elif False:

        error_level = 'layer'
        # error_level = 'channel'

        for idx_l, l in enumerate(self.model.layers_w_kernel):

            if idx_l == len(self.model.layers_w_kernel)-1:
                norm[l.name]=1.0
                continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            ann_act = self.model_ann.get_layer(l.name).record_output
            snn_act = l.act.spike_count_int/time

            if error_level == 'layer':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2, 3]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0, 1]
                else:
                    assert False
            elif error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0]
                else:
                    assert False
            else:
                assert False

            ann_act = tf.reduce_mean(ann_act,axis=axis)
            snn_act = tf.reduce_mean(snn_act,axis=axis)

            error_r = snn_act / ann_act

            norm[l.name] = error_r

    # firing rate -> to 1
    elif True:
    #elif False:

        error_level = 'layer'
        #error_level = 'channel'

        for idx_l, l in enumerate(self.model.layers_w_kernel):

            if idx_l == len(self.model.layers_w_kernel)-1:
                norm[l.name]=1.0
                continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            snn_act = l.act.spike_count_int/time

            #ann_act = self.model_ann.get_layer(l.name).record_output
            #ann_act_s = self.model_ann.get_layer(l.name).bias
            #ann_act_d =

            #print(norm_s)
            #assert False


            if error_level == 'layer':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2, 3]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0, 1]
                else:
                    assert False
            elif error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0]
                else:
                    assert False
            else:
                assert False


            #fire_r_m = tf.reduce_mean(snn_act,axis=axis)
            #fire_r_m = tf.reduce_max(snn_act,axis=axis)
            #fire_r_m = tfp.stats.percentile(snn_act,99.91,axis=axis)
            fire_r_m = tfp.stats.percentile(snn_act,99.9,axis=axis)
            fire_r_m = tf.where(fire_r_m==0,tf.ones(fire_r_m.shape),fire_r_m)

            #time_r = time/self.conf.time_step
            #fire_r_m = fire_r_m/time_r

            norm[l.name] = fire_r_m

            #
            print('{} - t bias en: {}'.format(l.name,tf.reduce_mean(l.bias_en_time)))

    #elif True:
    elif False:
        stat='max_999'
        #stat='max'

        #error_level = 'layer'
        error_level = 'channel'

        for idx_l, l in enumerate(self.model.layers_w_kernel):

            if idx_l == len(self.model.layers_w_kernel)-1:
                norm[l.name]=1.0
                continue

            if self.conf.bias_control:
                time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
            else:
                time = self.conf.time_step

            self.stat_r = read_stat(self, l, stat)
            stat_r = self.stat_r

            # represent_stat = f_norm(self.dict_stat_r[l.name])
            #represent_stat = self.norm_b[l.name]

            norm_sat = stat_r / self.norm_b[l.name]
            norm_sat_e = norm_sat - 1
            norm_sat_e = tf.where(norm_sat_e > 0, norm_sat, tf.zeros(norm_sat.shape))

            if error_level == 'layer':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2, 3]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0, 1]
                else:
                    assert False
            elif error_level == 'channel':
                if isinstance(l, lib_snn.layers.Conv2D):
                    axis = [0, 1, 2]
                elif isinstance(l, lib_snn.layers.Dense):
                    axis = [0]
                else:
                    assert False
            else:
                assert False

            #norm_sat_e = tf.reduce_mean(norm_sat_e,axis=axis)

            #
            ann_act = self.model_ann.get_layer(l.name).record_output
            ann_act_s = self.model_ann.get_layer(l.name).bias
            ann_act_d = ann_act - ann_act_s
            ann_act_r = ann_act_d / ann_act_s

            print(l.name)
            print('ann_act_r')
            print(ann_act_r)
            print(ann_act_s)

            bias_comp_sub = norm_sat_e/ann_act_r
            bias_comp_sub = tf.where(tf.equal(ann_act,0.0), tf.zeros(bias_comp_sub.shape), bias_comp_sub)
            bias_comp_sub = tf.reduce_mean(bias_comp_sub,axis=axis)

            print('bias_comp_sub')
            print(bias_comp_sub)

            l.bias -= bias_comp_sub*0.001


        return

    #if True:

    else:
        #
        path_stat = os.path.join(self.path_model,self.conf.path_stat)
        #stat = 'max_999'
        #stat = 'max'
        #stat = 'max_75'
        stat = 'median'
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            #print(l.name)
            key=l.name+'_'+stat

            #f_name_stat = f_name_stat_pre+'_'+key
            f_name_stat = key
            f_name=os.path.join(path_stat,f_name_stat)
            f_stat=open(f_name,'r')
            r_stat=csv.reader(f_stat)

            for row in r_stat:
                #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
                stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

            #norm[l.name] = np.median(stat_r)
            norm[l.name] = 1/np.max(stat_r)
            print(norm[l.name])

    #
    #norm_wc['conv1'] = norm[0]
    #norm_b_wc['conv1'] = norm[0]
    #
    #    norm_wc['conv1_1'] = norm[1]/norm[0]
    #    norm_b_wc['conv1_1'] = norm[1]
    #

    if 'VGG' in self.conf.model:
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            if idx_l == 0:
                norm_wc[l.name] = norm[l.name]
            else:
                #norm_wc[l.name] = norm[l.name]/norm[prev_layer_name]
                norm_wc[l.name] = norm[l.name] / np.expand_dims(norm_b_wc[prev_layer_name],axis=0).T

            prev_layer_name = l.name
            norm_b_wc[l.name] = norm[l.name]

    for layer in self.model.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        if layer.name in norm_wc.keys():
            layer.kernel = layer.kernel / norm_wc[layer.name]
        if layer.name in norm_b_wc.keys():
            layer.bias = layer.bias / norm_b_wc[layer.name]

    for layer in self.model.layers_w_kernel:
        print(layer.name)
        print(norm_wc[layer.name])



def weight_calibration_inv_vth(self):
    #
    # scale - inv. vth
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if idx_l != 0:
            scale = self.conf.n_init_vth
            l.kernel = l.kernel*scale


# TODO: move
def vmem_calibration(self):

    #
    path_stat = os.path.join(self.path_model,self.conf.path_stat)
    #stat = 'max_999'
    stat = 'max'
    #stat = 'max_90'
    #stat = 'max_50'
    for idx_l, l in enumerate(self.model.layers_w_kernel):
        #print(l.name)
        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat=open(f_name,'r')
        r_stat=csv.reader(f_stat)

        for row in r_stat:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            stat_r=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])
        stat_r_max = np.max(stat_r)


        #represent_stat = np.max(dict_stat_r[l.name])

        #
        vmem_cal_norm = stat_r / stat_r_max
        #print(stat_r_max)
        #print(stat_r.shape)
        #print(vmem_cal_norm.shape)
        #assert False
        #vmem_cal = 0.7*(1-vmem_cal_norm)*l.act.vth
        vmem_cal = 0.7*(1-vmem_cal_norm)*l.act.vth
        #vmem_cal = 0.7*np.power((1-vmem_cal_norm),2)*l.act.vth

        #vth_cal = np.where(vth_cal>1, 1, vth_cal)
        #vmem_cal = np.expand_dims(vmem_cal, axis=0)
        #vmem_cal = np.broadcast_to(vmem_cal, l.act.dim)

        #l.act.set_vth_init(vth_cal)

        #if l.name=='conv1':
        l.act.reset_vmem(vmem_cal)

    #vmem_cal_norm
    #conv1_n = self.model.get_layer('conv1').act
    #conv1_n.reset_vmem()


################################################
# reproduce previous work - ICML-21, ICLR-21
################################################

#
def bias_calibration_ICLR_21(self):
    print('bias_calibration_ICLR_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if isinstance(l, lib_snn.layers.Conv2D):
            axis = [0,1,2]
        elif isinstance(l, lib_snn.layers.Dense):
            axis = [0]
        else:
            assert False

        time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)

        vth_channel = tf.reduce_mean(l.act.vth,axis=axis)
        bias_comp = vth_channel/(2*time)

        l.bias += bias_comp

    print('- Done')

# light pipeline
def bias_calibration_ICML_21(self):
    #print('here to stop')
    #print('')
    print('bias_calibration_ICML_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        if isinstance(l, lib_snn.layers.Conv2D):
            axis = [0,1,2]
        elif isinstance(l, lib_snn.layers.Dense):
            #axis = [0,1]
            axis = [0]
        else:
            assert False

        ann_out = self.model.get_layer(l.name).record_output

        dnn_act_mean = tf.reduce_mean(ann_out, axis=axis)
        # snn_act_mean = self.conf.n_init_vth*tf.reduce_mean(snn_out,axis=axis)/self.conf.time_step

        #time=self.conf.time_step
        time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)

        if l.name == 'predictions':
            continue
            if self.conf.snn_output_type is 'SPIKE':
                snn_out = l.act.spike_count_int
            elif self.conf.snn_output_type is 'VMEM':
                snn_out = l.act.vmem
            else:
                assert False
            snn_out = tf.nn.softmax(snn_out/time)
            #assert False
        else:
            snn_out = l.act.spike_count_int/time

        snn_act_mean = self.conf.n_init_vth*tf.reduce_mean(snn_out,axis=axis)

        bias_comp = dnn_act_mean - snn_act_mean

        # test
        #bias_comp *= 2
        bias_comp *= 4  # T=128, WP+B-ML
        #bias_comp *= 5  #

        #bias_comp = (dnn_act_mean - snn_act_mean)/self.conf.time_step
        #bias_comp = dnn_act_mean - self.conf.n_init_vth*snn_act_mean/self.conf.time_step

        #print(l.name)
        #print(bias_comp)

        l.bias += bias_comp

    print('- Done')


# adv. pipeline
def vmem_calibration_ICML_21(self):
    print('vmem_calibration_ICML_21')

    for idx_l, l in enumerate(self.model.layers_w_kernel):
        l_ann = self.model.get_layer(l.name)

        if l.name == 'predictions':
            continue

        dnn_act = l_ann.record_output
        snn_act = l.act.spike_count_int

        if self.conf.bias_control:
            time = tf.cast(self.conf.time_step - tf.reduce_mean(l.bias_en_time), tf.float32)
        else:
            time=self.conf.time_step

        error = dnn_act - self.conf.n_init_vth*snn_act/time
        error = tf.reduce_mean(error,axis=0)
        vmem_comp = error*time
        #vmem_comp = error
        #vmem_comp = error*self.conf.time_step
        #vmem_comp = error*(self.conf.time_step*0.01)
        #vmem_comp = error
        #vmem_comp = tf.where(vmem_comp>0,vmem_comp,tf.zeros(vmem_comp.shape))
        vmem_comp = tf.expand_dims(vmem_comp,axis=0)
        vmem_comp = tf.broadcast_to(vmem_comp, l.act.dim)

        #print(l.name)
        #print(vmem_comp)

        l.act.set_vmem_init(vmem_comp)


        #print(l.name)
        #print(bias_comp)

        #l.bias += bias_comp

    print('- Done')


