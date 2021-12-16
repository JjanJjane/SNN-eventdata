
import tensorflow as tf
import lib_snn

from lib_snn.sim import glb


def model_builder(
    eager_mode, model_top, batch_size, image_shape, conf, include_top, load_weight, num_class, model_name, lmb, initial_channels,
    train_epoch, train_steps_per_epoch,
    opt, learning_rate,
    lr_schedule, step_decay_epoch,
    metric_accuracy, metric_accuracy_top5
):

    print('Model Builder - {}'.format(conf.nn_mode))
    glb.model_compile_done_reset()

    # model
    model_top = model_top(batch_size=batch_size, input_shape=image_shape, conf=conf, include_top=include_top,
                          weights=load_weight, classes=num_class, name=model_name, lmb=lmb,
                          initial_channels=initial_channels)


    # TODO: parameterize
    # lr schedule
    lr_schedule_first_decay_step = train_steps_per_epoch * 10  # in iteration

    #lr_schedule = hp_lr_schedule
    #train_epoch = hp_train_epoch
    #step_decay_epoch = hp_step_decay_epoch

    if lr_schedule == 'COS':
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(learning_rate, train_steps_per_epoch * train_epoch)
    elif lr_schedule == 'COSR':
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate, lr_schedule_first_decay_step)
    elif lr_schedule == 'STEP':
        learning_rate = lib_snn.optimizers.LRSchedule_step(learning_rate, train_steps_per_epoch * step_decay_epoch, 0.1)
    elif lr_schedule == 'STEP_WUP':
        learning_rate = lib_snn.optimizers.LRSchedule_step_wup(learning_rate, train_steps_per_epoch * 100, 0.1,
                                                               train_steps_per_epoch * 30)
    else:
        assert False

    # optimizer
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, name='SGD')
    else:
        assert False

    #model = model_top.model
    model = model_top

    # set layer nn_mode
    #model.set_layers_nn_mode()



    # dummy
    img_input = tf.keras.layers.Input(shape=image_shape, batch_size=batch_size)
    model(img_input)

    # compile
    model.compile(optimizer=optimizer,
                  # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[metric_accuracy, metric_accuracy_top5], run_eagerly=eager_mode)
                #metrics = [metric_accuracy, metric_accuracy_top5], run_eagerly = False)


    print('-- model compile done')
    glb.model_compile_done()

    return model