#!/bin/bash


#epoch=$1
#epoch_start_train_tk=$2
#epoch_start_train_t_int=$3
#epoch_start_train_floor=$4
#epoch_start_train_clip_tw=$5
#epoch_start_loss_enc_spike=$6
#
#bypass_pr=$7
#bypass_target_epoch=$8
#
#cp_mode=$9
#f_training=$10




# training / inference mode
#f_training=True
f_training=False


#epoch=$1
ep=500
#ep=1

epoch_start_train_tk=$2
#ep_tk_arr=(0 250 500)
ep_tk_arr=(0)

#epoch_start_train_t_int=$3
ep_enc_int_arr=(0 250 500)
ep_enc_int_arr=(0)

#epoch_start_train_floor=$4
ep_enc_int_fl_arr=(0 250 500)
ep_enc_int_fl_arr=(0)

#epoch_start_train_clip_tw=$5
ep_dec_prun_arr=(0 250 500)
ep_dec_prun_arr=(0)

#epoch_start_loss_enc_spike=$6
ep_loss_enc_arr=(1000)

#bypass_pr=$7
bypass_pr_arr=(0 0.5 1)
bypass_pr_arr=(0 0.5)

#bypass_target_epoch=$8
bypass_tep_arr=(500)

# copy model - copy trained model
#cp_model=True
cp_model=False



for ((i_ep_tk=0;i_ep_tk<${#ep_tk_arr[@]};i_ep_tk++)) do
    ep_tk=${ep_tk_arr[$i_ep_tk]}

    for ((i_ep_enc_int=0;i_ep_enc_int<${#ep_enc_int_arr[@]};i_ep_enc_int++)) do
        ep_enc_int=${ep_enc_int_arr[$i_ep_enc_int]}

        for ((i_ep_enc_int_fl=0;i_ep_enc_int_fl<${#ep_enc_int_fl_arr[@]};i_ep_enc_int_fl++)) do
            ep_enc_int_fl=${ep_enc_int_fl_arr[$i_ep_enc_int_fl]}

            for ((i_ep_dec_prun=0;i_ep_dec_prun<${#ep_dec_prun_arr[@]};i_ep_dec_prun++)) do
                ep_dec_prun=${ep_dec_prun_arr[$i_ep_dec_prun]}

                for ((i_ep_loss_enc=0;i_ep_loss_enc<${#ep_loss_enc_arr[@]};i_ep_loss_enc++)) do
                    ep_loss_enc=${ep_loss_enc_arr[$i_ep_loss_enc]}

                    for ((i_bypass_pr=0;i_bypass_pr<${#bypass_pr_arr[@]};i_bypass_pr++)) do
                        bypass_pr=${bypass_pr_arr[$i_bypass_pr]}

                        for ((i_bypass_tep=0;i_bypass_tep<${#bypass_tep_arr[@]};i_bypass_tep++)) do
                            bypass_tep=${bypass_tep_arr[$i_bypass_tep]}

                             echo training_mode: ${f_training}
                             echo ep: ${ep}, tk: ${ep_tk}, int: ${ep_enc_int}, fl: ${ep_enc_int_fl}, cl: ${ep_dec_prun}, le: ${ep_loss_enc}, bp: ${bypass_pr}, bt: ${bypass_tep}

                            ./run_enc_dec_tk.sh ${ep} ${ep_tk} ${ep_enc_int} ${ep_enc_int_fl} ${ep_dec_prun} ${ep_loss_enc} ${bypass_pr} ${bypass_tep} ${cp_model} ${f_training}
                        done
                    done
                done
            done
        done
    done

done

