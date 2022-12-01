dataset=OfficeHome
cmd=$1
data_dir=$2
gpu_id=$3
algorithm=$4
lr=$5
batch_size=$6
resnet_dropout=$7
Gvariance=$8
last_k_epoch=$9
rsc_b_drop_factor=${10}
rsc_f_drop_factor=${11}
weight_decay=${12}
worst_case_p=${13}

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${cmd}\
       --datasets ${dataset}\
       --algorithms ${algorithm}\
       --data_dir ${data_dir}\
       --command_launcher multi_gpu\
       --single_test_envs\
       --steps 5001 \
       --holdout_fraction 0.1\
       --n_hparams 1\
       --n_trials 3\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/outputs/tuning/${algorithm}/lr_${lr}/batch_${batch_size}/rndropout_${resnet_dropout}/Gv_${Gvariance}/lkepoch_${last_k_epoch}/rscbdf_${rsc_b_drop_factor}/rscfdf_${rsc_f_drop_factor}/weightdecay_${weight_decay}/worstcp_${worst_case_p}"
