dataset=CelebA_Blond
command=$1
data_dir=$2
algorithm=$3
gpu_id=$4

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms ${algorithm}\
       --data_dir ${data_dir}\
       --command_launcher multi_gpu\
       --fixed_test_envs 2\
       --holdout_fraction 0.1\
       --n_hparams 20\
       --n_trials 3\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "sweep/${dataset}/outputs/run_all/${algorithm}"
