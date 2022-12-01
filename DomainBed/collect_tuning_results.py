import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect the tuning results")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resnet_dropout', type=float, default=0.0)
    parser.add_argument('--Gvariance', type=float, default=0.1)
    parser.add_argument('--last_k_epoch', type=float, default=0.25)
    parser.add_argument('--rsc_b_drop_factor', type=float, default=0.25)
    parser.add_argument('--rsc_f_drop_factor', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--worst_case_p', type=float, default=1/3)
    parser.add_argument('--irm_lambda', type=float, default=100.0)
    parser.add_argument('--irm_penalty_anneal_iters', type=int, default=500)
    args = parser.parse_args()
    
    os.system('python -m domainbed.scripts.collect_results --input_dir sweep/{0}/outputs/tuning/{1}/lr_{2}/batch_{3}/rndropout_{4}/Gv_{5}/lkepoch_{6}/rscbdf_{7}/rscfdf_{8}/weightdecay_{9}/worstcp_{10} --dataset {0} --algorithm {1}'.format(args.dataset, args.algorithm, args.lr, args.batch_size, args.resnet_dropout, args.Gvariance, args.last_k_epoch, args.rsc_b_drop_factor, args.rsc_f_drop_factor, args.weight_decay, args.worst_case_p))
