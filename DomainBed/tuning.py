import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a tuning")
    parser.add_argument('cmd', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--lr', type=float, default=6e-5)
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
    
    #Edit the gpu-id on your own (default is 0)
    os.system('bash ./sweep/{1}/tuning.sh {0} ../datasets/ 0 {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}'.format(args.cmd, args.dataset, args.algorithm, args.lr, args.batch_size, args.resnet_dropout, args.Gvariance, args.last_k_epoch, args.rsc_b_drop_factor, args.rsc_f_drop_factor, args.weight_decay, args.worst_case_p, args.irm_lambda, args.irm_penalty_anneal_iters))
