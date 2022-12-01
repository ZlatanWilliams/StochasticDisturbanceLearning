import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List the hyper-parameters searching results')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    args = parser.parse_args()
    
    os.system('python -u -m domainbed.scripts.list_top_hparams --input_dir sweep/{0}/outputs/run_all/{1}/ --dataset {0} --algorithm {1} --test_env {2}'.format(args.dataset, args.algorithm, args.test_env))
