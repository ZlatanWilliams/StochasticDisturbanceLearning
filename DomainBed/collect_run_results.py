import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect the results")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    args = parser.parse_args()
        
    os.system('python -m domainbed.scripts.collect_results --input_dir sweep/{0}/outputs/run_all/{1}/ --dataset {0} --algorithm {1}'.format(args.dataset, args.algorithm))
