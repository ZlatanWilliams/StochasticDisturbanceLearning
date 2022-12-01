import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('cmd', choices=['launch', 'delete_incomplete', 'list'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    args = parser.parse_args()
    
    # Edit the gpu-id on your own (default is 0)
    os.system('bash sweep/{0}/run.sh {1} ../datasets {2} 0'.format(args.dataset, args.cmd, args.algorithm))
