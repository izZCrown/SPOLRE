import argparse

def parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--adj_type', default='adj_conv', help='how to build A: adj_conv | adj_l1')

    return parser.parse_args()

def print_args(args):
    for k, v in vars(args).items():
        print(f'{k}: {v}')