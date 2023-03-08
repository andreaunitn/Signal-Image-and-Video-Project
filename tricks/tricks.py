import os.path as osp
import argparse
import os

def main(args):
    batch_size = args.k * args.p
    dataset = args.d

    k = args.k

    # command to execute triplet loss
    command = "python3 triplet_loss.py -d {} -b {} --num-instances {} -j 2 -a resnet50 --combine-trainval".format(dataset, batch_size, k)
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Bag of Tricks and A Strong Baseline for Deep Person Re-identification")

    # dataset to be used
    parser.add_argument("-d", type = str, default = "market1501", help = "dataset to be used for training and test")

    # k = number of images per identity in a batch
    parser.add_argument("-k", type = int, default = 4, help = "number of images per identity in a batch")

    # p = number of identities in a batch
    parser.add_argument("-p", type = int, default = 16, help = "number of identities in a batch")

    main(parser.parse_args())
