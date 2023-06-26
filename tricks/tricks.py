import os.path as osp
import subprocess
import argparse

def main(args):
    batch_size = args.k * args.p
    dataset = args.d
    
    trick_number = args.tricks

    k = args.k
    epochs = args.epochs
    logs_dir = args.logs_dir

    # command to execute triplet loss
    command = "python triplet_loss.py -d {} -b {} -t {} --num-instances {} -j 2 -a resnet50 --logs-dir {} --epochs {} --combine-trainval".format(dataset, batch_size, trick_number, k, logs_dir, epochs)
    #command = "python triplet_loss.py -d {} -b {} -t {} --num-instances {} -j 2 -a resnet50 --logs-dir {} --epochs {} --combine-trainval --evaluate --cross_domain --resume logs/model_best.pth.tar".format(dataset, batch_size, trick_number, k, logs_dir, epochs)
    #command = "python triplet_loss.py -d {} -b {} -t {} --num-instances {} -j 2 -a resnet50 --logs-dir {} --epochs {} --combine-trainval --evaluate --re_ranking --resume logs/model_best.pth.tar".format(dataset, batch_size, trick_number, k, logs_dir, epochs)
    #command = "python triplet_loss.py -d {} -b {} -t {} --num-instances {} -j 2 -a resnet50 --logs-dir {} --epochs {} --combine-trainval --resume logs/checkpoint.pth.tar".format(dataset, batch_size, trick_number, k, logs_dir, epochs)
    subprocess.run(command, shell = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Bag of Tricks and A Strong Baseline for Deep Person Re-identification")

    # number of tricks to be executed
    parser.add_argument("--tricks", type = int, default = 0, help = "number of tricks executed starting from the baseline")

    # dataset to be used
    parser.add_argument("-d", type = str, default = "market1501", help = "dataset to be used for training and test")

    # k = number of images per identity in a batch
    parser.add_argument("-k", type = int, default = 16, help = "number of images per identity in a batch")

    # p = number of identities in a batch
    parser.add_argument("-p", type = int, default = 4, help = "number of identities in a batch")

    # number of epochs
    parser.add_argument("--epochs", type = int, default = 120, help = "number of epochs")

    # logs file
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument("--data-dir", type = str, metavar = "PATH", default = osp.join(working_dir, "data"))
    parser.add_argument("--logs-dir", type = str, metavar = "PATH", default = osp.join(working_dir, "logs"), help = "where to save logs files")

    main(parser.parse_args())