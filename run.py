import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import importlib
from torch.utils.data import DataLoader
from srs.layers.seframe import SEFrame
from srs.utils.data.load import read_dataset, AugmentedDataset
from srs.utils.data.augmentation import OfflineItemSimilarity, OnlineItemSimilarity
from srs.utils.train_runner_cl import TrainRunnerCL

# Utility functions and classes
def same_seeds(seed):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ArgumentParser(argparse.ArgumentParser):
    """Custom argument parser class."""
    def __init__(self, **kwargs):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
        self.optional = self._action_groups.pop()
        self.required = self.add_argument_group('required arguments')
        self._action_groups.append(self.optional)

    def add_argument(self, *args, **kwargs):
        if kwargs.get('required', False):
            return self.required.add_argument(*args, **kwargs)
        else:
            return super().add_argument(*args, **kwargs)

# Argument parsing
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    # Define arguments
    parser.add_argument('--model', required=True, help='the prediction model')
    parser.add_argument('--num-cluster', type=int, default=1024, help='number of cluster')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--augment-type', default='substitute', type=str, help="chosen from: mask, crop, reorder, substitute, insert")
    parser.add_argument('--similarity-method', default='ItemCF', type=str, help='chosen from: ItemTrans, ItemCF, ItemCF_IUF')
    parser.add_argument('--hybrid-epoch', default=0, type=int, help='offline:0, online:1, hybrid epoch')
    parser.add_argument(
        '--cluster-interval', type=int, default=3, help='number of cluster'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1, help='intra-session CL loss ratio'
    )
    parser.add_argument(
        '--beta', type=float, default=0.1, help='inter-session CL loss ratio'
    )
    parser.add_argument(
        '--noise-ratio', type=float, default=0.0, help='noise ratio during test'
    )
    parser.add_argument(
        '--dataset-dir', type=Path, required=True, help='the dataset set directory'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=128, help='the dimensionality of embeddings'
    )
    parser.add_argument(
        '--feat-drop', type=float, default=0.4, help='the dropout ratio for input features'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=1,
        help='the number of HGNN layers in the SSGE component',
    )
    parser.add_argument(
        '--num-neighbors',
        default='10',
        help='the number of neighbors to sample at each layer.'
        ' Give an integer if the number is the same for all layers.'
        ' Give a list of integers separated by commas if this number is different at different layers, e.g., 10,10,5'
    )
    parser.add_argument(
        '--model-args',
        type=str,
        default='{}',
        help="the extra arguments passed to the model's initializer."
        ' Will be evaluated as a dictionary.',
    )
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size')
    parser.add_argument(
        '--epochs', type=int, default=30, help='the maximum number of training epochs'
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='the weight decay for the optimizer',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='stop training if the performance does not improve in this number of consecutive epochs',
    )
    parser.add_argument(
        '--Ks',
        default='1,3,5,10,20',
        help='the values of K in evaluation metrics, separated by commas'
    )
    parser.add_argument(
        '--ignore-list',
        default='bias,batch_norm,activation',
        help='the names of parameters excluded from being regularized',
    )
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error'],
        default='debug',
        help='the log level',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='if log level is info or debug, print training information after every this number of iterations',
    )
    parser.add_argument(
        '--device', type=int, default=0, help='the index of GPU device (-1 for CPU)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='the number of processes for data loaders',
    )
    parser.add_argument(
        '--OTF',
        action='store_true',
        help='compute KG embeddings on the fly instead of precomputing them before inference to save memory',
    )
    parser.add_argument(
        '--model_drop', type=float, default=0.6, help='the dropout ratio for model'
    )
    # Parse and return arguments
    args = parser.parse_args()
    args.model_args = eval(args.model_args)
    args.num_neighbors = [int(x) for x in args.num_neighbors.split(',')]
    args.Ks = [int(K) for K in args.Ks.split(',')]
    args.ignore_list = [x.strip() for x in args.ignore_list.split(',') if x.strip() != '']
    return args

def setup_data_loaders(args,train_set,valid_set,test_set,batch_sampler):
    if args.hybrid_epoch == 0:
        collate_fn = args.CollateFn(similarity_model=args.offline_similarity_model, **args)
        collate_train = collate_fn.collate_train
        collate_test = collate_fn.collate_test

        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
            num_workers=args.num_workers,
        )
        cluster_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
            num_workers=args.num_workers,
        )

    elif args.hybrid_epoch == 1:
    
        collate_fn = args.CollateFn(similarity_model=args.online_similarity_model, **args)
        collate_train = collate_fn.collate_train
        collate_test = collate_fn.collate_test

        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
            num_workers=args.num_workers,
        )
        cluster_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
            num_workers=args.num_workers,
        )

    else:
        collate_fn1 = args.CollateFn(similarity_model=args.offline_similarity_model, **args)
        collate_train1 = collate_fn1.collate_train
        
        collate_test = collate_fn1.collate_test

        train_loader1 = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train1,
            num_workers=args.num_workers,
        )
        cluster_loader1 = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train1,
            num_workers=args.num_workers,
        ) 

        collate_fn2 = args.CollateFn(similarity_model=args.online_similarity_model, **args)
        collate_train2 = collate_fn2.collate_train

        train_loader2 = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train2,
            num_workers=args.num_workers,
        )
        cluster_loader2 = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_train2,
            num_workers=args.num_workers,
        ) 

        train_loader = [train_loader1, train_loader2]
        cluster_loader = [cluster_loader1, cluster_loader2]
    
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        collate_fn=collate_test,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate_test,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )
    
    return train_loader,cluster_loader,valid_loader,test_loader

# Main script
def main():
    args = parse_arguments()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(format='%(message)s', level=log_level)
    logging.debug(args)

    # Additional setup
    loca = time.strftime('%Y-%m-%d-%H-%M-%S')
    dataset_name = str(args.dataset_dir).split('/')[-1]
    args.log_file = os.path.join('./outputs/', str(loca) + '-' + dataset_name + '-' + str(args.model) + ".txt")

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # Import and configure the model
    module = importlib.import_module(f'srs.models.{args.model}')
    config = module.config
    for k, v in vars(args).items():
        config[k] = v
    args = config

    # Configure device
    args.gpu_id = args.device
    args.device = (torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}'))
    args.prepare_batch = args.prepare_batch_factory(args.device)

    # Dataset preparation
    logging.info(f'Reading dataset {args.dataset_dir}...')
    df_train, df_valid, df_test, stats = read_dataset(args.dataset_dir)
    # additional dataset preparation: social knowledge:
    if issubclass(args.Model, SEFrame):
        from srs.utils.data.load import (read_social_network, build_knowledge_graph)
        social_network = read_social_network(args.dataset_dir / 'edges.txt')
        args.knowledge_graph = build_knowledge_graph(df_train, social_network)
    args.num_users = getattr(stats, 'num_users', None)
    args.num_items = stats.num_items
    args.max_len = stats.max_len

    # Model instantiation
    model = args.Model(**args, **args.model_args)
    model = model.to(args.device)
    logging.debug(model)
    same_seeds(args.seed)
    args.model = model

    # DataLoader setup
    read_sid = args.Model.__name__ == 'DGRec'
    train_set = AugmentedDataset(df_train, read_sid)
    cluster_set = AugmentedDataset(df_train, read_sid) 
    valid_set = AugmentedDataset(df_valid, read_sid)
    test_set = AugmentedDataset(df_test, read_sid)
    
    logging.debug('using batch sampler')
    batch_sampler = config.BatchSampler(
        train_set, batch_size=args.batch_size, drop_last=True, seed=0
    )

    # offline:
    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_path = os.path.join(dataset_name+'_'+args.similarity_method+'_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(df=df_train, **args)
    args.offline_similarity_model = offline_similarity_model

    # # -----------   online based on shared item embedding for item similarity --------- #
    online_similarity_model = OnlineItemSimilarity(item_size=args.num_items, **args)
    args.online_similarity_model = online_similarity_model

    # (additional DataLoader setup)
    train_loader,cluster_loader,valid_loader, test_loader = setup_data_loaders(args,train_set,valid_set,test_set,batch_sampler)
    
    # Training
    runner = TrainRunnerCL(train_loader, valid_loader, test_loader, cluster_loader, **args)
    logging.info('Start training')
    runner.train(args.epochs, log_interval=args.log_interval)
    

# Run the main script
if __name__ == "__main__":
    main()
