import os
import random
import argparse
import logging
import time

import torch
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter

import models
from cfg import hyperparameters
from dataset import LoadDataset, ReconDataset
import engine


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data', help='Path that contains dataset folders')
parser.add_argument('--dataset', type=str, default='CUB',
                    help='Dataset used for training (e.g. CUB, SUN, AWA2, AWA1)')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--resume', action='store', type=str, default=None, help='Resume checkpoint directory')
parser.add_argument('--use_trainval', action='store_true')
parser.add_argument('--save_dir', type=str, default='./save', help='Folder to save checkpoints/logs')
parser.add_argument('--save_name', type=str, default='gating-ae', help='Save name for training.')
args = parser.parse_args()


def main():

    hyperparameters['dataset'] = args.dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparameters['device'] = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the save path
    save_path = os.path.join(args.save_dir, args.dataset + '_' + args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = os.path.join(save_path, 'logs')

    # Load the entire dataset
    dataset = LoadDataset(hyperparameters, args.data_path, use_trainval=args.use_trainval)

    # Data loaders for the autoencoder training
    ae_train_loader = torch.utils.data.DataLoader(ReconDataset(dataset.data['train_seen'], dataset.aux_data),
                                                  batch_size=hyperparameters['ae_batch_size'], shuffle=True)

    ae_val_loader = torch.utils.data.DataLoader(ReconDataset(dataset.data['test_seen'], dataset.aux_data),
                                                batch_size=hyperparameters['ae_batch_size'], shuffle=False)

    # Model definition
    ae = models.TwoStreamCAE(hyperparameters)
    ae = torch.nn.DataParallel(ae).to(device)

    # Optimizer definition
    ae_optimizer = optim.Adam(ae.parameters(), lr=hyperparameters['ae_lr'], betas=(0.9, 0.999),
                              eps=1e-08, weight_decay=0, amsgrad=True)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            ae.load_state_dict(checkpoint['ae_state_dict'])
            start_epoch = int(checkpoint['epoch'] + 1)
            print("=> loaded checkpoint '{}'".format(args.resume))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    # Define the loss criterion
    if hyperparameters['ae_loss'] == 'l1':
        recon_criterion = torch.nn.L1Loss(reduction='none')
    elif hyperparameters['ae_loss'] == 'l2':
        recon_criterion = torch.nn.MSELoss(reduction='none')

    # Define tags and a writer for logs
    loss_names = ['Loss', 'Recon_loss', 'Cross_recon', 'Latent_dist']
    writer = SummaryWriter(log_dir=log_dir)

    logger.info("*** Start training an autoencoder ***")
    start_time = time.time()

    for epoch in range(start_epoch, hyperparameters['ae_train_epochs']):
        # Training
        train_avg_losses = engine.ae_train_step(ae, ae_optimizer, ae_train_loader, dataset.aux_data,
                                                dataset.seenclasses, recon_criterion, hyperparameters, epoch, logger)
        # Validation
        val_avg_losses = engine.ae_val_step(ae, ae_val_loader, dataset.aux_data, dataset.seenclasses,
                                            recon_criterion, hyperparameters, epoch, logger)

        # Write logs
        for i, loss_name in enumerate(loss_names):
            writer.add_scalar('train/%s' % loss_name, train_avg_losses[i], epoch)
            writer.add_scalar('val/%s' % loss_name, val_avg_losses[i], epoch)

        if (epoch + 1) % 100 == 0:
            logger.info("*** Saving the autoencoder model ***")
            output_checkpoint = os.path.join(save_path, "checkpoint.pth")
            torch.save(
                {
                    "ae_state_dict": ae.state_dict(),
                    "ae_optimizer_state_dict": ae_optimizer.state_dict(),
                    "epoch": epoch
                },
                output_checkpoint,
            )

        with open(os.path.join(save_path, "hyperparameters.txt"), "w") as f:
            print(hyperparameters, file=f)

    print('Total processing time: %.4f seconds' % (time.time() - start_time))

if __name__ == '__main__':
    main()
