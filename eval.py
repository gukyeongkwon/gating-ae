import os
import random
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc

import models
from cfg import hyperparameters
from dataset import LoadDataset
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
parser.add_argument('--val_ae_resume', type=str, required=True,
                    help='The autoencoder trained only on the training set')
parser.add_argument('--val_clf_resume', type=str, required=True,
                    help='The seen expert trained only on the training set')
parser.add_argument('--trainval_ae_resume', type=str, required=True,
                    help='The autoencoder trained on the trainval set')
parser.add_argument('--trainval_clf_resume', type=str, required=True,
                    help='The seen expert trained on the trainval set')
args = parser.parse_args()


def compute_features(ae, dataset, betas, seen=True):

    if seen:
        vis_data = dataset.data['test_seen']['resnet_features']
    else:
        vis_data = dataset.data['test_unseen']['resnet_features']

    dist_features = torch.zeros([vis_data.shape[0], len(betas)])

    with torch.no_grad():
        vis_recon, attr_recon, z_from_img, z_from_attr = ae(vis_data, dataset.aux_data)

        seen_z_from_attr = z_from_attr[dataset.seenclasses, :]
        unseen_z_from_attr = z_from_attr[dataset.novelclasses, :]

        seen_vis_recon = vis_recon[1][dataset.seenclasses, :]
        unseen_vis_recon = vis_recon[1][dataset.novelclasses, :]

        for i in range(vis_data.shape[0]):
            vis_recon_seen_dist = F.l1_loss(vis_data[i, :].view(1, -1).repeat(dataset.ntrain_class, 1),
                                            seen_vis_recon, reduction='none').sum(dim=1)
            vis_recon_unseen_dist = F.l1_loss(vis_data[i, :].view(1, -1).repeat(dataset.ntest_class, 1),
                                              unseen_vis_recon, reduction='none').sum(dim=1)

            latent_seen_dist = torch.exp(torch.sqrt(
                torch.sum((z_from_img[i, :] - seen_z_from_attr) ** 2, dim=1))).detach()
            latent_unseen_dist = torch.exp(torch.sqrt(
                torch.sum((z_from_img[i, :] - unseen_z_from_attr) ** 2, dim=1))).detach()

            for idx, beta in enumerate(betas):
                if beta is None: # Only use latent distance features
                    dist_features[i, idx] = (latent_seen_dist).min() / (latent_unseen_dist).min()
                else: # Use both latent distance and reconstruction features.
                    dist_features[i, idx] = (vis_recon_seen_dist + beta * latent_seen_dist).min() / \
                                              (vis_recon_unseen_dist + beta * latent_unseen_dist).min()
            dist_features.detach()

    return dist_features


def search_beta(ae, dataset, betas):

    seen_dist_features = compute_features(ae, dataset, betas, seen=True)
    unseen_dist_features = compute_features(ae, dataset, betas, seen=False)

    label = np.concatenate((np.zeros([seen_dist_features.shape[0], ]),
                            np.ones([unseen_dist_features.shape[0], ])), axis=0)

    score = np.concatenate((seen_dist_features.detach().numpy(), unseen_dist_features.detach().numpy()), axis=0)

    auroc_result = np.zeros([len(betas)])
    for idx in range(len(betas)):
        fpr_auc, tpr_auc, _ = roc_curve(label, score[:, idx], pos_label=1)
        auroc_result[idx] = auc(fpr_auc, tpr_auc)

        logger.info('Beta @ %.2f AUROC result: %.4f', betas[idx], auroc_result[idx])

    best_beta = betas[np.argmax(auroc_result)]
    return best_beta, np.max(auroc_result)


def predict_with_taus(ae, dataset, beta, thresholds, seen_expert_pred, unseen_expert_pred=None, seen=True):
    if seen:
        vis_data = dataset.data['test_seen']['resnet_features']
    else:
        vis_data = dataset.data['test_unseen']['resnet_features']

    with torch.no_grad():
        vis_recon, attr_recon, z_from_img, z_from_attr = ae(vis_data, dataset.aux_data)

        seen_z_from_attr = z_from_attr[dataset.seenclasses, :]
        unseen_z_from_attr = z_from_attr[dataset.novelclasses, :]

        seen_vis_recon = vis_recon[1][dataset.seenclasses, :]
        unseen_vis_recon = vis_recon[1][dataset.novelclasses, :]

        prediction = torch.zeros([vis_data.shape[0], thresholds.shape[0]],
                                 dtype=torch.int64).to(seen_z_from_attr.device)

        for i in range(vis_data.shape[0]):
            vis_recon_seen_dist = F.l1_loss(vis_data[i, :].view(1, -1).repeat(dataset.ntrain_class, 1),
                                            seen_vis_recon, reduction='none').sum(dim=1)
            vis_recon_unseen_dist = F.l1_loss(vis_data[i, :].view(1, -1).repeat(dataset.ntest_class, 1),
                                              unseen_vis_recon, reduction='none').sum(dim=1)

            latent_seen_dist = torch.exp(torch.sqrt(
                torch.sum((z_from_img[i, :] - seen_z_from_attr) ** 2, dim=1))).detach()
            latent_unseen_dist = torch.exp(torch.sqrt(
                torch.sum((z_from_img[i, :] - unseen_z_from_attr) ** 2, dim=1))).detach()

            if beta is None:
                ratio = (latent_seen_dist).min() / (latent_unseen_dist).min()
            else:
                ratio = (vis_recon_seen_dist + beta * latent_seen_dist).min() / \
                        (vis_recon_unseen_dist + beta * latent_unseen_dist).min()

            for thre_idx, thre in enumerate(thresholds):
                if ratio < thre:
                    if seen:
                        prediction[i, thre_idx] = seen_expert_pred[i]
                    else:
                        # This prediction will be incorrect because of incorrect unseen class detection
                        prediction[i, thre_idx] = -1
                else:
                    if unseen_expert_pred is None:
                        # Use 1-NN classification
                        prediction[i, thre_idx] = dataset.novelclasses[latent_unseen_dist.topk(1, largest=False)[1]]
                    else:
                        if seen:
                            # This prediction will be incorrect because of incorrect unseen class detection
                            prediction[i, thre_idx] = -1
                        else:
                            prediction[i, thre_idx] = unseen_expert_pred[i]

    return prediction


def search_tau(ae, dataset, beta, taus, seen_expert_pred, unseen_expert_pred=None):

    seen_prediction = predict_with_taus(ae, dataset, beta, taus, seen_expert_pred,
                                        unseen_expert_pred=unseen_expert_pred, seen=True)
    unseen_prediction = predict_with_taus(ae, dataset, beta, taus, seen_expert_pred,
                                          unseen_expert_pred=unseen_expert_pred, seen=False)

    prediction = torch.cat([seen_prediction, unseen_prediction], dim=0)

    seen_label = dataset.data['test_seen']['labels']
    unseen_label = dataset.data['test_unseen']['labels']

    result = np.zeros([len(taus), 3])

    for tau_idx in range(len(taus)):

        acc_seen = engine.compute_per_class_acc_gzsl(seen_label, seen_prediction[:, tau_idx], dataset.seenclasses)
        acc_unseen = engine.compute_per_class_acc_gzsl(unseen_label, unseen_prediction[:, tau_idx],
                                                       dataset.novelclasses)

        if (acc_seen + acc_unseen) > 0:
            harmonic = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)
        else:
            harmonic = 0

        result[tau_idx, 0] = acc_seen
        result[tau_idx, 1] = acc_unseen
        result[tau_idx, 2] = harmonic

    best_tau_idx = np.argmax(result[:, 2])
    best_tau = taus[best_tau_idx]

    return best_tau, (result[best_tau_idx, 0], result[best_tau_idx, 1], result[best_tau_idx, 2]), prediction


def base_predict(ae, dataset, seen=True):
    if seen:
        vis_data = dataset.data['test_seen']['resnet_features']
    else:
        vis_data = dataset.data['test_unseen']['resnet_features']

    ae.zero_grad()

    with torch.no_grad():
        vis_recon, attr_recon, z_from_img, z_from_attr = ae(vis_data, dataset.aux_data)

        prediction = torch.zeros([vis_data.shape[0], 1], dtype=torch.int64).to(z_from_attr.device)

        for i in range(vis_data.shape[0]):
            latent_dist = torch.exp(torch.sqrt(torch.sum((z_from_img[i, :] - z_from_attr) ** 2,
                                                              dim=1))).detach()
            prediction[i, 0] = latent_dist.topk(1, largest=False)[1]

    return prediction


def main():
    hyperparameters['dataset'] = args.dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparameters['device'] = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #########################################################################################################
    # Find the best beta and tau using the validation set
    print('Find the best beta and tau using the validation set...')
    dataset = LoadDataset(hyperparameters, args.data_path, use_trainval=False)

    ae = models.TwoStreamCAE(hyperparameters)
    ae = torch.nn.DataParallel(ae).to(device)
    ae.eval()

    if args.val_ae_resume:
        if os.path.isfile(args.val_ae_resume):
            print("=> loading checkpoint '{}'".format(args.val_ae_resume))
            checkpoint = torch.load(args.val_ae_resume)
            ae.load_state_dict(checkpoint['ae_state_dict'])
            print("=> loaded checkpoint '{}'".format(args.val_ae_resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.val_ae_resume))
            exit()

    # Search for best beta
    coarse_betas = [i for i in range(10, 110, 10)]
    coarse_best_beta, course_best_auroc = search_beta(ae, dataset, coarse_betas)
    logger.info('\n[Coarse Search] Beta @ %.2f AUROC result: %.4f', coarse_best_beta, course_best_auroc)

    fine_betas = [i for i in range(coarse_best_beta - 5, coarse_best_beta + 5, 1)]
    fine_best_beta, fine_best_auroc = search_beta(ae, dataset, fine_betas)
    logger.info('\n[Fine Search] Beta @ %.2f AUROC result: %.4f', fine_best_beta, fine_best_auroc)

    # Search for the best tau
    if args.val_clf_resume:
        if os.path.isfile(args.val_clf_resume):
            clf = models.Classifier(dataset.data['train_seen']['resnet_features'].shape[1], len(dataset.seenclasses))
            clf =  torch.nn.DataParallel(clf).to(device)
            print("=> loading checkpoint '{}'".format(args.val_clf_resume))
            checkpoint = torch.load(args.val_clf_resume)
            clf.load_state_dict(checkpoint['clf_state_dict'])
            clf.eval()
            print("=> loaded checkpoint '{}'".format(args.val_clf_resume))
            with torch.no_grad():
                output = clf(dataset.data['test_seen']['resnet_features'])
                seen_expert_pred = dataset.seenclasses[torch.argmax(output, 1)]
        else:
            print("=> no checkpoint found at '{}'".format(args.val_clf_resume))
            exit()

    taus = np.arange(0.9, 0.955, 0.005)
    best_thre, best_acc, _ = search_tau(ae, dataset, fine_best_beta, taus, seen_expert_pred)

    logger.info('\n[Threshold Search] thre: @ %.3f, Seen: %.4f, Unseen: %.4f, Harmonic: %.4f',
                best_thre, best_acc[0], best_acc[1], best_acc[2])

    #########################################################################################################
    # Evaluation on test set
    print('Start evaluation on the test set...')
    dataset = LoadDataset(hyperparameters, args.data_path, use_trainval=True)

    if os.path.isfile(args.trainval_ae_resume):
        print("=> loading checkpoint '{}'".format(args.trainval_ae_resume))
        checkpoint = torch.load(args.trainval_ae_resume)
        ae.load_state_dict(checkpoint['ae_state_dict'])
        print("=> loaded checkpoint '{}'".format(args.trainval_ae_resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.trainval_ae_resume))
        exit()

    if os.path.isfile(args.trainval_clf_resume):
        clf = models.Classifier(dataset.data['train_seen']['resnet_features'].shape[1], len(dataset.seenclasses))
        clf = torch.nn.DataParallel(clf).to(device)
        print("=> loading checkpoint '{}'".format(args.trainval_clf_resume))
        checkpoint = torch.load(args.trainval_clf_resume)
        clf.load_state_dict(checkpoint['clf_state_dict'])
        clf.eval()
        print("=> loaded checkpoint '{}'".format(args.trainval_clf_resume))
        with torch.no_grad():
            output = clf(dataset.data['test_seen']['resnet_features'])
            seen_expert_pred = dataset.seenclasses[torch.argmax(output, 1)]
    else:
        print("=> no checkpoint found at '{}'".format(args.val_clf_resume))
        exit()

    _, best_test_acc, _ = search_tau(ae, dataset, fine_best_beta, np.array([best_thre]), seen_expert_pred)

    logger.info('\n[Test Set Result] Beta: %d, Thre: %.3f Seen: %.4f, Unseen: %.4f, Harmonic: %.4f',
                fine_best_beta, best_thre, best_test_acc[0], best_test_acc[1], best_test_acc[2])

if __name__ == '__main__':
    main()
