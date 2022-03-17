import torch
import torch.nn.functional as func

from utils import map_label


def linear_warmup(loss_warmup_params, epoch):
    alpha = loss_warmup_params['factor'] * (epoch - loss_warmup_params['start_epoch']) / \
            (loss_warmup_params['end_epoch'] - loss_warmup_params['start_epoch'])

    # Start to linearly increase the scale after start epoch and the maximum is defined by the factor.
    scale = min(max(alpha, 0), loss_warmup_params['factor'])

    return scale


def ae_loss(vis_recon, attr_recon, img, attr, negative_classes, label, recon_criterion,
            cross_recon, latent_var):
    # Reconstruction loss
    img_from_img_loss = recon_criterion(vis_recon[0], img).sum()

    # Among all the attributes, reconstruct attributes that only correspond to the images
    attr_from_attr_loss = recon_criterion(attr_recon[0][label, :], attr[label, :]).sum()
    recon_loss = img_from_img_loss + attr_from_attr_loss

    (z_from_img, z_from_attr) = latent_var

    # Cross Entropy for clustering loss
    negative_z_from_attr = z_from_attr[negative_classes, :]
    dist_img_attr = torch.zeros([z_from_img.shape[0], negative_z_from_attr.shape[0]]).to(z_from_img.device)
    mapped_label = map_label(label, negative_classes).to(label.device)

    # dist_img_attr = torch.zeros([z_from_img.shape[0], z_from_attr.shape[0]]).to(z_from_img.device)
    for i in range(img.shape[0]):
        tmp_z_img = z_from_img[i, :]
        dist_img_attr[i, :] = torch.sqrt(torch.sum((tmp_z_img - negative_z_from_attr) ** 2, dim=1))

    latent_dist_loss = func.cross_entropy(-1 * dist_img_attr, mapped_label, reduction='sum')

    # Cross-reconstruction loss
    if cross_recon:
        # Original cross-reconstruction error
        img_from_attr_loss = recon_criterion(vis_recon[1][label, :], img).sum()
        attr_from_img_loss = recon_criterion(attr_recon[1], attr[label, :]).sum()
        cross_recon_loss = img_from_attr_loss + attr_from_img_loss
    else:
        cross_recon_loss = 0

    return recon_loss, cross_recon_loss, latent_dist_loss


def compute_per_class_acc_gzsl(labels, predicted_labels, target_classes):
    per_class_accuracies = torch.zeros(target_classes.shape[0])

    for i in range(target_classes.shape[0]):
        is_class = (labels == target_classes[i])

        if is_class.sum() == 0:
            per_class_accuracies[i] = 0
        else:
            per_class_accuracies[i] = (predicted_labels[is_class] == labels[is_class]).sum().float() / is_class.sum()

    return per_class_accuracies.mean()


def ae_train_step(ae, optimizer, ae_train_loader, aux_data, negative_classes, recon_criterion,
                  hyperparameters, epoch, logger):
    ae.train()
    avg_loss_collect = torch.zeros([4, ])
    alpha = hyperparameters['alpha'][hyperparameters['dataset']]

    for batch_idx, (img, _, label) in enumerate(ae_train_loader):
        ae.zero_grad()
        optimizer.zero_grad()

        vis_recon, attr_recon, z_from_img, z_from_attr = ae(img, aux_data)

        # Loss calculation
        (recon_loss, cross_recon_loss, latent_dist_loss) = ae_loss(
            vis_recon, attr_recon, img, aux_data, negative_classes, label, recon_criterion,
            hyperparameters['cross_recon'], (z_from_img, z_from_attr))

        loss = recon_loss + cross_recon_loss + alpha * latent_dist_loss

        # Model update
        loss.backward()
        optimizer.step()

        loss_collect = [loss, recon_loss, cross_recon_loss, latent_dist_loss]
        avg_loss_collect += (torch.FloatTensor(loss_collect) * img.shape[0])

    avg_loss_collect = avg_loss_collect / len(ae_train_loader.dataset)

    logger.info('[Train] [Epoch %d / %d] Loss %.3f Recon_loss %.3f Cross_recon %.3f Latent_dist %.3f'
                % (epoch, hyperparameters['ae_train_epochs'], avg_loss_collect[0], avg_loss_collect[1],
                   avg_loss_collect[2], avg_loss_collect[3]))

    return avg_loss_collect


def ae_val_step(ae, ae_val_loader, aux_data, negative_classes, recon_criterion,
                hyperparameters, epoch, logger):
    ae.eval()
    alpha = hyperparameters['alpha'][hyperparameters['dataset']]
    avg_loss_collect = torch.zeros([4, ])
    with torch.no_grad():
        for batch_idx, (img, _, label) in enumerate(ae_val_loader):
            ae.zero_grad()

            vis_recon, attr_recon, z_from_img, z_from_attr = ae(img, aux_data)

            # Loss calculation
            (recon_loss, cross_recon_loss, latent_dist_loss) = ae_loss(
                vis_recon, attr_recon, img, aux_data, negative_classes, label, recon_criterion,
                hyperparameters['cross_recon'], (z_from_img, z_from_attr))

            loss = recon_loss + cross_recon_loss + alpha * latent_dist_loss

            loss_collect = [loss, recon_loss, cross_recon_loss, latent_dist_loss]
            avg_loss_collect += (torch.FloatTensor(loss_collect) * img.shape[0])

    avg_loss_collect = avg_loss_collect / len(ae_val_loader.dataset)
    logger.info('[Val] [Epoch %d / %d] Loss %.3f Recon_loss %.3f Cross_recon %.3f Latent_dist %.3f'
                % (epoch, hyperparameters['ae_train_epochs'], avg_loss_collect[0], avg_loss_collect[1],
                   avg_loss_collect[2], avg_loss_collect[3]))

    return avg_loss_collect


def clf_train_step(clf, optimizer, train_loader, criterion, target_classes, device, epoch, logger):
    clf.train()

    target_labels = torch.zeros([len(train_loader.dataset), ], dtype=torch.int64).to(device)
    predicted_labels = torch.zeros([len(train_loader.dataset), ], dtype=torch.int64).to(device)

    cnt = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        clf.zero_grad()

        optimizer.zero_grad()

        # Forward propagation
        output = clf(features)

        # Loss calculation
        loss = criterion(output, labels)

        # Model update
        loss.backward()
        optimizer.step()

        target_labels[cnt: cnt + features.shape[0]] = labels
        predicted_labels[cnt: cnt + features.shape[0]] = torch.argmax(output, 1)
        cnt += features.shape[0]

    acc = compute_per_class_acc_gzsl(target_labels, predicted_labels, target_classes)

    logger.info('[CLF Train] Epoch %d Loss %.3f Acc %.4f ', epoch, loss, acc)


def clf_eval_step(clf, eval_loader, target_classes, device):
    clf.eval()
    clf.zero_grad()

    prediction = torch.zeros([len(eval_loader.dataset), ], dtype=torch.int64).to(device)
    cnt = 0
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(eval_loader):
            # Forward propagation
            output = clf(features)

            prediction[cnt: cnt + features.shape[0]] = target_classes[torch.argmax(output, 1)]
            cnt += features.shape[0]

        acc = compute_per_class_acc_gzsl(eval_loader.dataset.labels, prediction, target_classes)

    return acc, prediction
