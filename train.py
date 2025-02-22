import time
import datetime
import logging
import torch
from tools.utils import AverageMeter
from torch import nn


def train_cal(config, epoch, model, model2, clothes_classifier, clothes_classifier2, clothes_classifier3, criterion_cla, criterion_pair,
              criterion_clothes, criterion_adv, criterion_mmd, criterion_parsing, optimizer, optimizer2, optimizer_cc, trainloader,
              pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_parsing_loss = AverageMeter()
    batch_mmd_loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    clothes_classifier.train()
    clothes_classifier2.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, parsing_label, unclo, clo) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        parsing_label = parsing_label.cuda().long()
        unclo = unclo.cuda()
        clo = clo.cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        s_noclo, y_noclo, y_clo, s_fore, y_fore, part_cls_score, unclo_s, unclo_f = model((imgs, unclo))
        clo = model2(clo)
        # outputs = classifier(y_noclo)
        # outputs2 = classifier2(y_fore)

        pred_clothes2 = clothes_classifier2(y_fore.detach())
        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes2, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            clothes_loss.backward()
            optimizer_cc.step()
        # Update the backbone
        new_pred_clothes2 = clothes_classifier2(y_fore)

        # Compute loss
        cla_loss = criterion_cla(s_noclo, pids) + criterion_cla(s_fore, pids)
        pair_loss = criterion_pair(y_noclo, pids)
        adv_loss = criterion_adv(new_pred_clothes2, clothes_ids, pos_mask)
        parsing_loss = criterion_parsing(part_cls_score, parsing_label)
        mmd_loss = criterion_mmd(y_noclo, unclo_f)
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + parsing_loss
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + parsing_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_parsing_loss.update(parsing_loss.item(), pids.size(0))
        batch_mmd_loss.update(mmd_loss.item(), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'CloLoss:{clo_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'ParsingLoss:{parsing_loss.avg:.4f} '
                'MMDLoss:{mmd_loss.avg:.4f} '
                .format(epoch + 1, batch_time=batch_time, data_time=data_time,
                        cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
                        clo_loss=batch_clo_loss, adv_loss=batch_adv_loss,
                        parsing_loss=batch_parsing_loss, mmd_loss=batch_mmd_loss))


def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair,
                          criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
        epoch + 1, batch_time=batch_time, data_time=data_time,
        cla_loss=batch_cla_loss, pair_loss=batch_pair_loss,
        adv_loss=batch_adv_loss, acc=corrects))
