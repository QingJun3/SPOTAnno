from config import parser_add_main_args
from utils import seed_everything, MemoryQueue, entropy
from datasets import *
import datetime
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, variable_to_numpy, clear_output
from model import CLS, ProtoCLS, MLPTransXY
from tqdm import tqdm
from lib import sinkhorn, ubot_SCTD, adaptive_filling, entropy_loss
import ot
import pandas as pd
import numpy as np
from eval import eval, get_embedding

def main():
    # get args
    args = parser_add_main_args()
    
    # set seed
    seed_everything(args.seed)

    if args.dataset == 'Hubmap_CL':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Hubmap_CL_intra_data(args)
    elif args.dataset == 'Hubmap_SB':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Hubmap_SB_intra_data(args)
    elif args.dataset == 'Hyp':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Hyp_intra_data(args)
    elif args.dataset == 'Spe_Diabetes':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Spe_Diabetes_intra_data(args)
    elif args.dataset == 'Spe_WT':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Spe_WT_intra_data(args)
    elif args.dataset == 'Hubmap_CL_cross':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Hubmap_CL_cross_data(args)
    elif args.dataset == 'Hubmap_SB_cross':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Hubmap_SB_cross_data(args)
    elif args.dataset == 'Spe_Diabetes_cross':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Spe_Diabetes_cross_data(args)
    elif args.dataset == 'Spe_WT_cross':
        classes_set, in_dim, source_train_dl, target_train_dl, target_test_dl, target_initMQ_dl, source_test_dl = load_Spe_WT_cross_data(args)

    # set log
    log_path = "log/<dataset>/<now>_<name>"
    log_path = log_path.replace("<dataset>", args.dataset)
    now = datetime.datetime.now().strftime('%b%d_%H-%M')
    log_path = log_path.replace("<now>", now)
    log_path = log_path.replace("<name>", args.name)

    log_dir = f'{log_path}'
    logger = SummaryWriter(log_dir)

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    # define network architecture
    cls_output_dim = len(classes_set['source_classes'])
    feature_extractor = MLPTransXY(in_dim, args.hidden_dim)
    classifier = CLS(feature_extractor.output_dim, cls_output_dim, hidden_mlp=args.hidden_dim, feat_dim=args.feat_dim, temp=args.temp)
    cluster_head = ProtoCLS(args.feat_dim, args.K, temp=args.temp)

    feature_extractor = feature_extractor.to(device)
    classifier = classifier.to(device)
    cluster_head = cluster_head.to(device)

    optimizer_featex = optim.SGD(feature_extractor.parameters(), lr=args.lr*0.1, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
    optimizer_cls = optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)
    optimizer_cluhead = optim.SGD(cluster_head.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.sgd_momentum, nesterov=True)

    # learning rate decay
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.min_step)
    opt_sche_featex = OptimWithSheduler(optimizer_featex, scheduler)
    opt_sche_cls = OptimWithSheduler(optimizer_cls, scheduler)
    opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead, scheduler)

    # Memory queue init
    n_batch = int(args.MQ_size/args.batch_size)    
    memqueue = MemoryQueue(args.feat_dim, args.batch_size, n_batch, device, args.temp).to(device)
    cnt_i = 0
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
        while cnt_i < n_batch:
            for i, (im_target, _, x_target, y_target, id_target) in enumerate(target_initMQ_dl):
                im_target = im_target.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)
                id_target = id_target.to(device)
                feature_ex = feature_extractor(im_target, x_target, y_target)
                before_lincls_feat, after_lincls = classifier(feature_ex)
                memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
                cnt_i += 1
                if cnt_i > n_batch-1:
                    break
    
    total_steps = tqdm(range(args.min_step), desc='global step')
    global_step = 0
    beta = None

    while global_step < args.min_step:
        iters = zip(source_train_dl, target_train_dl)
        for minibatch_id, ((im_source, label_source, x_source, y_source, id_source), (im_target, _, x_target, y_target, id_target)) in enumerate(iters):
            label_source = label_source.to(device)
            im_source = im_source.to(device)
            x_source = x_source.to(device)
            y_source = y_source.to(device)

            im_target = im_target.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            feature_ex_s = feature_extractor.forward(im_source, x_source, y_source)
            feature_ex_t = feature_extractor.forward(im_target, x_target, y_target)

            before_lincls_feat_s, after_lincls_s = classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            after_cluhead_t = cluster_head(before_lincls_feat_t)

            # =====Reference Supervision=====
            criterion = nn.CrossEntropyLoss().to(device)
            loss_cls = criterion(after_lincls_s, label_source)

            # =====Novel Cell Type Discovery=====
            minibatch_size = norm_feat_t.size(0)

            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / args.temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(device)
            feat_mat2.masked_fill_(mask, -1/args.temp)

            nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target.to(device))
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(device)
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
                neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]

            neighbor_output = cluster_head(neighbor_norm_feat)

            # OT process
            S_tt = torch.cat([after_cluhead_t, neighbor_output], 0)
            S_tt *= args.temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]

            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
                sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_NCTD = args.lam_local * loss_local + args.lam_global * loss_global

            # =====Seen Cell Type Detection=====
            if global_step > 500:
                source_prototype = classifier.ProtoCLS.fc.weight
                if beta is None:
                    beta = ot.unif(source_prototype.size()[0])

                # fill input features with memory queue
                fill_size_uot = n_batch*args.batch_size
                mqfill_feat_t = memqueue.random_sample(fill_size_uot)
                ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
                full_size = ubot_feature_t.size(0)

                # Adaptive filling
                newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, args.gamma, beta, fill_size_uot, device)

                # UOT-based SCTD
                high_conf_label_id, high_conf_label, _, new_beta, k_weight, u_weight = ubot_SCTD(newsim, beta, fake_size=fake_size, device=device,
                                                                        fill_size=fill_size_uot, mode='minibatch')
                # adaptive update for marginal probability vector
                beta = args.mu*beta + (1-args.mu)*new_beta

                if high_conf_label_id.size(0) > 0:
                    loss_SCTD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
                else:
                    loss_SCTD = 0
                t_prediction = F.softmax(after_lincls_t, dim=1)
                loss_ent = args.lam_pe * entropy_loss(t_prediction, k_weight) - args.lam_ne * entropy_loss(t_prediction, u_weight)
            else:
                loss_SCTD = 0
                loss_ent = 0

            loss_all = loss_cls + args.lam_NCTD * loss_NCTD + args.lam_SCTD * loss_SCTD + args.lam_ent * loss_ent

            with OptimizerManager([opt_sche_featex, opt_sche_cls, opt_sche_cluhead]):
                loss_all.backward()

            classifier.ProtoCLS.weight_norm() # very important for proto-classifier
            cluster_head.weight_norm() # very important for proto-classifier
            memqueue.update_queue(norm_feat_t, id_target.to(device))
            global_step += 1
            total_steps.update()

            if global_step % args.log_interval == 0:
                counter = AccuracyCounter()
                counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(classes_set['source_classes']))), variable_to_numpy(after_lincls_s))
                acc_source = torch.tensor([counter.reportAccuracy()]).to(device)
                logger.add_scalar('loss_all', loss_all, global_step)
                logger.add_scalar('loss_cls', loss_cls, global_step)
                logger.add_scalar('loss_NCTD', loss_NCTD, global_step)
                logger.add_scalar('loss_SCTD', loss_SCTD, global_step)
                logger.add_scalar('acc_source', acc_source, global_step)

            if global_step > 500 and global_step % args.test_interval == 0:
                results = eval(feature_extractor, classifier, target_test_dl, classes_set, device, gamma=args.gamma, beta=beta)

                df = pd.DataFrame(results['report_1']).T
                df.to_csv(f"{log_dir}/report_1_{global_step}.csv")

                df = pd.DataFrame(results['report_2']).T
                df.to_csv(f"{log_dir}/report_2_{global_step}.csv")

                df = pd.DataFrame(results['report_3']).T
                df.to_csv(f"{log_dir}/report_3_{global_step}.csv")

                df = pd.DataFrame(results['report_4']).T
                df.to_csv(f"{log_dir}/report_4_{global_step}.csv")

                logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
                logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
                logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)

                s_feat, t_feat, s_gt, t_gt = get_embedding(feature_extractor, classifier, source_test_dl, target_test_dl, device)
                import scanpy as sc
                adata = sc.AnnData(np.concatenate((s_feat, t_feat), axis=0))
                adata.obs["CellType"] = np.concatenate((s_gt, t_gt), axis=0)
                adata.obs["Batch"] = ["training set"] * s_feat.shape[0] + ["test set"] * t_feat.shape[0]
                adata.write(f'{log_dir}/embedding_{global_step}.h5ad')

    results = eval(feature_extractor, classifier, target_test_dl, classes_set, device, gamma=args.gamma, beta=beta)
    
    acc_df = pd.DataFrame({
        "seen_acc": results['common_acc'],
        "novel_acc": results['novel_acc'],
        "overall_acc": results['overall_acc'],
    }, index=[0])

    acc_df.to_csv(f"{log_dir}/acc.csv")

if __name__ == '__main__':
    main()