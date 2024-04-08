import argparse

def parser_add_main_args():
    parser = argparse.ArgumentParser(description='Code for STOTAnno',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--gpu_index', type=str, default='1', help='')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--name', type=str, default='test')

    parser.add_argument('--dataset', type=str, default='Hubmap_CL', help='dataset')
    parser.add_argument('--class_balance', action='store_true', help='use class balance', default=True)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--batch_size', type=int, help='batch size', default=36)
    
    parser.add_argument('--K', type=int, help='K', default=50)
    parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
    parser.add_argument('--mu', type=float, help='mu', default=0.7)
    parser.add_argument('--temp', type=float, help='temp', default=0.1)
    parser.add_argument('--lam_NCTD', type=float, help='lam_NCTD', default=0.1)
    parser.add_argument('--lam_SCTD', type=float, help='lam_SCTD', default=0.01)
    parser.add_argument('--lam_ent', type=float, help='lam_ent', default=0.01)

    parser.add_argument('--lam_local', type=float, help='lam_local', default=10)
    parser.add_argument('--lam_global', type=float, help='lam_global', default=1)
    parser.add_argument('--lam_pe', type=float, help='lam_pe', default=0.01)
    parser.add_argument('--lam_ne', type=float, help='lam_ne', default=2)

    parser.add_argument('--MQ_size', type=int, help='MQ_size', default=5000)

    parser.add_argument('--min_step', type=int, help='min_step', default=10000)
    parser.add_argument('--lr', type=float, help='lr', default=0.01)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.0005)
    parser.add_argument('--sgd_momentum', type=float, help='sgd_momentum', default=0.9)

    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--root_dir', type=str, default="log/MLP_1020_baseline_B004_reg003_only_ce")

    parser.add_argument('--feat_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)

    args = parser.parse_args()
    return args