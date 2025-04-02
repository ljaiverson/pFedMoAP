import argparse
import torch
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time

import copy
import numpy as np
from utils.fed_utils import average_weights, count_parameters

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pFedMoAP", help="model of aggregation, choose from: pFedMoAP (used with pFedMoAP), fedavg, fedprox, local(The last three are used with PromptFL)")
    parser.add_argument("--trainer", type=str, default="PFEDMOAP", help="name of trainer, choose from: CLIP (used with fedavg), PromptFL, PFEDMOAP")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--local_epochs', type=int, default=5, help="number of local epochs")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=1.0, help='the client sample ratio: C')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of single_step (learning rate decay multiplier)')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for training all samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir100', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    # domainnet, office
    parser.add_argument('--imbalance_train', default=False, help="is adding label skew to feature skew datasets")
    parser.add_argument('--split_client', default=False, help="is adding label skew to feature skew datasets and split one domain to multi clients")
    parser.add_argument('--num_domain', type=int, default=4, help="number of domain")

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")

    # parameters of pFedMoAP
    parser.add_argument("--num_experts", type=int, default=10, help="number of experts")
    parser.add_argument("--sparse_selection", type=str, default="nearest", choices=["nearest", "random"], help="type of expert selection, choose between random and nearest")
    parser.add_argument("--gating_heads", type=int, default=8, help="number of heads in gating network")
    parser.add_argument("--gating_embed_dim", type=int, default=128, help="number of heads in gating network")
    parser.add_argument("--lmbda", type=float, default=0.5, help="the coefficient of the local output loss")
    parser.add_argument("--scaling", type=float, default=10.0, help="the scaling factor in attention dot product for attention weights")

    # parameters of path
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--root", type=str, default="./dataset/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="./output/food101/pFedMoAP/rn50_16shots/prompt16/10users_10rounds_seed1", help="output directory")
    parser.add_argument("--config-file", type=str, default="./configs/trainers/PFEDMOAP/rn50.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/food101.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    args = parser.parse_args()
    return args


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.round:
        cfg.OPTIM.ROUND = args.round # global round
        
    if args.local_epochs:
        cfg.OPTIM.MAX_EPOCH = args.local_epochs

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = False  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for PFEDMOAP
    cfg.TRAINER.PFEDMOAP = CN()
    cfg.TRAINER.PFEDMOAP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PFEDMOAP.CSC = False  # class-specific context
    cfg.TRAINER.PFEDMOAP.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.PFEDMOAP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PFEDMOAP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PFEDMOAP.NUM_EXPERTS = min(args.num_experts, args.num_users) # number of experts
    cfg.TRAINER.PFEDMOAP.GATING_HEADS = args.gating_heads # number of heads in gating network
    cfg.TRAINER.PFEDMOAP.GATING_EMBED_DIM = args.gating_embed_dim # embedding dimension in gating network
    cfg.TRAINER.PFEDMOAP.LMBDA = args.lmbda # number of heads in gating network
    cfg.TRAINER.PFEDMOAP.SCALING = args.scaling # scaling of the distance matrix

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATASET.IMBALANCE_TRAIN = args.imbalance_train # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = args.split_client # is adding label skew to feature skew datasets and split one domain to multi clientss

    cfg.DATALOADER.TRAIN_X.N_DOMAIN = args.num_domain # number of domain
    
    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = args.local_epochs # local epoch
    cfg.OPTIM.GAMMA = args.gamma # gamma of single-step (learning rate decay multiplier)
    cfg.OPTIM.LR = args.lr #learning rate

    # cfg.OPTIMGATING = cfg.OPTIM.clone() # optimizer for pFedMoAP

    cfg.MODEL.BACKBONE.PRETRAINED = True

    cfg.TEST.NO_TEST = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    cfg.set_new_allowed(True)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    local_weights = [[] for i in range(args.num_users)]
    local_gatings = [{} for i in range(args.num_users)]
    local_prompts = [[] for i in range(args.num_users)]
    global_prompt = None

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model,"prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")
    count_parameters(local_trainer.model, "gating")

    datanumber_client = []
    if args.trainer == 'CLIP':
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    # global_trainer.before_train()
    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()

    def evaluate_trainer(results, mode="CLIP"):
        nonlocal global_time_list, global_test_acc_list, global_test_error_list, global_test_f1_list, global_epoch_list
        nonlocal start, max_epoch
        nonlocal cfg

        if mode == "CLIP" or mode == "local":
            condition = (epoch == max_epoch - 1)
        else:
            condition = (epoch >= 2)

        global_test_acc = []
        global_test_error = []
        global_test_f1 = []
        for k in range(len(results)):
            global_test_acc.append(results[k][0])
            global_test_error.append(results[k][1])
            global_test_f1.append(results[k][2])
        global_time_list.append(time.time() - start)
        global_test_acc_list.append(sum(global_test_acc)/len(global_test_acc))
        global_test_error_list.append(sum(global_test_error) / len(global_test_error))
        global_test_f1_list.append(sum(global_test_f1) / len(global_test_f1))
        global_epoch_list.append(epoch)
        print("Global test acc:", sum(global_test_acc)/len(global_test_acc))
        print("Global test error:", sum(global_test_error) / len(global_test_error))
        print("Global test macro_f1:", sum(global_test_f1) / len(global_test_f1))
        if (cfg.DATASET.NAME == "DomainNet" or cfg.DATASET.NAME == "Office") and condition and args.split_client:
            domains = {"DomainNet":["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
                       "Office":["amazon", "caltech", "dslr", "webcam"]}
            num_domains = len(domains[cfg.DATASET.NAME])
            num_clients_per_domain = args.num_users // num_domains
            print("Test acc of clients:", global_test_acc)
            for i in range(num_domains):
                accs = global_test_acc[i*num_clients_per_domain:(i+1)*num_clients_per_domain]
                print("Test acc of", domains[cfg.DATASET.NAME][i], np.mean(accs), "±", np.std(accs))
            print("Test acc of all",np.mean(global_test_acc),np.std(global_test_acc))
        print("------------local test finish-------------")

    for epoch in range(start_epoch, max_epoch):

        if args.trainer == 'CLIP':
            print("------------local test start-------------")
            results = []
            idxs_users = list(range(0, cfg.DATASET.USERS))
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            local_trainer.model.load_state_dict(global_weights)
            for idx in idxs_users:
                results.append(local_trainer.test(idx=idx))
            evaluate_trainer(results, mode=args.trainer)
            print("Round on server :", epoch)
            break

        elif args.model == "fedavg":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights, strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            local_trainer.model.load_state_dict(global_weights, strict=False)
            local_weights = [[] for i in range(args.num_users)] # release gpu memory
            for idx in all_users:
                results.append(local_trainer.test(idx=idx))
            evaluate_trainer(results, mode=args.model)
            print("Round on server :", epoch)

        elif args.model == "fedprox":
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            # idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights, strict=False)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, global_weight=global_weights, fedprox=True, mu=args.mu)
                local_weight = local_trainer.model.state_dict()
                local_weights[idx] = copy.deepcopy(local_weight)
            print("------------local train finish epoch:", epoch, "-------------")

            global_weights = average_weights(local_weights, idxs_users, datanumber_client)

            print("------------local test start-------------")
            results = []
            all_users = list(range(0, cfg.DATASET.USERS))
            local_trainer.model.load_state_dict(global_weights, strict=False)
            local_weights = [[] for i in range(args.num_users)] # release gpu memory
            for idx in all_users:
                results.append(local_trainer.test(idx=idx))
            evaluate_trainer(results, mode=args.model)
            print("Round on server :", epoch)

        elif args.model == "pFedMoAP":
            num_client_selected = int(args.frac * args.num_users)
            # assert num_client_selected >= args.num_experts, "Do not support less selected clients per round than number of experts!"
            m = max(num_client_selected, 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            results = [None for _ in range(cfg.DATASET.USERS)]
            all_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")

            for idx in idxs_users:
                # download
                if epoch == 0:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                else:
                    # ensuring training only the local prompts (no non-local) for the first time each client is selected
                    if local_prompts[idx] != []:
                        # not the first time
                        if local_gatings[idx] != []:
                            local_trainer.model.load_state_dict(local_gatings[idx], strict=False)
                        else:
                            local_trainer.model.load_state_dict(global_weights, strict=False)
                        # experts
                        selected_experts = local_trainer.sparse_selection(idx, local_prompts)
                        local_trainer.download_nonlocal_ctx([local_prompts[iii] for iii in selected_experts])
                    else:
                        # the first time
                        local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.model.load_state_dict({"prompt_learner.ctx": global_prompt}, strict=False)

                # train
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)

                # test selected clients for this round
                results[idx] = local_trainer.test(idx=idx)

                # upload
                local_weight = local_trainer.model.state_dict()
                if local_prompts[idx] != []:
                    local_gatings[idx] = {name: copy.deepcopy(local_weight[name]) for name in local_weight if 'gating' in name}  # gating dict
                local_prompts[idx] = copy.deepcopy(local_weight['prompt_learner.ctx'])  # prompts

            print("------------local train finish epoch:", epoch, "-------------")
            local_trainer.update_lr(["gating"])
            local_trainer.reset_distance_cache(update_indices=idxs_users)
            global_prompt = average_weights(local_prompts, idxs_users, datanumber_client, islist=True)

            print("------------local test start-------------")
            
            # test clients that are not selected
            for idx in all_users:
                if results[idx] is not None:
                    continue
                if local_gatings[idx] != []:
                    local_trainer.model.load_state_dict(local_gatings[idx], strict=False)
                    selected_experts = local_trainer.sparse_selection(idx, local_prompts)
                    local_trainer.download_nonlocal_ctx([local_prompts[iii] for iii in selected_experts])
                    local_trainer.model.load_ctx(local_prompts[idx])
                elif local_prompts[idx] != []:
                    local_trainer.model.load_ctx(local_prompts[idx])
                else:
                    local_trainer.model.load_ctx(global_prompt)
                            
                results[idx] = local_trainer.test(idx=idx)
            evaluate_trainer(results, mode=args.model)
            print("Round on server :", epoch)

        elif args.model == "local":
            # CoOp
            idxs_users = list(range(0, cfg.DATASET.USERS))
            print("idxs_users", idxs_users)
            print("------------local train start epoch:", epoch, "-------------")
            results = []
            for idx in idxs_users:
                local_trainer.model.load_state_dict(global_weights)
                local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                results.append(local_trainer.test(idx=idx))
            evaluate_trainer(results, mode=args.model)
            break
        
        else:
            raise NotImplementedError(f"Model '{args.model}' is not implemented.")
    
    for idx in idxs_users:
        local_trainer.fed_after_train()
    # global_trainer.fed_after_train()
    print("global_test_acc_list:",global_test_acc_list)
    print("maximum test acc:", max(global_test_acc_list))
    print("mean of acc:",np.mean(global_test_acc_list[-5:]))
    print("std of acc:",np.std(global_test_acc_list[-5:]))

if __name__ == "__main__":
    args = get_args()
    main(args)



