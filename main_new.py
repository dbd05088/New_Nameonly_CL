import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configuration import config
from utils.data_loader import get_test_datalist
from utils.data_loader import get_train_datalist

from utils.method_manager_new import select_method


def main():
    args = config.base_parser()
    num_samples = {'cifar10': 10000, 'PACS':1670, "OfficeHome":4357, "DomainNet":50872}
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}/{args.type_name}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}/{args.type_name}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/{args.type_name}/seed_{args.rnd_seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    #writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.rnd_seed}')

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)


    # get datalist
    print("args.dataset", args.dataset, "num_samples", num_samples[args.dataset])
    train_datalist, cls_dict, cls_addition = get_train_datalist(args.dataset, args.sigma, args.repeat, args.init_cls, args.rnd_seed, args.type_name)
    test_domain_name, test_datalists = get_test_datalist(args.dataset)
    samples_cnt = 0

    # Reduce datalist in Debug mode
    if args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]


    logger.info(f"Select a CIL method ({args.mode})")
    method = select_method(args, train_datalist, test_datalists, device)

    print("\n###flops###\n")
    #method.get_flops_parameter()

    eval_results = defaultdict(list)

    samples_cnt = 0
    task_id = 0

    for i, data in enumerate(train_datalist):
        eval_point = [int(point) for point in args.eval_point.split(" ")]
        # explicit task boundary for twf
        if samples_cnt in [0] + eval_point and args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l"]:
            method.online_before_task(task_id)
            task_id += 1

        samples_cnt += 1
        method.online_step(data, samples_cnt, args.n_worker)
        #if samples_cnt % args.eval_period == 0:
        if samples_cnt in eval_point or samples_cnt % args.eval_period == 0:
            avg_acc = []
            cls_acc = []
            avg_loss = []
            for domain_name, test_datalist  in zip(test_domain_name, test_datalists):
                sample_num, eval_dict = method.online_evaluate(domain_name, test_datalist, samples_cnt, 32, args.n_worker, cls_dict, cls_addition, data["time"])
                avg_acc.append(eval_dict['avg_acc'])
                avg_loss.append(eval_dict['avg_loss'])
                cls_acc.append(eval_dict['cls_acc'])
            method.report_test("Total", sample_num, np.mean(avg_loss), np.mean(avg_acc))
            eval_results["test_acc"].append(np.mean(avg_acc))
            eval_results["percls_acc"].append(np.mean(cls_acc))
            eval_results["data_cnt"].append(samples_cnt)
            if samples_cnt in eval_point:
                eval_results["avg_test_acc"].append(np.mean(avg_acc))
                method.report_test("Task", sample_num, np.mean(avg_loss), np.mean(avg_acc))
                
        if samples_cnt in eval_point and (args.mode in ["ewc", "memo", "xder", "afec"]) and samples_cnt != num_samples[args.dataset]:
            method.online_after_task()
        
    if eval_results["data_cnt"][-1] != samples_cnt:
        for domain_name, test_datalist  in zip(test_domain_name, test_datalists):
            sample_num, eval_dict = method.online_evaluate(domain_name, test_datalist, samples_cnt, 32, args.n_worker, cls_dict, cls_addition, data["time"])
            avg_acc.append(eval_dict['avg_acc'])
            avg_loss.append(eval_dict['avg_loss'])
            cls_acc.append(eval_dict['cls_acc'])
        method.report_test("Total", sample_num, np.mean(avg_loss), np.mean(avg_acc))
        eval_results["test_acc"].append(np.mean(avg_acc))
        eval_results["percls_acc"].append(np.mean(cls_acc))
        eval_results["data_cnt"].append(samples_cnt)
        if samples_cnt in eval_point:
            eval_results["avg_test_acc"].append(np.mean(avg_acc))
            method.report_test("Task", sample_num, np.mean(avg_loss), np.mean(avg_acc))

    A_last = eval_results["test_acc"][-1] #eval_dict['avg_acc']

    if args.mode == 'gdumb':
        eval_results = method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker, cls_dict, cls_addition)

    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_per_cls.npy', eval_results['percls_acc'])
    np.save(f'results/{args.dataset}/{args.note}/seed_{args.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])
    A_avg = np.mean(eval_results["avg_test_acc"])

    # KLR_avg = np.mean(method.knowledge_loss_rate[1:])
    # KGR_avg = np.mean(method.knowledge_gain_rate)
    KLR_avg = 0.0
    KGR_avg = 0.0

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc:6f} | A_last {A_last:6f} | A_avg {A_avg:6f} | KLR_avg {KLR_avg:6f} | KGR_avg {KGR_avg:6f} | Total FLOPs {method.total_flops:4f}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
