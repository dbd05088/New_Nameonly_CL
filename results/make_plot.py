import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings(action='ignore')
dir = 'DomainNet'
if 'PACS' in dir:
    in_dis = ['final_test_ma']
    ood_dis = ['final_cartoon', 'final_art_painting', 'final_sketch']
elif 'cct' in dir:
    in_dis = ['in_test_ma']
    ood_dis = ['out_test_ma']
elif 'OfficeHome' in dir:
    in_dis = ['test_ma']
    ood_dis = ['Clipart', 'Art', 'Product']
elif 'DomainNet' in dir:
    in_dis = ['test_ma']
    ood_dis = ['painting', 'quickdraw', 'sketch', 'clipart', 'infograph']


target_exp_list = [
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/gen_fake', \
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/sdxl_fake', \
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/web2_fake',
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/RMD_classwise_temp_3',
        '/home/user/mjlee/New_Nameonly_CL/results/DomainNet/resnet18_er_DomainNet_iter2_mem200/RMD_classwise_temp_5'
        ]
label_list = ["generated", "sdxl", "web", "classwise_3", "classwise_5"]

target_exp_list = np.array(target_exp_list)[[0,1,2,3,4]]
label_list = np.array(label_list)[[0,1,2,3,4]]
seed = 1

if seed == 1:
    added_timing = []
elif seed == 2:
    added_timing = []
elif seed == 3:
    added_timing = []

added_timing = added_timing

def print_from_log(exp_name, in_dis, ood_dis):
    in_avg = []
    in_last = []
    ood_avg = []
    ood_last = []
    overall_avg = []
    overall_last = []
    f = open(f'{exp_name}/seed_{seed}.log', 'r')
    lines = f.readlines()
    in_dis_acc = []
    ood_dis_acc = []
    in_acc, ood_acc = [], []
    all_acc = []
    for line in lines[11:]:
        if 'Test' in line:
                domains = in_dis+ood_dis
                for domain in domains:
                    if domain in line and domain in in_dis:
                        # print(line)
                        in_acc.append(float(line.split(" ")[-4]))
                    if domain in line and domain in ood_dis:
                        ood_acc.append(float(line.split(" ")[-4]))
        if 'Total Test' in line:
            in_dis_acc.append(np.mean(in_acc))
            ood_dis_acc.append(np.mean(ood_acc))
            in_acc, ood_acc = [], []
            all_acc.append(float(line.split(" ")[-4]))
    return ood_dis_acc

# exp_list = sorted([dir + '/' + exp for exp in os.listdir(dir)])

for exp, label in zip(target_exp_list, label_list):
    ood_dis_acc = print_from_log(exp, in_dis, ood_dis)
    # plt.plot(range(len(ood_dis_acc)), savgol_filter(ood_dis_acc, 5, 3), label=label, linewidth=1.0)
    plt.plot(range(len(ood_dis_acc)), ood_dis_acc, label=label, linewidth=1.0)
for i in added_timing:
    plt.axvline(x=i/100, color='r', linestyle='--', linewidth=0.5)
plt.xlabel("# of iteration (X 100)", fontsize=15)
plt.ylabel("Out Accuracy", fontsize=15)
plt.legend()
plt.title("DomainNet_out")
plt.savefig('DomainNet_out.png')