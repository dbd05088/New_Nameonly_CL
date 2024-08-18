import os
import numpy as np

exp_dirs = os.listdir(".")
seeds = [1,2,3]
print(f"{'A_avg':>62} \t\t\t A_last")
for exp_dir in exp_dirs:
    last_accuracy = []
    average_accuracy = []
    for seed in seeds:
        accuracy = []
        try:
            log_file = open(os.path.join(f"{exp_dir}/seed{seed}/round_None.log"), 'r')
            curr_task = 1
            curr_eval_results = []
            for line in log_file.readlines():
                if "curr_task" in line:
                    if int(line.split()[9]) > curr_task:  #== line.split()[12]
                        accuracy.append(np.mean(curr_eval_results))
                        curr_task = int(line.split()[9])
                        curr_eval_results = []
                    curr_eval_results.append(float(line.split()[-2])*100)
            accuracy.append(np.mean(curr_eval_results))
            last_accuracy.append(accuracy[-1])
            average_accuracy.append(np.mean(accuracy))
        except:
            pass
    print(f"{exp_dir:<50} \t {np.mean(average_accuracy):.2f}/{np.std(average_accuracy):.2f} \t\t {np.mean(last_accuracy):.2f}/{np.std(last_accuracy):.2f}")
        
        
        