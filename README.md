# New Nameonly CL

## How to download datasets
1. Setup gdrive for linux by following the instructions
    ```bash
    cd dataset
    mv ./gdrive /usr/local/bin
    gdrive account import gdrive_export-dbd05088_naver_com.tar
    ```

2. Now you can see the files in the google drive by using the following command
    ```bash
    gdrive files list
    ```

3. Download the dataset you want by using the following command. For example, if you want to download the dataset PACS, use the following command. This will automatically download the dataset and extract it to the current directory.
    ```bash
    ./PACS_gdrive.sh
    ```

## How to run the code

You can run the code by first:

1. Modify the `ex.sh` to set parameters for the experiment.
2. Modify the `pyfile` path in the `ex.sh` to the path of the python file you want to run.
3. Submit the job to the slurm by using the following command
    ```bash
    sbatch ex.sh
    ```

### Detailed explanation of the parameters in `ex.sh`
- `NOTE`: Name of the experiment. You should set this to the name that you can recognize the experiment.

    (IMPORTANT: Experiments with the same name will be overwritten)

- `MODE`: Method (e.g. `xder`, `er`, `bic`, `mir`)
- `SEEDS`: Between 1 and 5.
- `DATASET`: Name of the dataset (e.g. `cifar10`, `food101`, `cct`)
- `TYPES`: Training dataset (e.g. `ma`, `generated`, `web`)



### Results
You can see the results in the `results` directory.