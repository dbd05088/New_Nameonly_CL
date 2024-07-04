./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # New (0521)
# mkdir -p cifar10
# ./gdrive files download 1VxdSntW9y6tzzSD1aiG_YGaL3jzpJv4g #nc
# ./gdrive files download 16DTKBJjjzU4Z3jmiIYwgEPrXb4Hju3B4 #c
# ./gdrive files download 1jgqUvtGf7Ta1Uq-jMUm6vLmDgjaXzqzu #ma (updated - 0522)
# ./gdrive files download 1TdIXr9uwdCcjDPBmnD_VQ2qv30uNKyxQ #original test set
# tar -xf cifar10_ma.tar -C cifar10
# tar -xf cifar10_nc.tar -C cifar10
# tar -xf cifar10_c.tar -C cifar10
# tar -xf cifar10_original_test.tar -C cifar10
# rm cifar10_ma.tar
# rm cifar10_nc.tar
# rm cifar10_c.tar
# rm cifar10_original_test.tar


# # Base prompt
# gdrive files download 1YKSZtuh4qxl0QUAczZB9Q-EGyxVqj7ek
# tar -xf cifar10_sdxl_base.tar -C cifar10
# rm cifar10_sdxl_base.tar



# # SDBP
# gdrive files download 1QpJ08Ci9f5PSTx96eF1P9geu1MR0hGsx
# tar -xf cifar10_sdbp.tar -C cifar10
# rm cifar10_sdbp.tar



# # sdxl diversified and generated equalweight (0528)
# ./gdrive files download 1VpElBUOtlUK3KNkbrgkdS-VGFqx7k9ww # sdxl diversified
# ./gdrive files download 1GcVfh1r0WVkXZHtcGcULIzw7sll7YOnt # equalweight
# ./gdrive files download 1dmWsy_9vMdJtBu8Kh9pCTeXQTNu-xGtt
# tar -xf cifar10_static_cot_50_sdxl.tar -C cifar10
# tar -xf cifar10_generated_equalweight.tar -C cifar10
# tar -xf CIFAR10_glide.tar -C cifar10 && mv cifar10/CIFAR10_glide cifar10/cifar10_glide # CIFAR10 -> cifar10 (change case)
# rm cifar10_static_cot_50_sdxl.tar
# rm cifar10_generated_equalweight.tar
# rm CIFAR10_glide.tar


# Compare between 5000 and 10000 (using sdxl_diversified)
./gdrive files download 1FXM3vJ0I9ntijCLtD4IxgmPz-M1T5Iop
./gdrive files download 1teeqK_nsl1MtSfI9kd9GfbpN-Z9uOFdX
tar -xf cifar10_sdxl_5000.tar -C cifar10
tar -xf cifar10_sdxl_10000.tar -C cifar10
rm cifar10_sdxl_5000.tar
rm cifar10_sdxl_10000.tar
