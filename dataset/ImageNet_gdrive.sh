./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # Required
# mkdir -p ImageNet
# ./gdrive files download 1mM5xy9nkuvWY1Bv2-VLnfNO2zXYvExV_ # ImageNet_c
# ./gdrive files download 11XEooWsl5SSL4XgiXvDnDNOsQiz_Ebjb # ImageNet_r_50
# ./gdrive files download 1NLRNn8HeC41JfidifkwUwL2OKKAH4ovB # ImageNet_test
# tar -xf ImageNet_c.tar -C ImageNet
# tar -xf ImageNet_r_50.tar -C ImageNet
# tar -xf ImageNet_test.tar -C ImageNet
# rm ImageNet_c.tar
# rm ImageNet_r_50.tar
# rm ImageNet_test.tar

# # ImageNet RMD
# ./gdrive files download 120YZogsTblFW0Li-hbYg2uMMCSuOB-2k # ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C ImageNet
# rm ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar

# ImageNet sdxl
./gdrive files download 1l-LGHNfqcDiuUsB5JvoeJ6CQlP537Te- # ImageNet_50_2_sdxl
tar -xf ImageNet_50_2_sdxl.tar -C ImageNet
rm ImageNet_50_2_sdxl.tar