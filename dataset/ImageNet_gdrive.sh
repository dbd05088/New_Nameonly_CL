./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # Required
# mkdir -p ImageNet
## ./gdrive files download 1mM5xy9nkuvWY1Bv2-VLnfNO2zXYvExV_ # ImageNet_c
# ./gdrive files download 11XEooWsl5SSL4XgiXvDnDNOsQiz_Ebjb # ImageNet_r_50
# ./gdrive files download 1NLRNn8HeC41JfidifkwUwL2OKKAH4ovB # ImageNet_test
# ./gdrive files download 13VQ678_Cs2qnWLULQR4Vp5cK2XV4j1n5 # ImageNet_c_all
# ./gdrive files download 1P23z_1FNJhyYS82eWVwnREIRkkkPuRIB # ImageNet_d_new
## tar -xf ImageNet_c.tar -C ImageNet
# tar -xf ImageNet_r_50.tar -C ImageNet
# tar -xf ImageNet_test.tar -C ImageNet
# tar -xf ImageNet_c_all.tar -C ImageNet
# tar -xf ImageNet_d_new.tar -C ImageNet
## rm ImageNet_c.tar
# rm ImageNet_r_50.tar
# rm ImageNet_test.tar
# rm ImageNet_c_all.tar
# rm ImageNet_d_new.tar


# # ImageNet RMD
# ./gdrive files download 120YZogsTblFW0Li-hbYg2uMMCSuOB-2k # ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C ImageNet
# rm ImageNet_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar

# # ImageNet sdxl
# ./gdrive files download 1l-LGHNfqcDiuUsB5JvoeJ6CQlP537Te- # ImageNet_50_2_sdxl
# tar -xf ImageNet_50_2_sdxl.tar -C ImageNet
# rm ImageNet_50_2_sdxl.tar

# ImageNet train_ma 400
./gdrive files download 1ZC1cfRtsXQNVKpI2hOT3_8NsGu5tbmbQ # ImageNet_train_400
tar -xf ImageNet_train_400.tar -C ImageNet
rm ImageNet_train_400.tar