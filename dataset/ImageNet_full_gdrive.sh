./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com

#
# # Optional: ImageNet ours full
# ./gdrive files download 1ktrf3RVKGanHKuKgqDhuxsP3PoMO7x86 # ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow.tar -C ImageNet_full
# rm ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow.tar

## Previous Test Benchmark (No Need)
# ./gdrive files download 1mM5xy9nkuvWY1Bv2-VLnfNO2zXYvExV_ # ImageNet_c
# tar -xf ImageNet_c.tar -C ImageNet_full
# rm ImageNet_c.tar


# # # Required
# mkdir -p ImageNet_full

# ./gdrive files download 11XEooWsl5SSL4XgiXvDnDNOsQiz_Ebjb # ImageNet_r_50
# ./gdrive files download 1NLRNn8HeC41JfidifkwUwL2OKKAH4ovB # ImageNet_test
# ./gdrive files download 13VQ678_Cs2qnWLULQR4Vp5cK2XV4j1n5 # ImageNet_c_all
# ./gdrive files download 1P23z_1FNJhyYS82eWVwnREIRkkkPuRIB # ImageNet_d_new

# tar -xf ImageNet_r_50.tar -C ImageNet_full
# tar -xf ImageNet_test.tar -C ImageNet_full
# tar -xf ImageNet_c_all.tar -C ImageNet_full
# tar -xf ImageNet_d_new.tar -C ImageNet_full
# rm ImageNet_r_50.tar
# rm ImageNet_test.tar
# rm ImageNet_c_all.tar
# rm ImageNet_d_new.tar

# # ImageNet train_ma full
# ./gdrive files download 1qERUAcUQSzrH3ESseCCU8pqhPqzt0f8G # ImageNet_full_train_ma
# tar -xf ImageNet_full_train_ma.tar -C ImageNet_full
# rm ImageNet_full_train_ma.tar

# # ImageNet ours full_subsampled
# ./gdrive files download 1Yz4MfmJdlYSD8tq4OS1nvgEQFtiClyqm # ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow_subsampled
# tar -xf ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow_subsampled.tar -C ImageNet_full
# rm ImageNet_full_ours_sdxl_floyd_cogview2_sd3_auraflow_subsampled.tar


