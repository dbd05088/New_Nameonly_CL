./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com

# # Required
# mkdir -p birds31
# ./gdrive files download 1HebDpIFKJhb9F2RH7joZ4qpY-W56LLeo # cub_test
# ./gdrive files download 1fD5oe5ZWad9sLp1mWrhFdgrC0oyxZVWg # inaturalist test
# ./gdrive files download 1nPX7HIZelgRhEszRS3QctVGTQvJRYBnr # nabirds test
# tar -xf birds31_cub_test.tar -C birds31
# tar -xf birds31_inaturalist_test.tar -C birds31
# tar -xf birds31_nabirds_test.tar -C birds31
# rm birds31_cub_test.tar
# rm birds31_inaturalist_test.tar
# rm birds31_nabirds_test.tar

# # train_ma
# ./gdrive files download 1f_WwjsDIiRvqu4pgh9L7hnDxZR8qQQYp
# tar -xf birds31_train_ma.tar -C birds31
# rm birds31_train_ma.tar

# # 50_2_sdxl
# ./gdrive files download 1AQVS8qlaUQCQH7LNPdwVK3LdNa9YUZ65 # birds31_50_2_sdxl
# tar -xf birds31_50_2_sdxl.tar -C birds31
# rm birds31_50_2_sdxl.tar

# # 50_2_sdxl DB finetuned
# ./gdrive files download 19DPE7q9EO-lqK0pBWcKlWK61Bmhx5E0e # birds31_db_3
# ./gdrive files download 1EsPn-RvPHpctCAduLKJJueGwvMvvZ-VA # birds31_db_5
# ./gdrive files download 1dKjh5r6on8Wj6XI8mKCuMA1YRZ9sKlyc # birds31_db_10
# tar -xf birds31_db_3.tar -C birds31
# tar -xf birds31_db_5.tar -C birds31
# tar -xf birds31_db_10.tar -C birds31
# rm birds31_db_3.tar
# rm birds31_db_5.tar
# rm birds31_db_10.tar

# # prompt baselines
# ./gdrive files download 1i_pYTI_oRf1ncJximPz1wwnXorWzSttq # birds31_LE
# ./gdrive files download 1F0JiFu_jjVlveKZwd7Rs01jxbAAIQs2K # birds31_fake_f
# ./gdrive files download 1nA2H8iTi2vkQyQEr_NvSj0j5eY2YzPce # birds31_synclr
# ./gdrive files download 15sJUj9AnR0h4gH_aItFQ_xL5q-NsgY62 # birds31_synthclip
# tar -xf birds31_LE.tar -C birds31
# tar -xf birds31_fake_f.tar -C birds31
# tar -xf birds31_synclr.tar -C birds31
# tar -xf birds31_synthclip.tar -C birds31
# rm birds31_LE.tar
# rm birds31_fake_f.tar
# rm birds31_synclr.tar
# rm birds31_synthclip.tar

# RMD parameter search
./gdrive files download 1i7wAtXkHknQjdlUTbWsiDy4PFklObjt9 # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_125
./gdrive files download 1TtwU9PQVX1Ha0Ky78FLu9ngfgdff5Ny- # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_25
./gdrive files download 1SBmPPV1MWoOXad5yedUoDpTd0PzpBAaP # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_5
./gdrive files download 1m5HsxtkTyHoH5VhaDSMVhptnpd73f8E4 # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_75
./gdrive files download 1FcyKe4VLb7rWYC19QAnTlrsAwnrRINLl # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp1
./gdrive files download 1tlsviJuSzuP8n69eibndr_gM6QG8mOIx # birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp2
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_125.tar -C birds31
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_25.tar -C birds31
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_5.tar -C birds31
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_75.tar -C birds31
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp1.tar -C birds31
tar -xf birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp2.tar -C birds31
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_125.tar
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_25.tar
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_5.tar
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp0_75.tar
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp1.tar
rm birds31_50_2_sdxl_floyd_cogview2_sd3_auraflow_temp2.tar