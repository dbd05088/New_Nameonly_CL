./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com

# ./gdrive files download 1eCHHO8eheuM-FT0t13LbmOoTGxM0e-jO # train_ma
# ./gdrive files download 1Gnqna2AjQk-wVIwogGWS04MCvYnTmkdi # test_ma
# ./gdrive files download 1VNIz2xPm6ICKzp8i7KhhyN2VYv4fkQZY # sdxl_diversified
# ./gdrive files download 1I5W7bL46cduMMrNq4gH5KsUOonLejMuZ # DomainNet_MA
# ./gdrive files download 1G_aILsxwKAhfFPwSuc2ecnuhE9JH1RH- # DomainNet_generated
# ./gdrive files download 1sZdh04o80X8xKNBSXsnOe1VBlhk1f1Oy # DomainNet_web2
./gdrive files download 1KSXL7_0Nf53EvmsfV-NrD7e_c7wUPHZn # DomainNet_glide

# tar -xvf DomainNet_train_ma.tar
# tar -xvf DomainNet_test_ma.tar
# tar -xvf DomainNet_sdxl_diversified.tar
# tar -xvf DomainNet_web2.tar
# tar -xvf DomainNet_MA.tar
# tar -xvf DomainNet_generated.tar
tar -xvf DomainNet_glide.tar

mkdir -p DomainNet
# mv DomainNet_MA DomainNet
# mv DomainNet_train_ma DomainNet
# mv DomainNet_test_ma DomainNet
# mv DomainNet_web2 DomainNet
# mv DomainNet_sdxl_diversified DomainNet
# mv DomainNet_generated DomainNet
mv DomainNet_glide DomainNet

# rm DomainNet_MA.tar
# rm DomainNet_sdxl_diversified.tar
# rm DomainNet_train_ma.tar
# rm DomainNet_test_ma.tar
# rm DomainNet_web2.tar
# rm DomainNet_generated.tar
rm DomainNet_glide.tar

# ./gdrive files download 1xVSq77maFcHh089ina-gGMuu80yGP8CL
# ./gdrive files download 16kV6FtqtHGyr12iUgfuhYndMw-IrLXl2

# tar -xvf DomainNet_RMD_classwise_temp_3.tar
# tar -xvf DomainNet_RMD_classwise_temp_5.tar

# mv DomainNet_RMD_classwise_temp_3 DomainNet
# mv DomainNet_RMD_classwise_temp_5 DomainNet

# rm DomainNet_RMD_classwise_temp_3.tar
# rm DomainNet_RMD_classwise_temp_5.tar

# ./gdrive files download 1QS2mVXaTanGJISgDyg_16La1rlr99MqC
# ./gdrive files download 1WFLpCPrhEHdEjNoCRDPKCJBKEzb65Jc1

# tar -xvf DomainNet_RMD_web_temp_2.tar
# tar -xvf DomainNet_RMD_web_temp_3.tar

# mv DomainNet_RMD_web_temp_2 DomainNet
# mv DomainNet_RMD_web_temp_3 DomainNet

# rm DomainNet_RMD_web_temp_2.tar
# rm DomainNet_RMD_web_temp_3.tar

# mkdir -p DomainNet
# ./gdrive files download 1LRfSItkn8G-2zptq5JJjVrsVYaEFhb3Y
# ./gdrive files download 1w7ht9yY4gi9-mFFzFjwJ7YqeGP1RjZxA
# tar -xf DomainNet_newsample_flickr.tar
# tar -xf DomainNet_newsample_equalweight.tar
# mv DomainNet_newsample_flickr DomainNet/
# mv DomainNet_newsample_equalweight DomainNet/
# rm DomainNet_newsample_flickr.tar
# rm DomainNet_newsample_equalweight.tar

# ./gdrive files download 1-n5e24Rqep7bUpndsg_uZGB7M-_rNvoP
# tar -xf DomainNet_static_cot_50_sdxl_filtered.tar
# mv DomainNet_static_cot_50_sdxl_filtered DomainNet/
# rm DomainNet_static_cot_50_sdxl_filtered.tar

# # DomainNet RMD - 0520
# ./gdrive files download 1WBqWhp2gifhm7nAId7otq3sTYkyGFNh_
# ./gdrive files download 14GrjceE3WJUhniyqTVJWa9IeGXKuE0Tj
# tar -xf DomainNet_generated_RMD_equalweight.tar
# tar -xf DomainNet_generated_RMD_w_normalize_clip_90_temp_0_25.tar
# mv DomainNet_generated_RMD_equalweight DomainNet/
# mv DomainNet_generated_RMD_w_normalize_clip_90_temp_0_25 DomainNet/
# rm DomainNet_generated_RMD_equalweight.tar
# rm DomainNet_generated_RMD_w_normalize_clip_90_temp_0_25.tar

# # Fix: DomainNet temp 0.25 -> 0.5
# ./gdrive files download 1AkZUACpAgK3EMKqB7DMAZ7wwwVNqyGmX
# tar -xf DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5.tar
# mv DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5 DomainNet/
# rm DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5.tar

# ./gdrive files download 1-1fAiY2xIleQZrKzJ_uwdrtnVh1lkyk3
# ./gdrive files download 1-1fAiY2xIleQZrKzJ_uwdrtnVh1lkyk3
# tar -xf DomainNet_web_RMD_w_normalize_clip_90_temp_0_5.tar
# mv DomainNet_web_RMD_w_normalize_clip_90_temp_0_5 DomainNet/
# rm DomainNet_web_RMD_w_normalize_clip_90_temp_0_5.tar

# ./gdrive files download 13icRhcAV4PX4AlBfXX7M-7salev9IDxO
# tar -xf DomainNet_sdbp.tar
# mv DomainNet_sdbp DomainNet/
# rm DomainNet_sdbp.tar
