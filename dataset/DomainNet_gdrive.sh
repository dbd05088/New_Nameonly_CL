./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # Required
# ./gdrive files download 1I5W7bL46cduMMrNq4gH5KsUOonLejMuZ # DomainNet_MA
# ./gdrive files download 1eCHHO8eheuM-FT0t13LbmOoTGxM0e-jO # train_ma
# ./gdrive files download 1Gnqna2AjQk-wVIwogGWS04MCvYnTmkdi # test_ma
# tar -xf DomainNet_MA.tar -C DomainNet
# tar -xf DomainNet_train_ma.tar -C DomainNet
# tar -xf DomainNet_test_ma.tar -C DomainNet
# rm DomainNet_MA.tar
# rm DomainNet_train_ma.tar
# rm DomainNet_test_ma.tar



# # Temperature experiments
# ./gdrive files download 1xVSq77maFcHh089ina-gGMuu80yGP8CL
# ./gdrive files download 16kV6FtqtHGyr12iUgfuhYndMw-IrLXl2
# tar -xf DomainNet_RMD_classwise_temp_3.tar -C DomainNet
# tar -xf DomainNet_RMD_classwise_temp_5.tar -C DomainNet
# rm DomainNet_RMD_classwise_temp_3.tar
# rm DomainNet_RMD_classwise_temp_5.tar



# ./gdrive files download 1QS2mVXaTanGJISgDyg_16La1rlr99MqC
# ./gdrive files download 1WFLpCPrhEHdEjNoCRDPKCJBKEzb65Jc1
# tar -xf DomainNet_RMD_web_temp_2.tar -C DomainNet
# tar -xf DomainNet_RMD_web_temp_3.tar -C DomainNet
# rm DomainNet_RMD_web_temp_2.tar
# rm DomainNet_RMD_web_temp_3.tar



# # Web newsample
# ./gdrive files download 1LRfSItkn8G-2zptq5JJjVrsVYaEFhb3Y
# ./gdrive files download 1w7ht9yY4gi9-mFFzFjwJ7YqeGP1RjZxA
# tar -xf DomainNet_newsample_flickr.tar -C DomainNet
# tar -xf DomainNet_newsample_equalweight.tar -C DomainNet
# rm DomainNet_newsample_flickr.tar
# rm DomainNet_newsample_equalweight.tar



# static cot sdxl
# ./gdrive files download 1-n5e24Rqep7bUpndsg_uZGB7M-_rNvoP
# tar -xf DomainNet_static_cot_50_sdxl_filtered.tar -C DomainNet
# rm DomainNet_static_cot_50_sdxl_filtered.tar




# # DomainNet RMD - 0520
# ./gdrive files download 1WBqWhp2gifhm7nAId7otq3sTYkyGFNh_
# ./gdrive files download 14GrjceE3WJUhniyqTVJWa9IeGXKuE0Tj
# tar -xf DomainNet_generated_RMD_equalweight.tar -C DomainNet
# tar -xf DomainNet_generated_RMD_w_normalize_clip_90_temp_0_25.tar -C DomainNet
# rm DomainNet_generated_RMD_equalweight.tar
# rm DomainNet_generated_RMD_w_normalize_clip_90_temp_0_25.tar




# # Fix: DomainNet temp 0.25 -> 0.5
# ./gdrive files download 1AkZUACpAgK3EMKqB7DMAZ7wwwVNqyGmX
# tar -xf DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5.tar -C DomainNet
# rm DomainNet_generated_RMD_w_normalize_clip_90_temp_0_5.tar



# # Temp normalizing
# ./gdrive files download 1-1fAiY2xIleQZrKzJ_uwdrtnVh1lkyk3
# tar -xf DomainNet_web_RMD_w_normalize_clip_90_temp_0_5.tar -C DomainNet
# rm DomainNet_web_RMD_w_normalize_clip_90_temp_0_5.tar



# # SDBP
# ./gdrive files download 13icRhcAV4PX4AlBfXX7M-7salev9IDxO
# tar -xf DomainNet_sdbp.tar -C DomainNet
# rm DomainNet_sdbp.tar



# DomainNet glide (0607 - fixed size)
./gdrive files download 11X-OPxul6dDX5c0qjt_77ScpPCsiTp0o
tar -xf DomainNet_glide.tar -C DomainNet
rm DomainNet_glide.tar