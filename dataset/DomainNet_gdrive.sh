./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # Required
# ./gdrive files download 144Vmht8QlAJ3eXoFHUWm8qg76I6XzrXj # DomainNet_MA (changed 0611)
# ./gdrive files download 1eCHHO8eheuM-FT0t13LbmOoTGxM0e-jO # train_ma
# ./gdrive files download 1Gnqna2AjQk-wVIwogGWS04MCvYnTmkdi # test_ma
# tar -xf DomainNet_MA.tar -C DomainNet
# tar -xf DomainNet_train_ma.tar -C DomainNet
# tar -xf DomainNet_test_ma.tar -C DomainNet
# rm DomainNet_MA.tar
# rm DomainNet_train_ma.tar
# rm DomainNet_test_ma.tar

# # Generated
# ./gdrive files download 1G_aILsxwKAhfFPwSuc2ecnuhE9JH1RH- # DomainNet_generated
# tar -xf DomainNet_generated.tar -C DomainNet
# rm DomainNet_generated.tar

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
# ./gdrive files download 11X-OPxul6dDX5c0qjt_77ScpPCsiTp0o
# tar -xf DomainNet_glide.tar -C DomainNet
# rm DomainNet_glide.tar



# # DomainNet glide diversified
# ./gdrive files download 1JDySFohyrSGxmSkvw1W_Ir-_naKncZHK
# tar -xf DomainNet_glide_diversified.tar -C DomainNet
# rm DomainNet_glide_diversified.tar


# # DomainNet wo hierarchy
# ./gdrive files download 1vRA8ysQhk9shBtEfOtXNk0V-K2Ckt3yQ
# tar -xf DomainNet_wo_hierarchy.tar -C DomainNet
# rm DomainNet_wo_hierarchy.tar

# # DomainNet wo cot
# ./gdrive files download 1xJcgL5-p7CrEo6aeL5DbDkHEOIPPhQmp
# tar -xf DomainNet_wo_cot.tar -C DomainNet
# rm DomainNet_wo_cot.tar

# # DomainNet wo cot & wo hierarchy
# ./gdrive files download 1rzVpd6EMqhh9eA_OLBeWQlHYYu-BSvEu
# tar -xf DomainNet_wo_cot_wo_hierarchy.tar -C DomainNet
# rm DomainNet_wo_cot_wo_hierarchy.tar

# # DomainNet MA changed
# rm -r DomainNet/DomainNet_MA
# ./gdrive files download 144Vmht8QlAJ3eXoFHUWm8qg76I6XzrXj
# tar -xf DomainNet_MA.tar -C DomainNet
# rm DomainNet_MA.tar

# # DomainNet static cot 50 (prompt ours, 0616)
# ./gdrive files download 1w6Q9QHIFchFg5dlYN2T1qFLpPsvrYBh2
# tar -xf DomainNet_static_cot_50_sdxl.tar -C DomainNet
# rm DomainNet_static_cot_50_sdxl.tar

# # DomainNet sdxl_1 (new prompt from PACS)
# ./gdrive files download 1OoTs9e6ZlzTFDoR4xvLH316CZgfhFqVg
# tar -xf DomainNet_sdxl_1.tar -C DomainNet
# rm DomainNet_sdxl_1.tar

# # DomainNet sdxl_1 filtered (check filtering performance)
# ./gdrive files download 1wbRy_sZwOdnP3CSZgLUk-w1fcU7hwg9A
# tar -xf DomainNet_sdxl_1_filtered.tar -C DomainNet
# rm DomainNet_sdxl_1_filtered.tar

# # robustness
# ./gdrive files download 11x0RwjWfI_w3P0GYZktGrd4aVn2Kiho5
# tar -xf DomainNet_robustness.tar -C DomainNet
# rm DomainNet_robustness.tar

# # ensemble experiments (0901)
# ./gdrive files download 1Kle4nSTZXIPGnk1tm2BYt3ITs2DX89pn
# ./gdrive files download 15qDhoCkb2RMzb-FZpR4_im8eC2qIluBs
# ./gdrive files download 1VFvnUE68YRlBmaQJWWFhXap_QYJ9-qOF
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_clip.tar -C DomainNet
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb16_normalized.tar -C DomainNet
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8_normalized.tar -C DomainNet
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_clip.tar
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb16_normalized.tar
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8_normalized.tar

# # DomainNet ensemble experiments (0903)
# ./gdrive files download 1Jb4HimYcikh-ZG_AtGOVKNUqGCEgWOQ2 # DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1t6nJLYJ9ceT91791RihiTMU7ih08c5sq # DomainNet_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_sdxl_floyd_cogview2_sd3_auraflow.tar

# # sdturbo (after) experiments (0906)
# ./gdrive files download 1PRwyQWl9j5h_EdiEAdFuifUmtYxeb7VU  # DomainNet_sdxl_floyd_cogview2_sdturbo
# tar -xf DomainNet_sdxl_floyd_cogview2_sdturbo.tar -C DomainNet
# rm DomainNet_sdxl_floyd_cogview2_sdturbo.tar

# DomainNet fake (d)
# ./gdrive files download 1UauDhEa59Pg58PP6abnLh3BVEXaIr-qM # DomainNet_fake_d_sdxl
# tar -xf DomainNet_fake_d_sdxl.tar -C DomainNet
# rm DomainNet_fake_d_sdxl.tar

# DomainNet sdxl with cot_2.json
./gdrive files download 1UeAEhv_zAIreYD8_J934gGeKORrl1FF4 # DomainNet_sdxl_2
tar -xf DomainNet_sdxl_2.tar -C DomainNet
rm DomainNet_sdxl_2.tar