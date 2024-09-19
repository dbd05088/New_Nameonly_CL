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

# # DomainNet sdxl with cot_2.json
# ./gdrive files download 1UeAEhv_zAIreYD8_J934gGeKORrl1FF4 # DomainNet_sdxl_2
# tar -xf DomainNet_sdxl_2.tar -C DomainNet
# rm DomainNet_sdxl_2.tar

# # DomainNet sdxl with cot_3,4,5.json
# ./gdrive files download 1yc5fAlkW874yuNlxJLmmXGxZXwtI2Xmx # DomainNet_sdxl_3
# ./gdrive files download 1Gtep-A_eIdx6vUp7gJjZ-Yq5cEHrjWBM # DomainNet_sdxl_4
# ./gdrive files download 1ZqWKjzhfRcCc63RoGnVROM-pGrj0IGz2 # DomainNet_sdxl_5
# tar -xf DomainNet_sdxl_3.tar -C DomainNet
# tar -xf DomainNet_sdxl_4.tar -C DomainNet
# tar -xf DomainNet_sdxl_5.tar -C DomainNet
# rm DomainNet_sdxl_3.tar
# rm DomainNet_sdxl_4.tar
# rm DomainNet_sdxl_5.tar

# # DomainNet sdxl with cot_6.json
# ./gdrive files download 1grRlyUHuBrQ2-XXYP9NeibGwBst27EDZ # DomainNet_sdxl_6
# tar -xf DomainNet_sdxl_6.tar -C DomainNet
# rm DomainNet_sdxl_6.tar

# # DomainNet synthclip, synclr
# ./gdrive files download 1lXaMAB-Dv2sDw3ALZ3EqQCOW4BdygIoo # DomainNet_synclr
# ./gdrive files download 1V9Lkmwm2MkspOtPgMHyQeQ1w1dZW-UE2 # DomainNet_synthclip
# tar -xf DomainNet_synclr.tar -C DomainNet
# tar -xf DomainNet_synthclip.tar -C DomainNet
# rm DomainNet_synclr.tar
# rm DomainNet_synthclip.tar

# # DomainNet fake (f)
# ./gdrive files download 1Rsjw1Nint6xgV_RcziELseQeEh-jG384 # DomainNet_fake_f
# tar -xf DomainNet_fake_f.tar -C DomainNet
# rm DomainNet_fake_f.tar

# # DomainNet refined
# ./gdrive files download 1oaWsQgmCqeLA0CwzpU81KL-EYuN8Gh3J # DomainNet_sdxl_4_refined
# tar -xf DomainNet_sdxl_4_refined.tar -C DomainNet
# rm DomainNet_sdxl_4_refined.tar

# # DomainNet ensemble with cot_4 (0911)
# ./gdrive files download 1rWEThnulsvYRnTgbs4yUST72zaThbVdm # DomainNet_sdxl_floyd_cogview2_sd3_auraflow_4
# ./gdrive files download 1SEYPwXb3V3IkU47d8b3dukM7Qb_A7WtC # DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow_4
# ./gdrive files download 1SC6YFr1Y7lIejWDiqFpaehBAd6kOXvLN # DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_4
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_auraflow_4.tar -C DomainNet
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow_4.tar -C DomainNet
# tar -xf DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_4.tar -C DomainNet
# rm DomainNet_sdxl_floyd_cogview2_sd3_auraflow_4.tar
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_auraflow_4.tar
# rm DomainNet_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_4.tar

# # LE diversifed (0914)
# ./gdrive files download 1FVpuoG7MgY4lOHfMvw0YyOLnWtzCaBxO # DomainNet_LE_diversified
# tar -xf DomainNet_LE_diversified.tar -C DomainNet
# rm DomainNet_LE_diversified.tar

# # LE diversified with 100, sdxl (0915)
# ./gdrive files download 1A1xQ7ECYq6R2_cws_xb9cxJEmh05d7gN # DomainNet_LE_diversified_100_sdxl
# tar -xf DomainNet_LE_diversified_100_sdxl.tar -C DomainNet
# rm DomainNet_LE_diversified_100_sdxl.tar

# # cot_100_2 ensemble (0916)
# ./gdrive files download 1gG8ommB87RhaiakboSy93OFzrwKRWkfh # DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# ./gdrive files download 1rWaxXaGYT11fYRXYbDo3QO0Kz1dZQKje # DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1HCxycLIT_WYkhGJinKeZw298NdJanLSA # DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C DomainNet
# tar -xf DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# rm DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar
# rm DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow.tar

# # cot_50_2 / cot_100_2 only sdxl (0917)
# ./gdrive files download 1KydEiqUPMzoeXeMK_I6I2V6oDYy1Xlad # DomainNet_cot_50_2_sdxl
# ./gdrive files download 15mGSETe5HOHjHByJlG77LG3zH29pGavo # DomainNet_cot_100_2_sdxl
# tar -xf DomainNet_cot_50_2_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_100_2_sdxl.tar -C DomainNet
# rm DomainNet_cot_50_2_sdxl.tar
# rm DomainNet_cot_100_2_sdxl.tar

# # cot_50_2 ensemble
# ./gdrive files download 1aOFzD41EuVRggrJ9drE81p08t4KOdvre # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1y835L-a6JwWxwq4-vlChgl0nDncSy_l_ # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 13fV-NghCWCXYWk8H7cb_hBYyRqTKCipV # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C DomainNet
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar

# # DomainNet cot_50_2 refined sdxl
# ./gdrive files download 1wYqEb66Y3MkwWrHrfti0WAs7a2YLBENY # DomainNet_cot_50_2_refined_sdxl
# tar -xf DomainNet_cot_50_2_refined_sdxl.tar -C DomainNet
# rm DomainNet_cot_50_2_refined_sdxl.tar

# DomainNet_cot_50_4~10, sdxl
./gdrive files download 1Sq5mjUsh680Vm-4wnHYwGIpNa4CCN2bW # DomainNet_cot_50_4_sdxl
./gdrive files download 1_hdqBBGqRyWD_NDXad1-sO4kdJBYm30x # DomainNet_cot_50_5_sdxl
./gdrive files download 1rmSiipiHvKdwyyA3vORZvCope8Q0OPLc # DomainNet_cot_50_6_sdxl
./gdrive files download 1WNsZi_p-USs22vaGYBu1Elsa6hI6X2P1 # DomainNet_cot_50_7_sdxl
./gdrive files download 1kpx7YkRCXIW659iQNKvzA3PPianmHDjj # DomainNet_cot_50_8_sdxl
./gdrive files download 1BrwOnG3uLX4CGfIHJuZ15f5rcG0aEB1s # DomainNet_cot_50_9_sdxl
./gdrive files download 1ICM4Zwq_0xsY9TCNLTjn3KxAugpTmTIj # DomainNet_cot_50_10_sdxl
tar -xf DomainNet_cot_50_4_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_5_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_6_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_7_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_8_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_9_sdxl.tar -C DomainNet
tar -xf DomainNet_cot_50_10_sdxl.tar -C DomainNet
rm DomainNet_cot_50_4_sdxl.tar
rm DomainNet_cot_50_5_sdxl.tar
rm DomainNet_cot_50_6_sdxl.tar
rm DomainNet_cot_50_7_sdxl.tar
rm DomainNet_cot_50_8_sdxl.tar
rm DomainNet_cot_50_9_sdxl.tar
rm DomainNet_cot_50_10_sdxl.tar