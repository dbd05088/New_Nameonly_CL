./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # Required
# mkdir -p DomainNet
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

# # DomainNet_cot_50_4~10, sdxl
# ./gdrive files download 1Sq5mjUsh680Vm-4wnHYwGIpNa4CCN2bW # DomainNet_cot_50_4_sdxl
# ./gdrive files download 1_hdqBBGqRyWD_NDXad1-sO4kdJBYm30x # DomainNet_cot_50_5_sdxl
# ./gdrive files download 1rmSiipiHvKdwyyA3vORZvCope8Q0OPLc # DomainNet_cot_50_6_sdxl
# ./gdrive files download 1WNsZi_p-USs22vaGYBu1Elsa6hI6X2P1 # DomainNet_cot_50_7_sdxl
# ./gdrive files download 1kpx7YkRCXIW659iQNKvzA3PPianmHDjj # DomainNet_cot_50_8_sdxl
# ./gdrive files download 1BrwOnG3uLX4CGfIHJuZ15f5rcG0aEB1s # DomainNet_cot_50_9_sdxl
# ./gdrive files download 1ICM4Zwq_0xsY9TCNLTjn3KxAugpTmTIj # DomainNet_cot_50_10_sdxl
# tar -xf DomainNet_cot_50_4_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_5_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_6_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_7_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_8_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_9_sdxl.tar -C DomainNet
# tar -xf DomainNet_cot_50_10_sdxl.tar -C DomainNet
# rm DomainNet_cot_50_4_sdxl.tar
# rm DomainNet_cot_50_5_sdxl.tar
# rm DomainNet_cot_50_6_sdxl.tar
# rm DomainNet_cot_50_7_sdxl.tar
# rm DomainNet_cot_50_8_sdxl.tar
# rm DomainNet_cot_50_9_sdxl.tar
# rm DomainNet_cot_50_10_sdxl.tar

# ./gdrive files download 1Kdpi3KLtaR7ppQgdClWRIgPQ4b5-89sx # DomainNet_cot_50_11_sdxl
# tar -xf DomainNet_cot_50_11_sdxl.tar -C DomainNet
# rm DomainNet_cot_50_11_sdxl.tar

# # DomainNet wo_hierarchy_50, wo_cot_wo_hierarchy_50 (from our prompt)
# ./gdrive files download 1p74Pz2Xk92Ujm1mMhaLu05To4fkAEveF # DomainNet_wo_cot_wo_hierarchy_50_sdxl
# ./gdrive files download 1BsnbWwzKdK3U2Xy_ZZAm8IK_MdQPyIpY # DomainNet_wo_hierarchy_50_sdxl
# tar -xf DomainNet_wo_cot_wo_hierarchy_50_sdxl.tar -C DomainNet
# tar -xf DomainNet_wo_hierarchy_50_sdxl.tar -C DomainNet
# rm DomainNet_wo_cot_wo_hierarchy_50_sdxl.tar
# rm DomainNet_wo_hierarchy_50_sdxl.tar

# # LE_glide, sdxl, 50 / base_sdxl, 50 (0921)
# ./gdrive files download 15ibNAab3NbaQ_JIOElyLaA8HQkYNgE4n # DomainNet_LE_diversified_50_glide
# ./gdrive files download 1FnP9BE3HAFzIV1Z6-JppbggNCbs22S1t # DomainNet_LE_diversified_50_sdxl
# ./gdrive files download 1RHD5Q72qlBplC87WRZjiUyoU0ZZD65lC # DomainNet_base_sdxl
# tar -xf DomainNet_LE_diversified_50_glide.tar -C DomainNet
# tar -xf DomainNet_LE_diversified_50_sdxl.tar -C DomainNet
# tar -xf DomainNet_base_sdxl.tar -C DomainNet
# rm DomainNet_LE_diversified_50_glide.tar
# rm DomainNet_LE_diversified_50_sdxl.tar
# rm DomainNet_base_sdxl.tar

# # RMD, wo_c_wo_h_50
# ./gdrive files download 1nGwIkksDCAVkyNvcuJsvdwJnJXK9wK1H # DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 161OpoXWiB3AeCWiYKgnQDKeMeZ50qO2t # DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1oLWVMobYbKYEPO83acp06puoZvNY7Z5d # DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# tar -xf DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# tar -xf DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C DomainNet
# rm DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar

# # synclr ensemble (0922)
# ./gdrive files download 13-afAm5s9HCJ57IOKkQ-bpENODB8oHPR # DomainNet_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1OA5qCG16wsQwquHgXsx2cTg3qdxC3jVm # DomainNet_synclr_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf DomainNet_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# rm DomainNet_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar

# ./gdrive files download 1izoRaJ1l1PZcUp8lmT46xiXqahRSNM9x # DomainNet_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1Z3ysx6J0M4xUFJeP49-7UZTogNNdIOH3 # DomainNet_fake_f_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf DomainNet_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# tar -xf DomainNet_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# rm DomainNet_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm DomainNet_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar

# # DomainNet hcfr equalweight (wo flux)
# ./gdrive files download 1F94zSKSxP_wwCioTHQ6zUJdgDLP0h7U5 # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C DomainNet
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar

# ./gdrive files download 1cEpzz51wFKgGHS4t4akpI12MvOK0lHu2 # DomainNet_synthclip_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1_bBxwKuEZaD3tbAQcl365DBMna3rkMLs # DomainNet_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow
# tar -xf DomainNet_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# tar -xf DomainNet_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C DomainNet
# rm DomainNet_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm DomainNet_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow.tar

# # DomainNet sdbp (only sdxl)
# ./gdrive files download 1JZbLGdXtZf3WOSs1SwlWM4lp5g17_vEH # DomainNet_sdbp_sdxl
# tar -xf DomainNet_sdbp_sdxl.tar -C DomainNet
# rm DomainNet_sdbp_sdxl.tar

# # DomainNet sdbp + CONAN
# ./gdrive files download 15bEK1JQcIxw9BVM-sksTo-2wk4tWQAcr # DomainNet_sdbp_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf DomainNet_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# rm DomainNet_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar

# # DomainNet fake_f coreset
# ./gdrive files download 1X_9VeSsI9Yv_wnKMeDDtKgTDcB-J02JW # DomainNet_fake_f_wo_flux_CLIP_moderate
# ./gdrive files download 1hOTtKU_0UfjoLZmgrBbqA2mt_oba4QDB # DomainNet_fake_f_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1cjjsbXBnpg9OFn-A0J6th2lk71r-8pdr # DomainNet_fake_f_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1GQTPRLopF3PJBVesZxG2mOESBNXSBDhJ # DomainNet_fake_f_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1kZNxCv7rfhHmQUAfx_VTOeV1fCgKNvQT # DomainNet_fake_f_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1bAlzREXsTCJVJiqkAx73PCOUP7zEbfd3 # DomainNet_fake_f_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1qUrqHKKdmAebLq2ff3te-hLj4aQaLhYs # DomainNet_fake_f_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf DomainNet_fake_f_wo_flux_CLIP_moderate.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_fake_f_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# rm DomainNet_fake_f_wo_flux_CLIP_moderate.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_fake_f_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # LE RMD and RMD_EW
# ./gdrive files download 1UDb3T2qBR7YdbA5W6B_zv7cAp1oGdQ7V # DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1DhIsxqgY1bHS9gHJqO_vFsllvgHrQQop # DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow.tar -C DomainNet
# tar -xf DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C DomainNet
# rm DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm DomainNet_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar

# # HIWING coreset
# ./gdrive files download 1wXx6jZDNfUIsV2Xb9L2EZPgtcTg75nXY # DomainNet_cot_50_2_wo_flux_CLIP_moderate
# ./gdrive files download 1fHymODeMufoXG6yoxqDa41wBClUysotq # DomainNet_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1nrNBpHlxtxz9Te_VPeA01E3RuMP-lpKt # DomainNet_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1GC5Xbrcg5kEBivbqdHPPm1u_9faD0bsl # DomainNet_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1VPmg4p5mUkZPZhvilm931Xql8UxRWoUK # DomainNet_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1hhy-QrS3kRrNoNsCZWUOH8O0jCH7daUK # DomainNet_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1QL76aDzEa5fkP9a3IU9GIdgFpJMj26AS # DomainNet_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf DomainNet_cot_50_2_wo_flux_CLIP_moderate.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# rm DomainNet_cot_50_2_wo_flux_CLIP_moderate.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # DomainNet HIWING equalweight experiments
# ./gdrive files download 1lZ8KL7RSKOXBseFkrx9jxSc0EestcfsX # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver5
# ./gdrive files download 12emIyFsOmiZeZt1-oPYCjHVbegsN13Y_ # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver4
# ./gdrive files download 16T0aXHy-n6WAFraDaNAlaJaAZUj4sDA5 # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver3
# ./gdrive files download 1r0geQypHA2phVSqkQVzz7CaIsz7M7liQ # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver2
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver5.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver4.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver3.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver2.tar -C DomainNet
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver5.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver4.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver3.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight_ver2.tar

# # DomainNet LE coresets wo flux
# ./gdrive files download 1XyPTgbNkKSxyiHjSLWMmxc6SDqHb4sLq # DomainNet_LE_wo_flux_CLIP_moderate
# ./gdrive files download 1dGwPfn_RJHYly9bs6sR0QmuldqnFE4MX # DomainNet_LE_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1e_VeCUVwuE9yAIonFb6MR4xYHUsQ9j5T # DomainNet_LE_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1OxH13omXKn2nARWdZiqS7SPRTnIdn8Wc # DomainNet_LE_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1z9sVtLfyKSpAplb0G5cteQPuFQ2o3q0r # DomainNet_LE_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1tdeGa4CbCjUJZ9MickyBiDPoyJjQfta1 # DomainNet_LE_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1i4qsSA-RamTVgz3EHrL_G_d2CFuGrIKg # DomainNet_LE_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf DomainNet_LE_wo_flux_CLIP_moderate.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_LE_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# rm DomainNet_LE_wo_flux_CLIP_moderate.tar
# rm DomainNet_LE_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_LE_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_LE_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_LE_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_LE_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_LE_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # DomainNet synclr wo flux
# ./gdrive files download 1w6sKlS3TO2JRQC8Rsb4naOnzisiw9Oqa # DomainNet_synclr_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 11l5ydbAYqJ7vPpVzzD8PIPIKhl3rNRr3 # DomainNet_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1LOCIoc53dN1CABCx78Ak93JD3dsdcKca # DomainNet_synclr_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1fqN069OhNVE2wxdINqft89g9sYzwbDmM # DomainNet_synclr_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1glN4kz9joTQoKbFclmgkzloNZgvOmp3Q # DomainNet_synclr_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1AqiGTFyNMPoZg4xVqAPW2jYEsuMkkmlw # DomainNet_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1gta7Tz7c31FAutgAEO1GHgXOAV6cNY-y # DomainNet_synclr_wo_flux_CLIP_moderate
# tar -xf DomainNet_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synclr_wo_flux_CLIP_moderate.tar -C DomainNet
# rm DomainNet_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar
# rm DomainNet_synclr_wo_flux_CLIP_moderate.tar

# # DomainNet synthclip wo flux
# ./gdrive files download 1WZrHnL5Amx7ZNhIA5uFl04DGZasOukHb # DomainNet_synthclip_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1j_8Aoua69J0OVHuicpJcwuNMj_1-fckz # DomainNet_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1XSQt-prdNYN8WZBt4Fh8ApQ-I4kxNa5l # DomainNet_synthclip_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1GBXg3lVoaptMVgVtkcQ09jyq1kotsMHX # DomainNet_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1R2CLXK0DvcW5Wgs3M7s5aeVNYLhPLN-J # DomainNet_synthclip_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1SM5eZr9wBCfhtTL3sOVEvfr9WR1P4lvK # DomainNet_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1Kytptg3Puhwxewa4u09h-NBcwoRquYKf # DomainNet_synthclip_wo_flux_CLIP_moderate
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_synthclip_wo_flux_CLIP_moderate.tar -C DomainNet
# rm DomainNet_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar
# rm DomainNet_synthclip_wo_flux_CLIP_moderate.tar

# # DomainNet sdbp wo flux
# ./gdrive files download 1qvsQ6kKTd1rO6SucJaibpglFyC1WI-2- # DomainNet_sdbp_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1_3a8SuDw7HX9uvpEJmpwp-s0O-GmXWJY # DomainNet_sdbp_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1fkE1a6i0SiY2FgeR6gSYLDgd3v11q9sd # DomainNet_sdbp_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 16iI9Yk0E-2_vIf8Yi5Y7WIPiinUgsHtM # DomainNet_sdbp_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1VZY5bSOj_DYnUf6u-soC4b709QpnamnY # DomainNet_sdbp_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1akf8ogbj4tJ5v-hsHY2MFggIiacFKgy5 # DomainNet_sdbp_wo_flux_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1hQcQwlTFMNWDqkHcjuJXE1fBlIb-NSHy # DomainNet_sdbp_wo_flux_CLIP_moderate
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_Adacore_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_Glister_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_Submodular_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C DomainNet
# tar -xf DomainNet_sdbp_wo_flux_CLIP_moderate.tar -C DomainNet
# rm DomainNet_sdbp_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_DINO_base_Uncertainty_10_0.0001.tar
# rm DomainNet_sdbp_wo_flux_CLIP_moderate.tar

# # DomainNet dynamic 50 sdxl
# ./gdrive files download 1M-wT4kiVgX9t7TRCn1OFjC31e_cpdcAn # DomainNet_dynamic_50_sdxl
# tar -xf DomainNet_dynamic_50_sdxl.tar -C DomainNet
# rm DomainNet_dynamic_50_sdxl.tar

# # DomainNet equalweight ver6, ver7
# ./gdrive files download 1fcnDDA8OVuXI2XM4nEF0USDTSkSJpMMe # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver6
# ./gdrive files download 1koZAf5mNldBOpGsxmmw9EEDSP4lfbEsT # DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver7
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver6.tar -C DomainNet
# tar -xf DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver7.tar -C DomainNet
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver6.tar
# rm DomainNet_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_ver7.tar

# ./gdrive files download 14BKE63_O5cxOxM9_xpCoSght2BlNKVrK # DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C DomainNet
# rm DomainNet_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar

# ./gdrive files download 1ZIP983w2N70gbKzaT3QcMKLisIxjYtTT # DomainNet_50_2_glide
# tar -xf DomainNet_50_2_glide.tar -C DomainNet
# rm DomainNet_50_2_glide.tar

# # Temperature experiments
# ./gdrive files download 1PRdkrxz__xCVzhYV_1__ZKG37SBKs6eY # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_125
# ./gdrive files download 1WOd3OqjdN1GsKFLx3KwerdBxNgOwpt1B # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_25
# ./gdrive files download 1JNvO2s7MdNJ36kcJYr9C7QWEc1Mq6ehr # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_1
# ./gdrive files download 10Uxnsy_lF6hyDKB-tBuSi3RleA_vq8g0 # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_2
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_125.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_25.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_1.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_2.tar -C DomainNet
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_125.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_25.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_1.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_2.tar

# # DomainNet hierarchy experiments
# ./gdrive files download 1xrvTGTJdWOy1LKj0svi2RmZoObjvMsBa # DomainNet_tree_2
# ./gdrive files download 1YaL7vwq6VwWgRGyTmiAXI7dYOTPwJ2BL # DomainNet_tree_4
# tar -xf DomainNet_tree_2.tar -C DomainNet
# tar -xf DomainNet_tree_4.tar -C DomainNet
# rm DomainNet_tree_2.tar
# rm DomainNet_tree_4.tar

# # DomainNet Real-Fake
# ./gdrive files download 150rLPOkqw5cfToaMJJ2PVk4CixrN3f60 # DomainNet_train_ma_real_fake
# tar -xf DomainNet_train_ma_real_fake.tar -C DomainNet
# rm DomainNet_train_ma_real_fake.tar

# # Temperature more && CLIP ratio experiments
# ./gdrive files download 1_dbpA-VGSGMCVTMn7OOrw9dwr0FIqH09 # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_0625
# ./gdrive files download 1ULDX7TOSSzqfGguKEjkjilZgDbtusIle # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_4
# ./gdrive files download 1OTfBN7nzncP8wSPiTlvg-ePpAAzjyQ4F # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_87_5
# ./gdrive files download 1nyQRLbXzxDQAIsCE5stevGUsv8VnHBWJ # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_92_5
# ./gdrive files download 1RLhKtuUn3e9OuzQGT4EQvlN29irA9yUQ # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_95
# ./gdrive files download 1ozQz840JcIakm6geMEVj1XktXiVGghLb # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_97_5
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_0625.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_4.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_87_5.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_92_5.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_95.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_97_5.tar -C DomainNet
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_0_0625.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_4.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_87_5.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_92_5.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_95.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_97_5.tar

# # DB finetune with 5 samples
# ./gdrive files download 1IC_7ns5Ny5cwN8tZjZGDGa3W-oo9mXjr # DomainNet_db_5_500
# tar -xf DomainNet_db_5_500.tar -C DomainNet
# rm DomainNet_db_5_500.tar

# # DomainNet more truncate ratio
# ./gdrive files download 1lu_NRZLj1DbWmgnJGGo9o2-7nXAuip-q # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_30
# ./gdrive files download 1FeCEtjKkISbFOp-5MW0zxuxc9Ie0B8CQ # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_50
# ./gdrive files download 1BaNZS-G6i8lS61yZS3oqwTvJVQ0od-DQ # DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_70
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_30.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_50.tar -C DomainNet
# tar -xf DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_70.tar -C DomainNet
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_30.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_50.tar
# rm DomainNet_50_2_sdxl_floyd_cogview2_sd3_auraflow_70.tar

# Internet explorer
./gdrive files download 1-5zPSOvc0saxnjZ2zBJjTcJp0_YNIklm # DomainNet_internet_explorer
tar -xf DomainNet_internet_explorer.tar -C DomainNet
rm DomainNet_internet_explorer.tar