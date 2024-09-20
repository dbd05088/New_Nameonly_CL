./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com

# # Required
# mkdir -p PACS_final
# ./gdrive files download 1Q-leEeSjgWZNQUbapU1GmHKFr947_-K- # test_ma
# ./gdrive files download 1LfcRv1xFlLbT9Etx71JbpWzKSQiouxjO # PACS MA
# ./gdrive files download 11zV8OnLYqQ9JQZVkxv_1OTWdtbscBsUn # final train ma
# tar -xf PACS_final_test_ma.tar -C PACS_final
# tar -xf PACS_MA.tar -C PACS_final
# tar -xf PACS_final_train_ma.tar -C PACS_final
# rm PACS_final_test_ma.tar
# rm PACS_MA.tar
# rm PACS_final_train_ma.tar

### temperature exp ###
# ./gdrive files download 1mI5-cJY8ofwU6ZeiUBp9Ym_Q_kfQzzG-
# ./gdrive files download 1tDwpUQu69sSl2ZkcMfyHecOTpU1XU88B
# ./gdrive files download 1Z8pgas6mgGPuNYfXKP36Tm-bDlNJ3d53
# ./gdrive files download 1mwXwJesZg_067tEoPFR6S9GGkF_863DJ

# tar -xf PACS_ensembled_RMD_temp1.tar
# tar -xf PACS_ensembled_RMD_temp2.tar
# tar -xf PACS_ensembled_RMD_temp5.tar
# tar -xf PACS_ensembled_RMD_temp10.tar

# mv PACS_ensembled_RMD_temp1 PACS_final
# mv PACS_ensembled_RMD_temp2 PACS_final
# mv PACS_ensembled_RMD_temp5 PACS_final
# mv PACS_ensembled_RMD_temp10 PACS_final

# rm PACS_ensembled_RMD_temp1.tar
# rm PACS_ensembled_RMD_temp2.tar
# rm PACS_ensembled_RMD_temp5.tar
# rm PACS_ensembled_RMD_temp10.tar
# #######################

### modelwise exp ###
# ./gdrive files download 1Lhybt-iPWMRxaSgMwnA9q6sgglHOo4qu
# ./gdrive files download 1Vq0sT7sEhL1PRZBCw7WPialIZoYKHFD1
# ./gdrive files download 1pruaOIeVO4dy3EGryXIAJkySbUV5_ENY
# ./gdrive files download 1YNgu59uxnloTDyWznL_8UGV7D8KcmNAR

# tar -xf PACS_ensembled_RMD_0_5.tar
# tar -xf PACS_ensembled_RMD_1.tar
# tar -xf PACS_ensembled_RMD_3.tar
# tar -xf PACS_ensembled_RMD_5.tar

# mv PACS_ensembled_RMD_0_5 PACS_final
# mv PACS_ensembled_RMD_1 PACS_final
# mv PACS_ensembled_RMD_3 PACS_final
# mv PACS_ensembled_RMD_5 PACS_final

# rm PACS_ensembled_RMD_0_5.tar
# rm PACS_ensembled_RMD_1.tar
# rm PACS_ensembled_RMD_3.tar
# rm PACS_ensembled_RMD_5.tar
#######################

### samplewise exp ###
# ./gdrive files download 1MDjt93gD4ClkW61IrL6ZdOyOe8YQjf4A
# ./gdrive files download 1gd1BIV0gHBbMpNOg9ny0G0rfZ9myCSGh
# ./gdrive files download 1dvTEhMxX94vwXp-gQ8fJZHU87EY0VURc
# ./gdrive files download 1vll5aPNctm7zMuFShGjQFKsfIgAkJdMx

# tar -xf PACS_ensembled_samplewise_RMD_0_5.tar
# tar -xf PACS_ensembled_samplewise_RMD_1.tar
# tar -xf PACS_ensembled_samplewise_RMD_3.tar
# tar -xf PACS_ensembled_samplewise_RMD_5.tar

# mv PACS_ensembled_samplewise_RMD_0_5 PACS_final
# mv PACS_ensembled_samplewise_RMD_1 PACS_final
# mv PACS_ensembled_samplewise_RMD_3 PACS_final
# mv PACS_ensembled_samplewise_RMD_5 PACS_final

# rm PACS_ensembled_samplewise_RMD_0_5.tar
# rm PACS_ensembled_samplewise_RMD_1.tar
# rm PACS_ensembled_samplewise_RMD_3.tar
# rm PACS_ensembled_samplewise_RMD_5.tar
#######################

### classwise ensemble ###
# ./gdrive files download 1BgXNXE-qWqW_Rr3zKz9k1yIzPksSuxoA
# ./gdrive files download 1IinghAZHJlN1btD0Q2N_1LMbBJRZ26Ix
# ./gdrive files download 1mdNk3gLXI-5jdKrZWoENEzxMEysLgWw1

# tar -xf PACS_final_ensembled_RMD_classwise_temp_3.tar
# tar -xf PACS_final_ensembled_RMD_classwise_temp_1.tar
# tar -xf PACS_final_ensembled_RMD_classwise_temp_0_5.tar

# mv PACS_final_ensembled_RMD_classwise_temp_0_5 PACS_final
# mv PACS_final_ensembled_RMD_classwise_temp_1 PACS_final
# mv PACS_final_ensembled_RMD_classwise_temp_3 PACS_final

# rm PACS_final_ensembled_RMD_classwise_temp_0_5.tar
# rm PACS_final_ensembled_RMD_classwise_temp_1.tar
# rm PACS_final_ensembled_RMD_classwise_temp_3.tar

### equalweight ensemble ###
# ./gdrive files download 1hE8d1i9SjjmgkDmslfgiKhyA01xKrqoN
# tar -xf PACS_final_equalweighted.tar
# mv PACS_final_equalweighted PACS_final
# rm PACS_final_equalweighted.tar

### generated ###
# ./gdrive files download 1mZP20hEI7ZI8GvTXHtBqPysg_xex4Kjm
# tar -xf PACS_final_generated.tar
# mv PACS_final_generated PACS_final
# rm PACS_final_generated.tar

### sampling exp ###
# ./gdrive files download 12GrpmJyI99uN6CgJV96rMwjnTtSuS0-x
# ./gdrive files download 1mhR4vVbOqj8cjosdPHltB-HHXqq4D9ia
# ./gdrive files download 1eN6mBuxJ39WPQuuHmN-ly2vZrA2XVqbu
# ./gdrive files download 1hc4YCvHIge-S6vJ6gN7NYvUoo32F1S-R
# ./gdrive files download 1-sDloBihAs-BbtpEiLvML3skiw-eK1OA

# tar -xf PACS_final_sampling_4.tar
# tar -xf PACS_final_sampling_2.tar
# tar -xf PACS_final_sampling_0_5.tar
# tar -xf PACS_final_sampling_0_25.tar
# tar -xf PACS_final_sampling_0_125.tar

# mv PACS_final_sampling_4 PACS_final
# mv PACS_final_sampling_2 PACS_final
# mv PACS_final_sampling_0.5 PACS_final
# mv PACS_final_sampling_0_25 PACS_final
# mv PACS_final_sampling_0_125 PACS_final

# rm PACS_final_sampling_4.tar
# rm PACS_final_sampling_2.tar
# rm PACS_final_sampling_0_5.tar
# rm PACS_final_sampling_0_25.tar
# rm PACS_final_sampling_0_125.tar

### RMD exp ###
# ./gdrive files download 1JjFWjRYgPaX4F63blUg3mbbINFsYDWdX
# ./gdrive files download 1yCbcctL0j69F1o-jCbG8PLAzMfIB7lm4
# ./gdrive files download 1_LCPreeDUjn2A5fJ_pH5rRbmx-e2jBK8

# tar -xf PACS_final_bottomk.tar
# tar -xf PACS_final_topk.tar
# tar -xf PACS_final_inverseprob.tar

# mv PACS_final_bottomk PACS_final
# mv PACS_final_topk PACS_final
# mv PACS_final_inverseprob PACS_final

# rm PACS_final_bottomk.tar
# rm PACS_final_topk.tar
# rm PACS_final_inverseprob.tar

### RMD_norm exp ###
# ./gdrive files download 1JIdlOUaW_Y8_95st9Yi-HUWTLmUVHM4u
# ./gdrive files download 1onZlzK8x22NDVcFyofxt_JZ-UGLQXynV
# ./gdrive files download 10wRcD-r73nCBbc2e4Jyo0b2DFD8LSHnA

# tar -xf PACS_final_normalized_temp_0_5.tar
# tar -xf PACS_final_normalized_temp_1.tar
# tar -xf PACS_final_normalized_temp_3.tar

# mv PACS_final_normalized_temp_0_5 PACS_final
# mv PACS_final_normalized_temp_1 PACS_final
# mv PACS_final_normalized_temp_3 PACS_final

# rm PACS_final_normalized_temp_0_5.tar
# rm PACS_final_normalized_temp_1.tar
# rm PACS_final_normalized_temp_3.tar

# ./gdrive files download 1Z5c33H3EvnEmjuHd7MKoMN_O_ztG_Tra
# ./gdrive files download 1redm8e6M3vXFb6439MIPHgkqNWvbrsPI

# tar -xf PACS_final_RMD_web_temp_2.tar
# tar -xf PACS_final_RMD_web_temp_3.tar

# mv PACS_final_RMD_web_temp_2 PACS_final/PACS_final_web_RMD_temp_2
# mv PACS_final_RMD_web_temp_3 PACS_final/PACS_final_web_RMD_temp_3

# rm PACS_final_RMD_web_temp_2.tar
# rm PACS_final_RMD_web_temp_3.tar

# ./gdrive files download 1SO7kP1Le0OeUTIZDvWqKro4IFvmIg_B-
# ./gdrive files download 19sOeOcz1ePMwzphWRK7mS7yFDPfCjLq3
# ./gdrive files download 19lRuuu5e1j-YugmT6B9dQRo9YBtiOs7a
# ./gdrive files download 1R52S0_svDa2GI94yPTP2M0lbusD7Kmq3
# ./gdrive files download 1dcZcb7EO6iHh7lOzeyOXjt1MouF1k7Y5
# ./gdrive files download 1x59JUQDzJQ7-MCgDPvd21trdKDof96rv
# ./gdrive files download 1RHp4Y5lDFcl2M8kzRfPCpyGhkmHFtpkH

# tar -xf PACS_final_web_equalweight.tar
# tar -xf PACS_final_web_topk.tar
# tar -xf PACS_final_web_bottomk.tar
# tar -xf PACS_final_web_RMD_temp_0_5.tar
# tar -xf PACS_final_web_RMD_temp_1.tar
# tar -xf PACS_final_web_RMD_temp_2.tar
# tar -xf PACS_final_web_RMD_temp_3.tar

# mv PACS_final_web_RMD_temp_0_5 PACS_final
# mv PACS_final_web_RMD_temp_1 PACS_final
# mv PACS_final_web_RMD_temp_2 PACS_final
# mv PACS_final_web_RMD_temp_3 PACS_final
# mv PACS_final_web_bottomk PACS_final
# mv PACS_final_web_topk PACS_final
# mv PACS_final_web_equalweight PACS_final

# rm PACS_final_web_equalweight.tar
# rm PACS_final_web_topk.tar
# rm PACS_final_web_bottomk.tar
# rm PACS_final_web_RMD_temp_0_5.tar
# rm PACS_final_web_RMD_temp_1.tar
# rm PACS_final_web_RMD_temp_2.tar
# rm PACS_final_web_RMD_temp_3.tar

# ./gdrive files download 1Z5kSh5A9l3A4VOgn-0XmBfeckPmdTamx
# ./gdrive files download 16EL80IAe6nNVNliwsudn75sH0crYXakt
# ./gdrive files download 1MDJbthhSPmjUtjIH9dHbOUEaLcjVCXkD

# tar -xf PACS_final_web_inverse_temp_0_5.tar
# tar -xf PACS_final_bing.tar
# tar -xf PACS_final_flickr.tar

# mv PACS_final_web_inverse_temp_0_5 PACS_final 
# mv PACS_final_bing PACS_final
# mv PACS_final_flickr PACS_final

# rm PACS_final_web_inverse_temp_0_5.tar
# rm PACS_final_bing.tar
# rm PACS_final_flickr.tar

# ./gdrive files download 1HQl3f0pc4501b_L1zmmlYHE-eyb-Y3bw
# ./gdrive files download 17sbLbq8ZiCs3wgziSANfGKuuu8PDODOg

# tar -xf PACS_final_web_RMD_temp_0_5_WF.tar
# tar -xf PACS_final_web_inverse_temp_0_5_WF.tar

# mv PACS_final_web_RMD_temp_0_5_WF PACS_final 
# mv PACS_final_web_inverse_temp_0_5_WF PACS_final

# rm PACS_final_web_RMD_temp_0_5_WF.tar
# rm PACS_final_web_inverse_temp_0_5_WF.tar

# ./gdrive files download 1iDEAWYiIGBDfND-nFTeNIGl3-SLWS5pp
# tar -xf PACS_final_all_samples_prob.tar
# mv PACS_final_all_samples_prob PACS_final
# rm PACS_final_all_samples_prob.tar

# ./gdrive files download 1PpO6kRVc2nY352Jt0HuiAIiZLoHhXfxb
# ./gdrive files download 1w90tb5ouFT7eunFEycP6w-hwNwj2my9M
# ./gdrive files download 1aT4Gr4uO_BGikghLLkPur-hJT2AMpmf4
# ./gdrive files download 1K1PkR5XTlY_U9lbuWdxIOEQUM-1ejQDS
# tar -xf PACS_final_twostage_temp_0_5.tar
# tar -xf PACS_final_twostage_temp_1.tar
# tar -xf PACS_final_twostage_temp_2.tar
# tar -xf PACS_final_twostage_temp_3.tar
# mv PACS_final_twostage_temp_0_5 PACS_final
# mv PACS_final_twostage_temp_1 PACS_final
# mv PACS_final_twostage_temp_2 PACS_final
# mv PACS_final_twostage_temp_3 PACS_final
# rm PACS_final_twostage_temp_0_5.tar
# rm PACS_final_twostage_temp_1.tar
# rm PACS_final_twostage_temp_2.tar
# rm PACS_final_twostage_temp_3.tar

# ./gdrive files download 1bjeREOe9rs-HrPGa8cDSxBZu4wN0DzkW
# ./gdrive files download 1zfinJr15WFJn3DVVCvf0vH8SuNynpGG1
# ./gdrive files download 1jsP1Y1lNlREtpIKz8rnbsI8O0ZIKfYTf
# ./gdrive files download 1_DOHSGD-aGELhi50KMF6riIgBzId2kDm
# tar -xf PACS_final_web_newsample_rmd_temp_1.tar
# tar -xf PACS_final_web_newsample_rmd_temp_2.tar
# tar -xf PACS_final_web_newsample_rmd_equalweight.tar
# tar -xf PACS_final_web_flickr.tar
# mv PACS_final_web_newsample_rmd_temp_1 PACS_final
# mv PACS_final_web_newsample_rmd_temp_2 PACS_final
# mv PACS_final_web_newsample_rmd_equalweight PACS_final
# mv PACS_final_web_flickr PACS_final
# rm PACS_final_web_newsample_rmd_temp_1.tar
# rm PACS_final_web_newsample_rmd_temp_2.tar
# rm PACS_final_web_newsample_rmd_equalweight.tar
# rm PACS_final_web_flickr.tar

# 0513 - PACS final web RMD ensembled with new sampling, samplewise
# ./gdrive files download 10IyfveOheSRPoGJ4R8srVNYPS1df2vvH
# ./gdrive files download 1_sosVhC8tC90q0ZtjL7SySwHM7XAZECx
# ./gdrive files download 1uML9G22Up4JuR1CglKj3h-pYMt-t31nN
# tar -xf PACS_final_web_all_samples_prob_temp_0_5.tar
# tar -xf PACS_final_web_all_samples_prob_temp_1.tar
# tar -xf PACS_final_web_all_samples_prob_temp_2.tar
# mv PACS_final_web_all_samples_prob_temp_0_5 PACS_final
# mv PACS_final_web_all_samples_prob_temp_1 PACS_final
# mv PACS_final_web_all_samples_prob_temp_2 PACS_final
# rm PACS_final_web_all_samples_prob_temp_0_5.tar
# rm PACS_final_web_all_samples_prob_temp_1.tar
# rm PACS_final_web_all_samples_prob_temp_2.tar

# ./gdrive files download 1iETOC7FFvmjz6qgYeY97gpdZFfKXSkdK
# ./gdrive files download 1KctDy6IjUtIrMjWpTC7ap1kDvVICp8A4
# tar -xf PACS_final_sdxl_diversified_new.tar
# tar -xf PACS_final_sdxl_diversified_new_filtered.tar
# mv PACS_final_sdxl_diversified_new PACS_final
# mv PACS_final_sdxl_diversified_new_filtered PACS_final
# rm PACS_final_sdxl_diversified_new.tar
# rm PACS_final_sdxl_diversified_new_filtered.tar

# ./gdrive files download 1VH69J_570FfhJi3qKGYWx162UPFC2t4X
# tar -xf PACS_final_sdxl_clip_50prompts.tar
# mv PACS_final_sdxl_clip_50prompts.tar PACS_final
# rm PACS_final_sdxl_clip_50prompts.tar

# ./gdrive files download 1WTLnVhYHi_Fga8dao-E-PonrBRcDwQOn
# ./gdrive files download 10wCuJT6HIIJ2gsqcr8VQ2ke3Hm0cMbk7
# ./gdrive files download 1IBcdwMpNNBbPK8uFrcK4VOttepwFGLID
# ./gdrive files download 1V0ixFCi_pc-SEThjAydDZ91hTKIE1kvx
# tar -xf PACS_final_static_50prompts.tar
# tar -xf PACS_final_static_50prompts_filtered.tar
# tar -xf PACS_final_dynamic_50prompts.tar
# tar -xf PACS_final_dynamic_50prompts_filtered.tar
# mv PACS_final_static_50prompts PACS_final/
# mv PACS_final_static_50prompts_filtered PACS_final/
# mv PACS_final_dynamic_50prompts PACS_final/
# mv PACS_final_dynamic_50prompts_filtered PACS_final/
# rm PACS_final_static_50prompts.tar
# rm PACS_final_static_50prompts_filtered.tar
# rm PACS_final_dynamic_50prompts.tar
# rm PACS_final_dynamic_50prompts_filtered.tar

# ./gdrive files download 1r1wx9fuSBKPrR2SLJKrWy-D2sQSLPcJq
# tar -xf PACS_final_clip_50prompts_filtered.tar
# mv PACS_final_clip_50prompts_filtered PACS_final/
# rm PACS_final_clip_50prompts_filtered.tar

# ./gdrive files download 1a4hnM7ODVE85IfpzROF_h1MEmRXSQ_Y_
# ./gdrive files download 1FUysw1aotHM9UNx2Llsvgfm7flv-zmo4
# tar -xf PACS_final_static_without_cot.tar
# tar -xf PACS_final_static_without_cot_filtered.tar
# mv PACS_final_static_without_cot PACS_final/
# mv PACS_final_static_without_cot_filtered PACS_final/
# rm PACS_final_static_without_cot.tar
# rm PACS_final_static_without_cot_filtered.tar

# ./gdrive files download 1YQhtzeDSNDjX7mmOQ7jvVsa1zDS3TRXF
# ./gdrive files download 1iUvzO4AU1fJxFRZmMKC223bcKY-Iq-8n
# tar -xf PACS_final_static_clip_100_50.tar
# tar -xf PACS_final_static_clip_100_50_filtered.tar
# mv PACS_final_static_clip_100_50 PACS_final/
# mv PACS_final_static_clip_100_50_filtered PACS_final/
# rm PACS_final_static_clip_100_50.tar
# rm PACS_final_static_clip_100_50_filtered.tar

# ./gdrive files download 1d0Ei3ZDOmYzIpvFupR6XPCLmBX9LLjA1
# ./gdrive files download 1EODNs-ejK0k3AK52pDuKoioigbsZlfYr
# tar -xf PACS_final_static_llama3_cot_50.tar
# tar -xf PACS_final_static_llama3_cot_50_filtered.tar
# mv PACS_final_static_llama3_cot_50 PACS_final/
# mv PACS_final_static_llama3_cot_50_filtered PACS_final/
# rm PACS_final_static_llama3_cot_50.tar
# rm PACS_final_static_llama3_cot_50_filtered.tar

# ./gdrive files download 1t4kSnkegYcP8WDB13M5qm5wtP-TFKy86
# ./gdrive files download 1ofwVnfpoJVXLE23nL18fMCGMF-p67ElF
# tar -xf PACS_final_static_gemini_cot_50.tar
# tar -xf PACS_final_static_gemini_cot_50_filtered.tar
# mv PACS_final_static_gemini_cot_50 PACS_final/
# mv PACS_final_static_gemini_cot_50_filtered PACS_final/
# rm PACS_final_static_gemini_cot_50.tar
# rm PACS_final_static_gemini_cot_50_filtered.tar

# ./gdrive files download 1L1yTIEVWj5WZ03hCqJG603FtvFKlxizY
# ./gdrive files download 11CmoopL-i0yJru2LrvTD_LY53Dd_u-nq
# ./gdrive files download 1N6jELNEuTs3ZPzXvW1WTxPifyY4lm1Kf
# ./gdrive files download 1I4wM-HzRh7ipjVvWWBF81ox9kR2GhQ9t
# ./gdrive files download 1jpuAOmMgTW-78D4rVSLOsk4kufTSdlgH
# tar -xf PACS_final_generated_RMD_equalweight.tar
# tar -xf PACS_final_generated_RMD_wo_normalize_temp2.tar
# tar -xf PACS_final_generated_RMD_w_normalize.tar
# tar -xf PACS_final_generated_RMD_w_normalize_clip_90.tar
# tar -xf PACS_final_generated_RMD_w_normalize_clip_95.tar
# mv PACS_final_generated_RMD_equalweight PACS_final/
# mv PACS_final_generated_RMD_wo_normalize_temp2 PACS_final/
# mv PACS_final_generated_RMD_w_normalize PACS_final/
# mv PACS_final_generated_RMD_w_normalize_clip_90 PACS_final/
# mv PACS_final_generated_RMD_w_normalize_clip_95 PACS_final/
# rm PACS_final_generated_RMD_equalweight.tar
# rm PACS_final_generated_RMD_wo_normalize_temp2.tar
# rm PACS_final_generated_RMD_w_normalize.tar
# rm PACS_final_generated_RMD_w_normalize_clip_90.tar
# rm PACS_final_generated_RMD_w_normalize_clip_95.tar

# ./gdrive files download 1Rcsxt6TBtvEq7lggy6C-CmMnWJmfsk-P
# tar -xf PACS_final_glide.tar
# mv PACS_final_glide PACS_final
# rm PACS_final_glide.tar

# ./gdrive files download 1ZVhbjYiGZGbGj41bNPQ804-V00UEASoy
# ./gdrive files download 1OqwOUWvcWw2JBAalMhAznS1j6UshXz9L
# tar -xf PACS_final_web_RMD_w_normalize_clip_90.tar
# tar -xf PACS_final_web_RMD_w_normalize_clip_95.tar
# mv PACS_final_web_RMD_w_normalize_clip_90 PACS_final/
# mv PACS_final_web_RMD_w_normalize_clip_95 PACS_final/
# rm -r PACS_final_web_RMD_w_normalize_clip_90.tar
# rm -r PACS_final_web_RMD_w_normalize_clip_95.tar

# ./gdrive files download 1IoxBKh9ok_MBauPUoZmZM7ZxgkUoF9o7
# ./gdrive files download 1VZUNKyuDJDh3GlIDibDCW7x3TmaR2nGq
# ./gdrive files download 1fDXLfp-OZweL6MGDfeUuADQUFLv7xgF_
# ./gdrive files download 1I-AqbA4KAdDJgK-RVJnALyCgamnqxSDr
# tar -xf PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5.tar
# tar -xf PACS_final_generated_RMD_w_normalize_clip_90_temp_2.tar
# tar -xf PACS_final_generated_RMD_w_normalize_clip_95_temp_0_5.tar
# tar -xf PACS_final_generated_RMD_w_normalize_clip_95_temp_2.tar
# mv PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5 PACS_final/
# mv PACS_final_generated_RMD_w_normalize_clip_90_temp_2 PACS_final/
# mv PACS_final_generated_RMD_w_normalize_clip_95_temp_0_5 PACS_final/
# mv PACS_final_generated_RMD_w_normalize_clip_95_temp_2 PACS_final/
# rm PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5.tar
# rm PACS_final_generated_RMD_w_normalize_clip_90_temp_2.tar
# rm PACS_final_generated_RMD_w_normalize_clip_95_temp_0_5.tar
# rm PACS_final_generated_RMD_w_normalize_clip_95_temp_2.tar

# ./gdrive files download 1sjAteSteeir6-cg9fjAjpSMHNK-rIR3R
# tar -xf PACS_final_generated_RMD_w_normalize_clip_90_temp_0_25.tar 
# mv PACS_final_generated_RMD_w_normalize_clip_90_temp_0_25 PACS_final/
# rm PACS_final_generated_RMD_w_normalize_clip_90_temp_0_25.tar

# ./gdrive files download 1Ug1547oBy43AHHx-5AfAXtIoSEMEUvwt
# tar -xf PACS_final_static_cot_50_palm2_filtered.tar
# mv PACS_final_static_cot_50_palm2_filtered PACS_final/
# rm PACS_final_static_cot_50_palm2_filtered.tar

# # Web from large
# ./gdrive files download 1CtTYenNaoTNa07qL0HvmYFrhjAa-810y
# ./gdrive files download 16X-A3EGYMKaGnd_nHWkJzy3pL0bTOxPu
# ./gdrive files download 1QdzoKAd1rRTioV-Vj7tppF7eqkde8rci
# tar -xf PACS_final_flickr_from_large_filtered.tar
# tar -xf PACS_final_web_from_large_equalweight.tar
# tar -xf PACS_final_web_from_large_RMD_w_normalize_clip_90_temp_0_25.tar
# mv PACS_final_flickr_from_large_filtered PACS_final/
# mv PACS_final_web_from_large_equalweight PACS_final/
# mv PACS_final_web_from_large_RMD_w_normalize_clip_90_temp_0_25 PACS_final/
# rm PACS_final_flickr_from_large_filtered.tar
# rm PACS_final_web_from_large_equalweight.tar
# rm PACS_final_web_from_large_RMD_w_normalize_clip_90_temp_0_25.tar

# # PACS final inverse, topk, bottomk
# ./gdrive files download 18LZpP_z4Cl83Ie9aeelROccotXhQ143u
# ./gdrive files download 1qiSdox18FFkAUq8XAOjdkAshYfhDNcVU
# ./gdrive files download 1_BBN1ixlz7r1s64zDqziWcgBQGF-xfM
# tar -xf PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5_inverse.tar
# tar -xf PACS_final_generated_RMD_topk.tar
# tar -xf PACS_final_generated_RMD_bottomk.tar
# mv PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5_inverse PACS_final/
# mv PACS_final_generated_RMD_topk PACS_final/
# mv PACS_final_generated_RMD_bottomk PACS_final/
# rm PACS_final_generated_RMD_w_normalize_clip_90_temp_0_5_inverse.tar
# rm PACS_final_generated_RMD_topk.tar
# rm PACS_final_generated_RMD_bottomk.tar

# # # Web from large 2
# ./gdrive files download 1wK8JJHN0q7hvy_8DzJfXUzQOio7J4IZj
# ./gdrive files download 1XIOAgTay3shxMDPJIR90uICmGTPYGTHu
# tar -xf PACS_final_web_from_large2_equalweight.tar
# tar -xf PACS_final_web_from_large2_RMD_w_normalize_clip_90_temp_0_5.tar
# mv PACS_final_web_from_large2_equalweight PACS_final/
# mv PACS_final_web_from_large2_RMD_w_normalize_clip_90_temp_0_5 PACS_final/
# rm PACS_final_web_from_large2_equalweight.tar
# rm PACS_final_web_from_large2_RMD_w_normalize_clip_90_temp_0_5.tar


## Prompt ablation
# ./gdrive files download 18GhnA5UcBdjpFeJ-ptmzivzYm3-XALC1
# ./gdrive files download 1xtTXKRQJ9IDQtmujQTcVrqffIAcT6sC4
# ./gdrive files download 1iq5t6nlnaeNNHJhQtt_gyUMHTNmESJJw
# tar -xf PACS_final_wo_cot.tar
# tar -xf PACS_final_wo_hierarchy.tar
# tar -xf PACS_final_wo_cot_wo_hierarchy.tar
# mv PACS_final_wo_cot PACS_final/
# mv PACS_final_wo_hierarchy PACS_final/
# mv PACS_final_wo_cot_wo_hierarchy PACS_final/
# rm PACS_final_wo_cot.tar
# rm PACS_final_wo_hierarchy.tar
# rm PACS_final_wo_cot_wo_hierarchy.tar


# PACS_final SDBP
# ./gdrive files download 1RvMTXuMvXyWqTyOHVAY4RTI2d4bjr8kO
# tar -xf PACS_final_sdbp.tar -C PACS_final
# rm PACS_final_sdbp.tar

# ./gdrive files download 14fCp3EQ5VUR6USzVaeSazML2QMsxhtkX
# tar -xf PACS_final_glide_diversified.tar -C PACS_final
# rm PACS_final_glide_diversified.tar


# # PACS_final sd3
# ./gdrive files download 1t80w99laMxGcOlFvAeXbgup3_LHcO1u_
# ./gdrive files download 1M0DZbkrOhfEqCb3OfseV9Xid1zshl1de
# tar -xf PACS_final_sd3.tar -C PACS_final
# tar -xf PACS_final_sd3_filtered.tar -C PACS_final
# rm PACS_final_sd3.tar
# rm PACS_final_sd3_filtered.tar

# # # PACS_final sd3 ablation (wo filtering)
# ./gdrive files download 1EyL5IzkXM_pCvOVHhrUGVB3aD6d0-3K2
# ./gdrive files download 1MhpeHjSEBFyRrMwCSQ_W7XTIebnWgoUV
# ./gdrive files download 1wuCYLUEtDQQomyvH8nZQjtir2dw8Xj97
# tar -xf PACS_final_sd3_wo_hierarchy.tar -C PACS_final
# tar -xf PACS_final_sd3_wo_cot.tar -C PACS_final
# tar -xf PACS_final_sd3_wo_hierarchy_wo_cot.tar -C PACS_final
# rm PACS_final_sd3_wo_hierarchy.tar
# rm PACS_final_sd3_wo_cot.tar
# rm PACS_final_sd3_wo_hierarchy_wo_cot.tar


# # PACS_final sdxl ablation (wo filtering)
# ./gdrive files download 1R54KSHjXGiGGMDwNc7sqpdxWnOGKfhFh
# ./gdrive files download 1gR_1IM3Kp2YQHfV3AOb9gOoQaCpbUY9v
# ./gdrive files download 1bCS72hFcYQ78waYH0jLpPjZEbHpS5t7A
# ./gdrive files download 1F31Jtm9CVUHTEsd8n__5BGzkz2Rq6CjA
# tar -xf PACS_final_sdxl.tar -C PACS_final
# tar -xf PACS_final_sdxl_wo_hierarchy.tar -C PACS_final
# tar -xf PACS_final_sdxl_wo_cot.tar -C PACS_final
# tar -xf PACS_final_sdxl_wo_hierarchy_wo_cot.tar.tar -C PACS_final
# rm PACS_final_sdxl.tar
# rm PACS_final_sdxl_wo_hierarchy.tar
# rm PACS_final_sdxl_wo_cot.tar
# rm PACS_final_sdxl_wo_hierarchy_wo_cot.tar.tar


# # PACS_final sdxl grid search (for ablation)
# ./gdrive files download 1WADZitzQlhF0Rg4SXQzec5M-UWy0TdLQ
# ./gdrive files download 15_5pIHerbpBYqmT8tvUOqNvwjUUHlFBp
# ./gdrive files download 13SucW1Ylqqvty4ymWcY2BL6UmhBbLh0N
# tar -xf PACS_final_sdxl_1.tar -C PACS_final
# tar -xf PACS_final_sdxl_2.tar -C PACS_final
# tar -xf PACS_final_sdxl_3.tar -C PACS_final
# rm PACS_final_sdxl_1.tar
# rm PACS_final_sdxl_2.tar
# rm PACS_final_sdxl_3.tar

# # PACS_final sdxl_1 filtering (to watch the filtering difference)
# ./gdrive files download 1s-GADySBoL_owVqfuykGp_xwCkT5x0Bi
# tar -xf PACS_final_sdxl_1_filtered.tar -C PACS_final
# rm PACS_final_sdxl_1_filtered.tar


# # 0722 RMD experiment (dalle2 alternative)
# # Compare kandinsky2, karlo, sd3, sdturbo
# ./gdrive files download 1e8nK422czexVVdyLLpgkKWjpPLTJJWbW
# ./gdrive files download 1XOke56yV1YQavKN_6-9POPgB_7U4ZiVF
# ./gdrive files download 1HjCisHVDo85rI4nnEgrYU9ryU2z1lGTp
# ./gdrive files download 1_u7msLSSBr6UyqXpT7sSbNA2LI2KT_Ar
# ./gdrive files download 1-UG4f2S8_ZnHKTTkc0NQbP-Fxr6VX9ka
# ./gdrive files download 1UGtyYxgB0yZpRfZw2yGYustZwmRwbYm-
# ./gdrive files download 12daGuQTYjF5-g8kwzbWCGDEBvZAHyFc4
# ./gdrive files download 1lHbviYFmEBFxeJEY9d0mv0_Ayju4uAvT
# tar -xf PACS_final_kandinsky2_equalweight.tar -C PACS_final
# tar -xf PACS_final_kandinsky2_RMD.tar -C PACS_final
# tar -xf PACS_final_karlo_equalweight.tar -C PACS_final
# tar -xf PACS_final_karlo_RMD.tar -C PACS_final
# tar -xf PACS_final_sd3_equalweight.tar -C PACS_final
# tar -xf PACS_final_sd3_RMD.tar -C PACS_final
# tar -xf PACS_final_sdturbo_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdturbo_RMD.tar -C PACS_final
# rm PACS_final_kandinsky2_equalweight.tar
# rm PACS_final_kandinsky2_RMD.tar
# rm PACS_final_karlo_equalweight.tar
# rm PACS_final_karlo_RMD.tar
# rm PACS_final_sd3_equalweight.tar
# rm PACS_final_sd3_RMD.tar
# rm PACS_final_sdturbo_equalweight.tar
# rm PACS_final_sdturbo_RMD.tar

# # 0825 RMD check
# ./gdrive files download 1AKEharuL1i7V6cn65pQ1czNlMqQeagWN # PACS_final_sdxl_floyd_cogview2_sd3.tar
# ./gdrive files download 1aSg_xyyuH_XjSSPBxBu28h9oKsMoL_6o # PACS_final_sdxl_floyd_cogview2_sd3_flux.tar
# ./gdrive files download 1tsdgpXJuTNqSy9LVZ-ZESYQXk-Rce7p1 # PACS_final_sdxl_floyd_cogview2_sd3_auraflow.tar
# ./gdrive files download 1CVKBwq7VxgTC7ciNgHO4_NCCl0uz1OEx # PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors.tar
# ./gdrive files download 1I9EqJYvRrs6edpsmUQSZtWmGRLxEI6Oy # PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# ./gdrive files download 1xJNFyKGiU1ZLNTQlhdkrM4l118sGsYYH # PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow.tar
# ./gdrive files download 1JJlW6fsL1MBhN4vjk5cVQ5RIMLkopXJh # PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# rm PACS_final_sdxl_floyd_cogview2_sd3.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar

# # 0825 RMD check equalweight
# ./gdrive files download 1t61yPBbJGkZX20olhBMap0xgNaiqx1NF # PACS_final_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar
# ./gdrive files download 1Nku6TVVlPRHqdn_KQun5cLAEU4ljy6dp # PACS_final_sdxl_floyd_cogview2_sd3_equalweight.tar
# ./gdrive files download 1mRprtAIj7AFM5YDDq6bkthE1WeSOdK8d # PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow_equalweight.tar
# ./gdrive files download 1pjuh0xq6jUcsrjA8wFrFfAwCFzZAl_Ty # PACS_final_sdxl_floyd_cogview2_sd3_flux_equalweight.tar
# ./gdrive files download 1Z-uNoF8KaShpn1dA82W4VewWLHqDj1C1 # PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_equalweight.tar
# ./gdrive files download 1a2Nhx6hcNmgWGGMmC4qQN8cv8sxFAqY- # PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_equalweight.tar
# ./gdrive files download 10UczXeaI7J9LXm-CaynBcTPaCmaTqNee # PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow_equalweight.tar
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow_equalweight.tar -C PACS_final
# rm PACS_final_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_auraflow_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_equalweight.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_kolors_auraflow_equalweight.tar

# # glister, uncertainty
# ./gdrive files download 1ha3SzK5sYdo-yoK8CIdyw6WqhDBXdehm # PACS_final_glister_0_01.tar
# ./gdrive files download 1h7wj6gm_ckR1cjgvy_aofSAjL_DzvB8N # PACS_final_glister_0_001.tar
# ./gdrive files download 1WVjcpQyxNEJpY9_bpzNnKGPvuhJ6g8eU # PACS_final_uncertainty_0_01.tar
# ./gdrive files download 1pXua6E2K5S8Q1UicFutgvTtH8FQe3OrH # PACS_final_uncertainty_0_001.tar
# tar -xf PACS_final_glister_0_01.tar -C PACS_final
# tar -xf PACS_final_glister_0_001.tar -C PACS_final
# tar -xf PACS_final_uncertainty_0_01.tar -C PACS_final
# tar -xf PACS_final_uncertainty_0_001.tar -C PACS_final
# rm PACS_final_glister_0_01.tar
# rm PACS_final_glister_0_001.tar
# rm PACS_final_uncertainty_0_01.tar
# rm PACS_final_uncertainty_0_001.tar 

# # moderate
# ./gdrive files download 1LiJ5r1Jz_pdsHhpSpJlDkCp_uENXwsY3
# tar -xf PACS_final_moderate.tar -C PACS_final
# rm PACS_final_moderate.tar

# # robustness
# ./gdrive files download 1htI1WExpktTznQuWS2KRGTd3GWAGyqf6
# tar -xf PACS_final_robustness.tar -C PACS_final
# rm PACS_final_robustness.tar

# # RMD scores using DINO
# ./gdrive files download 1RSIixqYuRiAGQQ7EEjhhQFICV69ppNoL # dino vitb16
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino.tar -C PACS_final
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino.tar

# # RMD scores using DINO (compare patch size)
# ./gdrive files download 1yYzm8jPzFf6YgAFgSuBoR9tGubkFSocY # dino vitb8
# ./gdrive files download 1h7TQ_HslIhqzZp53FKApi2kSVdtQd-Aa # dino vits16
# ./gdrive files download 1k_B1VklXIoPOGB3IqKOTK4jyVzqTiMUx # dino vits8
# ./gdrive files download 1ruL-R-ngSkrmQn3JTScc6v7t1iV_WLg7 # dino vitb16 normalized
# ./gdrive files download 1b7bz3qgmiI00fBMqZ3-UakgMl7CbPejy # dion vitb8 normalized
# ./gdrive files download 1J_33KdtD8-aJK32Ms24F3TJJ8k1CDwL6 # dino vits16 normalized
# ./gdrive files download 1rq6q7ef0BdOn6gYMt8OoGzWom08Vz79y # dino vits8 normalized
# ./gdrive files download 1Xsfd02buhaurgURA6xkD-JRL07Rqvu4S # dinov2
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb8.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits16.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb16_normalized.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb8_normalized.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits16_normalized.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8_normalized.tar -C PACS_final
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dinov2_base.tar -C PACS_final
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb8.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits16.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb16_normalized.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vitb8_normalized.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits16_normalized.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dino_vits8_normalized.tar
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_dinov2_base.tar

# # RMD baseline added
# ./gdrive files download 1gi6MTVt8FklKM4X4z3tlAYb8_F4elOLw
# ./gdrive files download 1LrmZIAFgJP-53d7V68M6UseSUYU9a6L9
# ./gdrive files download 1MH70qM72RxN0ngB88AtmGlg2a9B_tjGh
# ./gdrive files download 1xhSWq8TmKvuhmAyHfr7Gx3sKC43-PdEt
# ./gdrive files download 1r8uO16OB84zCNOfTzDMKY5WSZFT0B3gF
# ./gdrive files download 1IovY_YBbJMdrTSajZXu6Dgtk762DLA-3
# ./gdrive files download 1xGePXLaVrsFkNVtl05owx4b0XYZwdujt
# ./gdrive files download 1_PSISVHsrxCM3cyWGTivgSdrIH2P5bo-
# ./gdrive files download 1Cgsu2TJNuG0RoIIleDWrVxN5ur3t_lLW
# ./gdrive files download 1TmX5yYI22t-hm3oigoiyk4ICK_M2BNLt
# ./gdrive files download 1oMZuB0KvqAYEvEIBHzJXm_yzpGC_-cif
# ./gdrive files download 1v4ETA4qiLeM4QQ78u_G7RhqoHPJrn20T
# tar -xf PACS_final_CLIP_Uncertainty_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_Uncertainty_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_Uncertainty_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_GradMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_GradMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_GradMatch_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_Glister_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_Glister_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_Glister_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_CurvMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_CurvMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_CLIP_CurvMatch_10_0_0001.tar -C PACS_final
# rm PACS_final_CLIP_Uncertainty_50_0_0001.tar
# rm PACS_final_CLIP_Uncertainty_25_0_0001.tar
# rm PACS_final_CLIP_Uncertainty_10_0_0001.tar
# rm PACS_final_CLIP_GradMatch_50_0_0001.tar
# rm PACS_final_CLIP_GradMatch_25_0_0001.tar
# rm PACS_final_CLIP_GradMatch_10_0_0001.tar
# rm PACS_final_CLIP_Glister_50_0_0001.tar
# rm PACS_final_CLIP_Glister_25_0_0001.tar
# rm PACS_final_CLIP_Glister_10_0_0001.tar
# rm PACS_final_CLIP_CurvMatch_50_0_0001.tar
# rm PACS_final_CLIP_CurvMatch_25_0_0001.tar
# rm PACS_final_CLIP_CurvMatch_10_0_0001.tar

# # RMD baselines new
# ./gdrive files download 1Sgjsmz5rYOPw_qtSvvO6Ij2ER88aTL8M # PACS_final_DINO_small_moderate_filtered
# ./gdrive files download 1qjqWalxNupajp3EGw8pk1rpmfPNJ2_5v # PACS_final_DINO_base_moderate_filtered
# ./gdrive files download 1D68O3smkFqAAG0kvG_pXoLDEpQJvhFxe # PACS_final_CLIP_moderate_filtered
# ./gdrive files download 1tNJb-8MoP3_6j9FiFdZYqaU6x1Rm5ops # PACS_final_DINO_base_CurvMatch_10_0_0001
# ./gdrive files download 1QhHGGiaYMtYSH7E5AUl8G-rfffnrvmTm # PACS_final_DINO_base_CurvMatch_25_0_0001
# ./gdrive files download 1LsWCuFOFqg84ytGuqk2gUv2SeaAYdr3Z # PACS_final_DINO_base_CurvMatch_50_0_0001
# ./gdrive files download 1JeE84GVJkMc0E95tC5E0mixeoNEkDNu3 # PACS_final_DINO_base_Glister_10_0_0001
# ./gdrive files download 1vZj26htFwUdHWQS6H7IGs260DSCW30te # PACS_final_DINO_base_Glister_25_0_0001
# ./gdrive files download 10riEeANzzRrwQduyrLKwpzdDcYlG_-24 # PACS_final_DINO_base_Glister_50_0_0001
# ./gdrive files download 1tbNqxbc57jyTrrypsGrVkN6TSLUfnRKl # PACS_final_DINO_base_GradMatch_10_0_0001
# ./gdrive files download 1J6NZeMK2_vGVgzTTCe20k48WZ7ukH-jV # PACS_final_DINO_base_GradMatch_25_0_0001
# ./gdrive files download 1wpVeKDHO0Z7Xd6Kyyu6K72gFwdQ0fIDM # PACS_final_DINO_base_GradMatch_50_0_0001
# ./gdrive files download 1ug3s_Ugg7lYKiMk_TdoS0CINwm08wi3X # PACS_final_DINO_base_Uncertainty_10_0_0001
# ./gdrive files download 1J7z1cMBZSc5ceQ6ygsmbVpJekDZk4nDF # PACS_final_DINO_base_Uncertainty_25_0_0001
# ./gdrive files download 19ZpmwiLx1q3l-2uDeu-omZmdC8aM4ILd # PACS_final_DINO_base_Uncertainty_50_0_0001
# ./gdrive files download 19V-eWBM-ckqht82tZHqlLsVpmel3BgJH # PACS_final_DINO_small_CurvMatch_10_0_0001
# ./gdrive files download 1i9dIHbRaqLLA1Zy1QDBgBimfAZHiO11J # PACS_final_DINO_small_CurvMatch_25_0_0001
# ./gdrive files download 10wUf8of-nO5gMIQWHphpVDXxWTzKeEyw # PACS_final_DINO_small_CurvMatch_50_0_0001
# ./gdrive files download 1p7_tsfc-MgV0s8cRLxFlHEhfRw3L9znb # PACS_final_DINO_small_Glister_10_0_0001
# ./gdrive files download 1IEhgz77GbcF9DOnjp5nZKneGCVmFzJRX # PACS_final_DINO_small_Glister_25_0_0001
# ./gdrive files download 1DiKaj6t6Q9nOS6sShmMM2_PmeEzydTfq # PACS_final_DINO_small_Glister_50_0_0001
# ./gdrive files download 1BTbWIIJHlmcSq2k_fx_x0iOxIpDu4KzB # PACS_final_DINO_small_GradMatch_10_0_0001
# ./gdrive files download 1gIF_I04wtoXeLTzYsIK8S_yh4rbQUgK- # PACS_final_DINO_small_GradMatch_25_0_0001
# ./gdrive files download 175r12n7CcOw8MNSjwQkte7zMNEt1VkXF # PACS_final_DINO_small_GradMatch_50_0_0001
# ./gdrive files download 1rKrzjk9LbTe1qtcxhf5_jjSSxaG__EUX # PACS_final_DINO_small_Uncertainty_10_0_0001
# ./gdrive files download 1LI5mfI9DDOErH-33dWot1jQQu27-o5oI # PACS_final_DINO_small_Uncertainty_25_0_0001
# ./gdrive files download 1feUMVmXK1BxxlhbzKSL6FlCEMb1v4Zgj # PACS_final_DINO_small_Uncertainty_50_0_0001
# tar -xf PACS_final_DINO_small_moderate_filtered.tar -C PACS_final
# tar -xf PACS_final_DINO_base_moderate_filtered.tar -C PACS_final
# tar -xf PACS_final_CLIP_moderate_filtered.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Uncertainty_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Uncertainty_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Uncertainty_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_CurvMatch_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_CurvMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_CurvMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Glister_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Glister_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Glister_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_GradMatch_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_GradMatch_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_GradMatch_50_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Uncertainty_10_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Uncertainty_25_0_0001.tar -C PACS_final
# tar -xf PACS_final_DINO_small_Uncertainty_50_0_0001.tar -C PACS_final
# rm PACS_final_DINO_small_moderate_filtered.tar
# rm PACS_final_DINO_base_moderate_filtered.tar
# rm PACS_final_CLIP_moderate_filtered.tar
# rm PACS_final_DINO_base_CurvMatch_10_0_0001.tar
# rm PACS_final_DINO_base_CurvMatch_25_0_0001.tar
# rm PACS_final_DINO_base_CurvMatch_50_0_0001.tar
# rm PACS_final_DINO_base_Glister_10_0_0001.tar
# rm PACS_final_DINO_base_Glister_25_0_0001.tar
# rm PACS_final_DINO_base_Glister_50_0_0001.tar
# rm PACS_final_DINO_base_GradMatch_10_0_0001.tar
# rm PACS_final_DINO_base_GradMatch_25_0_0001.tar
# rm PACS_final_DINO_base_GradMatch_50_0_0001.tar
# rm PACS_final_DINO_base_Uncertainty_10_0_0001.tar
# rm PACS_final_DINO_base_Uncertainty_25_0_0001.tar
# rm PACS_final_DINO_base_Uncertainty_50_0_0001.tar
# rm PACS_final_DINO_small_CurvMatch_10_0_0001.tar
# rm PACS_final_DINO_small_CurvMatch_25_0_0001.tar
# rm PACS_final_DINO_small_CurvMatch_50_0_0001.tar
# rm PACS_final_DINO_small_Glister_10_0_0001.tar
# rm PACS_final_DINO_small_Glister_25_0_0001.tar
# rm PACS_final_DINO_small_Glister_50_0_0001.tar
# rm PACS_final_DINO_small_GradMatch_10_0_0001.tar
# rm PACS_final_DINO_small_GradMatch_25_0_0001.tar
# rm PACS_final_DINO_small_GradMatch_50_0_0001.tar
# rm PACS_final_DINO_small_Uncertainty_10_0_0001.tar
# rm PACS_final_DINO_small_Uncertainty_25_0_0001.tar
# rm PACS_final_DINO_small_Uncertainty_50_0_0001.tar

# # RMD CLIP vit patch 32 -> 16, moderate new
# ./gdrive files download 1qOL0lAKJkqBwcLjbdD8-Nf8GDR_0G85t # PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_clip_16
# ./gdrive files download 1gjSGx-6hzeVIQiQDPEzvjZCQdsU9Er8U # PACS_final_CLIP_moderate
# tar -xf PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_clip_16.tar -C PACS_final
# tar -xf PACS_final_CLIP_moderate.tar -C PACS_final
# rm PACS_final_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow_clip_16.tar
# rm PACS_final_CLIP_moderate.tar


# ./gdrive files download 1ltM_SRqnJfkHn-XO8yonsrfAjgZCLp96 # PACS_final_fake_sdxl
# tar -xf PACS_final_fake_sdxl.tar -C PACS_final
# rm PACS_final_fake_sdxl.tar

# ./gdrive files download 1v7qAxHHFeXFDhJdfsNcdLtbo-wj_NzXz # PACS_final_f_sdxl
# tar -xf PACS_final_f_sdxl.tar -C PACS_final
# rm PACS_final_f_sdxl.tar

# # synclr, synthclip
# ./gdrive files download 1Qty4rkNHglP2Hqn3DxC3BbXIj-LS9oQZ # PACS_final_synclr
# ./gdrive files download 196EE_qt7GA_NFuicH8OqRpwpwbou2YfG # PACS_final_synthclip
# tar -xf PACS_final_synclr.tar -C PACS_final
# tar -xf PACS_final_synthclip.tar -C PACS_final
# rm PACS_final_synclr.tar
# rm PACS_final_synthclip.tar

# # LE diversified
# ./gdrive files download 1gd7OgD2tJcKP4YPJufVxrkC_MJL3fUN9 # PACS_final_le_diversified
# tar -xf PACS_final_le_diversified.tar -C PACS_final
# rm PACS_final_le_diversified.tar

# # Refined prompts
# ./gdrive files download 1dA3u_b7bAS4NKHwuXbJn2zmNVDI375R2 # PACS_final_sdxl_1_refined
# tar -xf PACS_final_sdxl_1_refined.tar -C PACS_final
# rm PACS_final_sdxl_1_refined.tar

# # with new dynamic prompt (0913)
# ./gdrive files download 1QHqh7J7uIPecoMfvrbHbi5Tyv0yv4ixV # PACS_final_dynamic_sdxl
# tar -xf PACS_final_dynamic_sdxl.tar -C PACS_final
# rm PACS_final_dynamic_sdxl.tar

# # static to dynamic
# ./gdrive files download 1Tznceu3wKAo0WTvm85chWmJuf1Q4kFyp # PACS_static2dynamic
# tar -xf PACS_static2dynamic.tar -C PACS_final
# rm PACS_static2dynamic.tar

# # with 100 prompts (0914)
# ./gdrive files download 1XkYsl6LttcLk3ddzVU87RLypg8A9FtcL # PACS_final_cot_100_sdxl
# ./gdrive files download 1RfBigQphks82mmaJ7nVy74cjHSnpcH6h # PACS_final_glide_100_sdxl
# tar -xf PACS_final_cot_100_sdxl.tar -C PACS_final
# tar -xf PACS_final_glide_100_sdxl.tar -C PACS_final
# rm PACS_final_cot_100_sdxl.tar
# rm PACS_final_glide_100_sdxl.tar

# # 50, 100 prompts (new)
# ./gdrive files download 1S38W-kHC3bnEdluA6sHXr4WONd2-djMi # PACS_final_cot_50_sdxl
# ./gdrive files download 11tw17SDQocf---cU_UZpiPudnIz3vaNc # PACS_final_cot_100_sdxl
# tar -xf PACS_final_cot_50_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_sdxl.tar -C PACS_final
# rm PACS_final_cot_50_sdxl.tar
# rm PACS_final_cot_100_sdxl.tar

# # PACS_final_cot_50_1~3, 100_1~3, sdxl (0914)
# ./gdrive files download 1wJTdt0tBhoUbXY5v9TGBmhqzrZyX2vI7 # PACS_final_cot_50_1_sdxl
# ./gdrive files download 1jGCyHZIYez9kzsqCxfgmRIQjpPBTBphK # PACS_final_cot_50_2_sdxl
# ./gdrive files download 134qEyhb9LAqEuZxhE8ukiY50eNxulhB9 # PACS_final_cot_50_3_sdxl
# ./gdrive files download 1pTyp5cavrwInVwf17kgJo9ASiWv_niky # PACS_final_cot_100_1_sdxl
# ./gdrive files download 153iJc04PQUNFyQJ4_3KljGSXdRFCdmqw # PACS_final_cot_100_2_sdxl
# ./gdrive files download 1IHUD5uIS1qweeimqJ_Kal2fA4smdmY9Q # PACS_final_cot_100_3_sdxl
# tar -xf PACS_final_cot_50_1_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_50_3_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_1_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_3_sdxl.tar -C PACS_final
# rm PACS_final_cot_50_1_sdxl.tar
# rm PACS_final_cot_50_2_sdxl.tar
# rm PACS_final_cot_50_3_sdxl.tar
# rm PACS_final_cot_100_1_sdxl.tar
# rm PACS_final_cot_100_2_sdxl.tar
# rm PACS_final_cot_100_3_sdxl.tar

# # LE_diversified_100_sdxl and glide (0915)
# ./gdrive files download 1Y628Qu852mt03H94I3ZMW2n_NBSPXbuh # PACS_final_LE_diversified_100_sdxl
# ./gdrive files download 15jzY5ngnU459VWcmCYJdg11KNrwNfDzc # PACS_final_LE_diversified_100_glide
# tar -xf PACS_final_LE_diversified_100_sdxl.tar -C PACS_final
# tar -xf PACS_final_LE_diversified_100_glide.tar -C PACS_final
# rm PACS_final_LE_diversified_100_sdxl.tar
# rm PACS_final_LE_diversified_100_glide.tar

# # synclr, synthclip with 100 (0915)
# ./gdrive files download 1Vz1Nha_iznjfKNldqF6Qi5Kz442BDWsl # PACS_final_synclr_100_sdxl
# ./gdrive files download 1n_BpvjePYb0O2l0kLLgao8iThwsne24M # PACS_final_synthclip_100_sdxl
# tar -xf PACS_final_synclr_100_sdxl.tar -C PACS_final
# tar -xf PACS_final_synthclip_100_sdxl.tar -C PACS_final
# rm PACS_final_synclr_100_sdxl.tar
# rm PACS_final_synthclip_100_sdxl.tar

# # cot_50, 100_2 with refinement, sdxl 
# ./gdrive files download 16ackcN1uJue3AnoKV6XVuvIRNkiuxoMz # PACS_final_cot_50_2_refined_sdxl
# ./gdrive files download 1NTCqdTYzuiulT8G2NxK8u0qtCi_FXbIy # PACS_final_cot_100_2_refined_sdxl
# tar -xf PACS_final_cot_50_2_refined_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_refined_sdxl.tar -C PACS_final
# rm PACS_final_cot_50_2_refined_sdxl.tar
# rm PACS_final_cot_100_2_refined_sdxl.tar

# ./gdrive files download 1c6SwO5uIQuwrfqQyUkFq_SPU33xtDOQ3 # PACS_final_cot_100_contrast
# tar -xf PACS_final_cot_100_contrast.tar -C PACS_final
# rm PACS_final_cot_100_contrast.tar

# ./gdrive files download 1UMjkUApg6jldotL-MGMfbMLJ68U7rM-E # PACS_final_cot_50_dynamic_sdxl
# ./gdrive files download 15ykU_9ITuKP50Y-4BWnsQuzyB6NtuaTi # PACS_final_cot_100_dynamic_sdxl
# tar -xf PACS_final_cot_50_dynamic_sdxl.tar -C PACS_final
# tar -xf PACS_final_cot_100_dynamic_sdxl.tar -C PACS_final
# rm PACS_final_cot_50_dynamic_sdxl.tar
# rm PACS_final_cot_100_dynamic_sdxl.tar

# # cot_50_2, 100_2 ensemble
# ./gdrive files download 121G1RDtQojgUw8ivUKh1IBhqhKrY-FKL # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1JTIV4iiKmOq-lRabQOc9M7zIMAvDIlnr # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1zfh94zruZ6Eqs3Lu8WmPc1Vp_7WyZBK7 # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# ./gdrive files download 17w3rV0A9hsUckxiD3hfjB8IsMOgN65ZU # PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1A3r_z6EgzrIMZVyK1HoOXiE5-F8aqxdt # PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 10hADk5SlsR1TbAr294RR65xaAJgut1Mw # PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar
# rm PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_cot_100_2_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar

# # cot_50_2, 100_2 refined with new1, new2 / sdxl (0916)
# ./gdrive files download 1HHfuNcPEEXU2lbgUXYU6g7ojuv64c2r1 # PACS_final_cot_100_2_refined_new1
# ./gdrive files download 1FFegMf8rjSRuOT5F8xXkCg2T8PEdVvF- # PACS_final_cot_100_2_refined_new2
# ./gdrive files download 1uqQMSfzowL8Vz1OrQ75TkFZUZi2W1F20 # PACS_final_cot_50_2_refined_new1
# ./gdrive files download 1YCwlCZs5dYyhesVr9aBScUsBab43kEbh # PACS_final_cot_50_2_refined_new2
# tar -xf PACS_final_cot_100_2_refined_new1.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_refined_new2.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_refined_new1.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_refined_new2.tar -C PACS_final
# rm PACS_final_cot_100_2_refined_new1.tar
# rm PACS_final_cot_100_2_refined_new2.tar
# rm PACS_final_cot_50_2_refined_new1.tar
# rm PACS_final_cot_50_2_refined_new2.tar

# # cot_50_2, 10_2 refined ensemble (0917)
# ./gdrive files download 12U3UsxrzjrowCc-8dk2GOwVFopzTYqTp # PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1pLfTAQd4poa-b3NX69Fk8B2g818AowUA # PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1GSDlUDj5zNXJCkJLb1OzikNjrMllGz_8 # PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# ./gdrive files download 1y_PQYilPd8LBpJOO5bWac4EGKr3aYw_g # PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1CQmyRabkjkqCKEerwOlw3TLfkFiwHVQi # PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1Aamn4oOm1pka_z16d1Crs0RHt1VgBbG1 # PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# tar -xf PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# rm PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_cot_50_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar
# rm PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_cot_100_2_refined_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar

# # wo_cot_wo_hierarchy_50 ensemble (0918)
# ./gdrive files download 1F7lplnXy5X6xuuyO6yLnhULpLVJQSong # PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow
# ./gdrive files download 1maCdnCJNyz6Oe9PLyhJzvqNwzK_stmXR # PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1CuP5KgGUCeSwuR-UpUPGDvhkXbHCFx0d # PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar -C PACS_final
# tar -xf PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# rm PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_kolors_auraflow.tar
# rm PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_wo_cot_wo_hierarhcy_50_sdxl_floyd_cogview2_sd3_auraflow.tar

# 6 web_DINO_base (0919)
./gdrive files download 10JeVMxs-THLiTx5imPZqfoepXKuBFh01 # PACS_final_web_DINO_base_Adacore_30_0.0001
./gdrive files download 19d_YV0awo9Ye3ZU76XuxPn5xojHGBRfM # PACS_final_web_DINO_base_CurvMatch_30_0.0001
./gdrive files download 1Ud_w1HaZ-E-MQwztUcivo87p_WYdgDAy # PACS_final_web_DINO_base_Glister_30_0.0001
./gdrive files download 1gk79XmbxZaqDSnln3FEBW2Ko4r5q7yLa # PACS_final_web_DINO_base_GradMatch_30_0.0001
./gdrive files download 1R7r7fq8vfRwCVbWD5i7hkKDuS2zavB-E # PACS_final_web_DINO_base_Submodular_30_0.0001
./gdrive files download 1q-RK9SOW-oc-IhzK0Ce7jeEY-apiijcL # PACS_final_web_DINO_base_Uncertainty_30_0.0001
tar -xf PACS_final_web_DINO_base_Adacore_30_0.0001.tar -C PACS_final
tar -xf PACS_final_web_DINO_base_CurvMatch_30_0.0001.tar -C PACS_final
tar -xf PACS_final_web_DINO_base_Glister_30_0.0001.tar -C PACS_final
tar -xf PACS_final_web_DINO_base_GradMatch_30_0.0001.tar -C PACS_final
tar -xf PACS_final_web_DINO_base_Submodular_30_0.0001.tar -C PACS_final
tar -xf PACS_final_web_DINO_base_Uncertainty_30_0.0001.tar -C PACS_final
rm PACS_final_web_DINO_base_Adacore_30_0.0001.tar
rm PACS_final_web_DINO_base_CurvMatch_30_0.0001.tar
rm PACS_final_web_DINO_base_Glister_30_0.0001.tar
rm PACS_final_web_DINO_base_GradMatch_30_0.0001.tar
rm PACS_final_web_DINO_base_Submodular_30_0.0001.tar
rm PACS_final_web_DINO_base_Uncertainty_30_0.0001.tar