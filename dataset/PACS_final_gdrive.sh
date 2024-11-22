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

# ./gdrive files download 1v7qAxHHFeXFDhJdfsNcdLtbo-wj_NzXz # PACS_final fake_f
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

# # 6 web_DINO_base (0919)
# ./gdrive files download 10JeVMxs-THLiTx5imPZqfoepXKuBFh01 # PACS_final_web_DINO_base_Adacore_30_0.0001
# ./gdrive files download 19d_YV0awo9Ye3ZU76XuxPn5xojHGBRfM # PACS_final_web_DINO_base_CurvMatch_30_0.0001
# ./gdrive files download 1Ud_w1HaZ-E-MQwztUcivo87p_WYdgDAy # PACS_final_web_DINO_base_Glister_30_0.0001
# ./gdrive files download 1gk79XmbxZaqDSnln3FEBW2Ko4r5q7yLa # PACS_final_web_DINO_base_GradMatch_30_0.0001
# ./gdrive files download 1R7r7fq8vfRwCVbWD5i7hkKDuS2zavB-E # PACS_final_web_DINO_base_Submodular_30_0.0001
# ./gdrive files download 1q-RK9SOW-oc-IhzK0Ce7jeEY-apiijcL # PACS_final_web_DINO_base_Uncertainty_30_0.0001
# tar -xf PACS_final_web_DINO_base_Adacore_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_web_DINO_base_CurvMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_web_DINO_base_Glister_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_web_DINO_base_GradMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_web_DINO_base_Submodular_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_web_DINO_base_Uncertainty_30_0.0001.tar -C PACS_final
# rm PACS_final_web_DINO_base_Adacore_30_0.0001.tar
# rm PACS_final_web_DINO_base_CurvMatch_30_0.0001.tar
# rm PACS_final_web_DINO_base_Glister_30_0.0001.tar
# rm PACS_final_web_DINO_base_GradMatch_30_0.0001.tar
# rm PACS_final_web_DINO_base_Submodular_30_0.0001.tar
# rm PACS_final_web_DINO_base_Uncertainty_30_0.0001.tar

# # base_sdxl (0920)
# ./gdrive files download 1NCJ-UYhwAvsEIfyZqCf6qd8867dnW7bH # PACS_final_base_sdxl
# tar -xf PACS_final_base_sdxl.tar -C PACS_final
# rm PACS_final_base_sdxl.tar

# # LE_diversified_50_sdxl
# ./gdrive files download 1UyvSkbwfAnDKP9ZJUXTiOiB_4qfBKVlT # PACS_final_LE_diversified_50_sdxl
# tar -xf PACS_final_LE_diversified_50_sdxl.tar -C PACS_final
# rm PACS_final_LE_diversified_50_sdxl.tar

# # LE_50_glide
# ./gdrive files download 1BaRq2E6htBKQsWUOvZPw-Zf7-VEYOE9H # PACS_final_LE_diversified_50_glide
# tar -xf PACS_final_LE_diversified_50_glide.tar -C PACS_final
# rm PACS_final_LE_diversified_50_glide.tar

# # synclr, synthclip ensemble (0921)
# ./gdrive files download 1crsLSf2Cd1lk1PbNRlxNxJuKKqthwYQ2 # PACS_final_synclr_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1f8L36-wGDN-UOiz0dD2vIJndF-Py2zmD # PACS_final_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow
# ./gdrive files download 1aR-bugs4Us6f06zS6B7Zmy1xEVs1jr6u # PACS_final_synthclip_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1nv0zsWF-zLua2WGITIRvMDz7hVw5tc1v # PACS_final_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow
# tar -xf PACS_final_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# tar -xf PACS_final_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# rm PACS_final_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_synclr_sdxl_floyd_cogview2_sd3_flux_auraflow.tar
# rm PACS_final_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_synthclip_sdxl_floyd_cogview2_sd3_flux_auraflow.tar

# # 16 DINO base lp (0921)
# ./gdrive files download 1rXvb0CzvCJdecOVq0VSm8I12nke1dvLW # PACS_final_DINO_base_Uncertainty_50_0.0001_lp
# ./gdrive files download 1WqVbRPENyqXci37ytAzYEShEC_lqEw98 # PACS_final_DINO_base_Adacore_10_0.0001_lp
# ./gdrive files download 1n3-vAHx2bKI_C7VqOyQE5S1_UvMj7oL8 # PACS_final_DINO_base_Adacore_30_0.0001_lp
# ./gdrive files download 1O8jc5NeTBOhE1hwm8cDMFxUDqqVYIjAI # PACS_final_DINO_base_Adacore_50_0.0001_lp
# ./gdrive files download 1EF48eJPmtXNfEvAxW-QR1KZAfktAXwbM # PACS_final_DINO_base_CurvMatch_10_0.0001_lp
# ./gdrive files download 1BXCqex0VPGCE2fwYUGfmcWG7adn9Nuru # PACS_final_DINO_base_CurvMatch_30_0.0001_lp
# ./gdrive files download 1PShcKqz8dG-YzXKTHFCTzgzKXC4eUf5R # PACS_final_DINO_base_CurvMatch_50_0.0001_lp
# ./gdrive files download 1Kn697vQVZGQmjBj5dJUxQyg9XvlB4O8D # PACS_final_DINO_base_Glister_10_0.0001_lp
# ./gdrive files download 1YVhdMjZnKboYvVtY5J0Bi2WGK1ByJPQg # PACS_final_DINO_base_Glister_30_0.0001_lp
# ./gdrive files download 1O4kT-YvhPmK8LxH_zwzIt6OZj8za9epa # PACS_final_DINO_base_Glister_50_0.0001_lp
# ./gdrive files download 1XbqUgmYa-So8x8JLzCeYJiy5zSe8sshJ # PACS_final_DINO_base_GradMatch_10_0.0001_lp
# ./gdrive files download 19twg4sBkAy6PYdBDMyIwhqeCnRVtgFGX # PACS_final_DINO_base_GradMatch_30_0.0001_lp
# ./gdrive files download 1Ag6MisBPPpcgafnlmn-H81phFqwmVFJ_ # PACS_final_DINO_base_GradMatch_50_0.0001_lp
# ./gdrive files download 1Nk5Nzoy_Z4l7VhcY5rwg9avJmdFUewEL # PACS_final_DINO_base_Submodular_50_0.0001_lp
# ./gdrive files download 1iwLQw8M8_CobirSdz-Z5SCBDfiJXiXhe # PACS_final_DINO_base_Uncertainty_10_0.0001_lp
# ./gdrive files download 1tRPvaFpK8mIVA8EPn8lk_fNTa2YkXfNi # PACS_final_DINO_base_Uncertainty_30_0.0001_lp
# tar -xf PACS_final_DINO_base_Uncertainty_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Adacore_10_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Adacore_30_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Adacore_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_10_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_30_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_CurvMatch_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_10_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_30_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Glister_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_10_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_30_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_GradMatch_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Submodular_50_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Uncertainty_10_0.0001_lp.tar -C PACS_final
# tar -xf PACS_final_DINO_base_Uncertainty_30_0.0001_lp.tar -C PACS_final
# rm PACS_final_DINO_base_Uncertainty_50_0.0001_lp.tar
# rm PACS_final_DINO_base_Adacore_10_0.0001_lp.tar
# rm PACS_final_DINO_base_Adacore_30_0.0001_lp.tar
# rm PACS_final_DINO_base_Adacore_50_0.0001_lp.tar
# rm PACS_final_DINO_base_CurvMatch_10_0.0001_lp.tar
# rm PACS_final_DINO_base_CurvMatch_30_0.0001_lp.tar
# rm PACS_final_DINO_base_CurvMatch_50_0.0001_lp.tar
# rm PACS_final_DINO_base_Glister_10_0.0001_lp.tar
# rm PACS_final_DINO_base_Glister_30_0.0001_lp.tar
# rm PACS_final_DINO_base_Glister_50_0.0001_lp.tar
# rm PACS_final_DINO_base_GradMatch_10_0.0001_lp.tar
# rm PACS_final_DINO_base_GradMatch_30_0.0001_lp.tar
# rm PACS_final_DINO_base_GradMatch_50_0.0001_lp.tar
# rm PACS_final_DINO_base_Submodular_50_0.0001_lp.tar
# rm PACS_final_DINO_base_Uncertainty_10_0.0001_lp.tar
# rm PACS_final_DINO_base_Uncertainty_30_0.0001_lp.tar

# # DINOv2 web
# ./gdrive files download 1Hiy_eBRJwQNXs_D5J6J4bqnao3Nd-K6_ # PACS_final_web_DINOv2_base_Adacore_120_1e-07
# ./gdrive files download 1Bt_ptYK1ac-G_fGJlCPbEy_HWTUpiD4o # PACS_final_web_DINOv2_base_CurvMatch_120_1e-07
# ./gdrive files download 1Jh4M2tOYgi75luIP90VJ8bj5ydA3Lhrw # PACS_final_web_DINOv2_base_Glister_120_1e-07
# ./gdrive files download 1xNbqBMYhduqljA6KpXozFfAx18EEan6q # PACS_final_web_DINOv2_base_GradMatch_120_1e-07
# ./gdrive files download 1KOG6LJaAteBHdcxZcH6KiDnJgFgaf_KR # PACS_final_web_DINOv2_base_Submodular_120_1e-07
# ./gdrive files download 1iQ6rCTw-kQImHerQiNnqq9ByShSyljpD # PACS_final_web_DINOv2_base_Uncertainty_120_1e-07
# tar -xf PACS_final_web_DINOv2_base_Adacore_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_web_DINOv2_base_CurvMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_web_DINOv2_base_Glister_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_web_DINOv2_base_GradMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_web_DINOv2_base_Submodular_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_web_DINOv2_base_Uncertainty_120_1e-07.tar -C PACS_final
# rm PACS_final_web_DINOv2_base_Adacore_120_1e-07.tar
# rm PACS_final_web_DINOv2_base_CurvMatch_120_1e-07.tar
# rm PACS_final_web_DINOv2_base_Glister_120_1e-07.tar
# rm PACS_final_web_DINOv2_base_GradMatch_120_1e-07.tar
# rm PACS_final_web_DINOv2_base_Submodular_120_1e-07.tar
# rm PACS_final_web_DINOv2_base_Uncertainty_120_1e-07.tar

# # wo2 and wo_h
# ./gdrive files download 1xIePcBrhQ8DJSPUiTkfU9STnJyJRUqPo # PACS_final_wo_cot_wo_hierarchy_50_sdxl
# ./gdrive files download 1baRIHNFJh_uvaMdOv5t1DurkXtjITeGo # PACS_final_wo_hierarchy_50_sdxl
# tar -xf PACS_final_wo_cot_wo_hierarchy_50_sdxl.tar -C PACS_final
# tar -xf PACS_final_wo_hierarchy_50_sdxl.tar -C PACS_final
# rm PACS_final_wo_cot_wo_hierarchy_50_sdxl.tar
# rm PACS_final_wo_hierarchy_50_sdxl.tar

# # synclr coreset
# ./gdrive files download 1_lJjhGX2nrUSyo8xY1e4sgwOKL5E-4MF # PACS_final_synclr_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1AYR7LlmyNcLaPs0DGj7biTVTD0ZfgD5T # PACS_final_synclr_DINO_base_Adacore_30_0.0001
# ./gdrive files download 1V_5WOJkjyfAzTrCZLpxRRLlfbepY3fL5 # PACS_final_synclr_DINO_base_Adacore_50_0.0001
# ./gdrive files download 1zbW4Njac2A0A9wkNJliM1K67lhKCXzni # PACS_final_synclr_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1XWlWvXTDEHY9reiF71z8_SX7ILpqqSmY # PACS_final_synclr_DINO_base_CurvMatch_30_0.0001
# ./gdrive files download 1EJVHRC78-_3mDKQXJ-5TM5hIdjqjE0OS # PACS_final_synclr_DINO_base_CurvMatch_50_0.0001
# ./gdrive files download 172h0dG7xtSyE_34WPXvsGwfS9iEJPbro # PACS_final_synclr_DINO_base_Glister_10_0.0001
# ./gdrive files download 1vAdJe9XtsChJA1q_XM00w5K7ZQ8V6fQ0 # PACS_final_synclr_DINO_base_Glister_30_0.0001
# ./gdrive files download 1A8pye8Jl0UNs4lwXSSApGbtoLUz-oL_L # PACS_final_synclr_DINO_base_Glister_50_0.0001
# ./gdrive files download 1IJCWIKnPxY5_6F9KlPy14W1d2h5Ra-Tm # PACS_final_synclr_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 11rBH30nzgoPo8eRgMUer8gQ4vrD5LJFp # PACS_final_synclr_DINO_base_GradMatch_30_0.0001
# ./gdrive files download 1FyRRvQ9Cmf_YPZ-khB1xg4g8qUFOv2HS # PACS_final_synclr_DINO_base_GradMatch_50_0.0001
# ./gdrive files download 1oeM5pjjF_20oAbgHGaJXf3SpenqC0Ww_ # PACS_final_synclr_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1Q5lQ4TLsYJtayRJ9QM2hwtvlSkGZY7tn # PACS_final_synclr_DINO_base_Submodular_30_0.0001
# ./gdrive files download 1yEpjYBN4NlU3vqLdQvo1YtXZv1yMEUNY # PACS_final_synclr_DINO_base_Submodular_50_0.0001
# ./gdrive files download 19E1dSEb2lPl5EcrOyiA-7IJAwpSGr5FR # PACS_final_synclr_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1JDutdMSeWKrb889c-b-rSCxorpCcPKGR # PACS_final_synclr_DINO_base_Uncertainty_30_0.0001
# ./gdrive files download 1aUGJg5URXF_bIrKZJmsuYIS4MGfXd84w # PACS_final_synclr_DINO_base_Uncertainty_50_0.0001
# ./gdrive files download 1-CySbUQTKeXfAzTVxuQ-Y540SGmvudR9 # PACS_final_synclr_DINOv2_base_Adacore_100_1e-07
# ./gdrive files download 1gQmeVG0b17jLSDkO1PmpQIJCETG4krCy # PACS_final_synclr_DINOv2_base_Adacore_120_1e-07
# ./gdrive files download 1TGPs07dWk6XTV30xPgN_K15xIY5mGCw8 # PACS_final_synclr_DINOv2_base_CurvMatch_100_1e-07
# ./gdrive files download 1mbuTJEKqxv8AvT0yN1gS_OSXuPAq5K_q # PACS_final_synclr_DINOv2_base_CurvMatch_120_1e-07
# ./gdrive files download 1UAFnV-ZnmwcTXGyLvmo2nOp-ZOOhow6S # PACS_final_synclr_DINOv2_base_Glister_100_1e-07
# ./gdrive files download 1M5zGPkLtrKFvDEB_-kGf5z6XiHV6BvZv # PACS_final_synclr_DINOv2_base_Glister_120_1e-07
# ./gdrive files download 1FZtxSxXhpTMva4Dp6IrQnmh0Sy0749sK # PACS_final_synclr_DINOv2_base_GradMatch_100_1e-07
# ./gdrive files download 11aObIWvLfi23CC4k_bCuZ1FylvOD-yNw # PACS_final_synclr_DINOv2_base_GradMatch_120_1e-07
# ./gdrive files download 1I8CxNAfwFLM7urZAdiNkia0bVioKZKKV # PACS_final_synclr_DINOv2_base_Submodular_100_1e-07
# ./gdrive files download 1TN7wz3BVZ4s2n2XS3q0t_0tELBS-L98p # PACS_final_synclr_DINOv2_base_Submodular_120_1e-07
# ./gdrive files download 1NddaU9trs-XJ3vpFCU3piJ2kuK0B_qCN # PACS_final_synclr_DINOv2_base_Uncertainty_100_1e-07
# ./gdrive files download 13MNY8d-ytQMosFnEpvJEfZjiitIyVymT # PACS_final_synclr_DINOv2_base_Uncertainty_120_1e-07
# tar -xf PACS_final_synclr_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Adacore_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Adacore_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_CurvMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_CurvMatch_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Glister_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Glister_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_GradMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_GradMatch_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Submodular_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Submodular_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Uncertainty_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINO_base_Uncertainty_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Adacore_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Adacore_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_CurvMatch_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_CurvMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Glister_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Glister_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_GradMatch_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_GradMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Submodular_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Submodular_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Uncertainty_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synclr_DINOv2_base_Uncertainty_120_1e-07.tar -C PACS_final
# rm PACS_final_synclr_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_Adacore_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_Adacore_50_0.0001.tar
# rm PACS_final_synclr_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_CurvMatch_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_CurvMatch_50_0.0001.tar
# rm PACS_final_synclr_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_Glister_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_Glister_50_0.0001.tar
# rm PACS_final_synclr_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_GradMatch_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_GradMatch_50_0.0001.tar
# rm PACS_final_synclr_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_Submodular_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_Submodular_50_0.0001.tar
# rm PACS_final_synclr_DINO_base_Uncertainty_10_0.0001.tar
# rm PACS_final_synclr_DINO_base_Uncertainty_30_0.0001.tar
# rm PACS_final_synclr_DINO_base_Uncertainty_50_0.0001.tar
# rm PACS_final_synclr_DINOv2_base_Adacore_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Adacore_120_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_CurvMatch_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_CurvMatch_120_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Glister_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Glister_120_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_GradMatch_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_GradMatch_120_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Submodular_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Submodular_120_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Uncertainty_100_1e-07.tar
# rm PACS_final_synclr_DINOv2_base_Uncertainty_120_1e-07.tar

# # synthclip coreset
# ./gdrive files download 1tV0bauaj2k1JHgV0ZX4vMM2lYqHbPefu # PACS_final_synthclip_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1Fg3Dy2ioeMQ7-EI1u7o5U_lzMQ8-n4U5 # PACS_final_synthclip_DINO_base_Adacore_30_0.0001
# ./gdrive files download 1wWJfTvzmPPkrrvhKiiFA0sdk6eEXNiTk # PACS_final_synthclip_DINO_base_Adacore_50_0.0001
# ./gdrive files download 1Tx-iypkzkvDI8p5XmdxlFuKwEivHTFFd # PACS_final_synthclip_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1ANIgPxfpeeEE0aL_Y-wGK8BNfjPB670f # PACS_final_synthclip_DINO_base_CurvMatch_30_0.0001
# ./gdrive files download 1fb3l0uJnPe6ZyZk-F9kS7UQYeICzDTWg # PACS_final_synthclip_DINO_base_CurvMatch_50_0.0001
# ./gdrive files download 1O05mcHYWt0hhBwGMrhbEli3JOvMWzW4g # PACS_final_synthclip_DINO_base_Glister_10_0.0001
# ./gdrive files download 1su7BMifstNHg43ob8q6LpFd8V_NG-_V7 # PACS_final_synthclip_DINO_base_Glister_30_0.0001
# ./gdrive files download 1KNorwuq0FW8_jbE2lHcMaTByD-quVwy- # PACS_final_synthclip_DINO_base_Glister_50_0.0001
# ./gdrive files download 135bc3WXZ-DMHaci1kddtAs3l5fUTlZKY # PACS_final_synthclip_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1ksui6BnqgVP3bWxI7EbIq9SDcXUAnkq1 # PACS_final_synthclip_DINO_base_GradMatch_30_0.0001
# ./gdrive files download 1zTHt801001FCS40eW14b9Wg6UC__p_3J # PACS_final_synthclip_DINO_base_GradMatch_50_0.0001
# ./gdrive files download 1_MQWx_6gvbfUpXAt32XSL1miCMb1USYT # PACS_final_synthclip_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1xbFtbM0nyEDnRZme0a4fdSjmd6tfK0Wi # PACS_final_synthclip_DINO_base_Submodular_30_0.0001
# ./gdrive files download 1HaC_BAShI5BN_JpjAtWDnvgOSVTLkT7- # PACS_final_synthclip_DINO_base_Submodular_50_0.0001
# ./gdrive files download 1VCzz3C8u5H9ReMhZz-VLrb9YwJD18LXi # PACS_final_synthclip_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1goMuOitJidTkg6tUNx0HCZ05CBf6LZ6H # PACS_final_synthclip_DINO_base_Uncertainty_30_0.0001
# ./gdrive files download 1hk7kBD2ZeWy001hBUPCbyIGtIRUayXQ- # PACS_final_synthclip_DINO_base_Uncertainty_50_0.0001
# ./gdrive files download 167CcTzYa8JJxjeiXKzjQOlL5i-xs-Qbz # PACS_final_synthclip_DINOv2_base_Adacore_100_1e-07
# ./gdrive files download 13rO4-QvuNxHwY_wkwzgC_zNVhp4y4FV8 # PACS_final_synthclip_DINOv2_base_Adacore_120_1e-07
# ./gdrive files download 1fC0PL936XjHje_1SDb0Tu-wOLNVAITI_ # PACS_final_synthclip_DINOv2_base_CurvMatch_100_1e-07
# ./gdrive files download 1jVrsGM9KBSX6Hs1T2FFq_q30SHFgfwBH # PACS_final_synthclip_DINOv2_base_CurvMatch_120_1e-07
# ./gdrive files download 1P_Ew25Wbx0i8SojeTrCyMik0o31pFuW8 # PACS_final_synthclip_DINOv2_base_Glister_100_1e-07
# ./gdrive files download 1QSgpNxohr1-oda6_qHOcL1a9Wvbcno9P # PACS_final_synthclip_DINOv2_base_Glister_120_1e-07
# ./gdrive files download 1vo2J9i-g4YXTs-BEgFxZc1Uqip7KMsxH # PACS_final_synthclip_DINOv2_base_GradMatch_100_1e-07
# ./gdrive files download 1ynOwmOpwxhsOv6lA4wpHbZDITmGGkdDd # PACS_final_synthclip_DINOv2_base_GradMatch_120_1e-07
# ./gdrive files download 1o2CQCPxK5pe26A5M1r6tu4wNdQn5JGea # PACS_final_synthclip_DINOv2_base_Submodular_100_1e-07
# ./gdrive files download 1_Nmvi4pbEPTy_bon5Okc5MfNpVrBKbT8 # PACS_final_synthclip_DINOv2_base_Submodular_120_1e-07
# ./gdrive files download 1EPx3r3fgwZXdJO1mm1c8Z2zs4-5-61ue # PACS_final_synthclip_DINOv2_base_Uncertainty_100_1e-07
# ./gdrive files download 1fqR0E4jNRmH0kSJ8xo3U7NTPfIaaZkT4 # PACS_final_synthclip_DINOv2_base_Uncertainty_120_1e-07
# tar -xf PACS_final_synthclip_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Adacore_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Adacore_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_CurvMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_CurvMatch_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Glister_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Glister_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_GradMatch_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_GradMatch_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Submodular_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Submodular_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Uncertainty_30_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINO_base_Uncertainty_50_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Adacore_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Adacore_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_CurvMatch_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_CurvMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Glister_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Glister_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_GradMatch_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_GradMatch_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Submodular_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Submodular_120_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Uncertainty_100_1e-07.tar -C PACS_final
# tar -xf PACS_final_synthclip_DINOv2_base_Uncertainty_120_1e-07.tar -C PACS_final
# rm PACS_final_synthclip_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Adacore_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Adacore_50_0.0001.tar
# rm PACS_final_synthclip_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_CurvMatch_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_CurvMatch_50_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Glister_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Glister_50_0.0001.tar
# rm PACS_final_synthclip_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_GradMatch_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_GradMatch_50_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Submodular_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Submodular_50_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Uncertainty_10_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Uncertainty_30_0.0001.tar
# rm PACS_final_synthclip_DINO_base_Uncertainty_50_0.0001.tar
# rm PACS_final_synthclip_DINOv2_base_Adacore_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Adacore_120_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_CurvMatch_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_CurvMatch_120_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Glister_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Glister_120_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_GradMatch_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_GradMatch_120_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Submodular_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Submodular_120_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Uncertainty_100_1e-07.tar
# rm PACS_final_synthclip_DINOv2_base_Uncertainty_120_1e-07.tar

# # PACS_final fake_f + RMD
# ./gdrive files download 1FhzMAyA-Ebr4TITwKfuOZDn60fYjLnVk # PACS_final_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow
# tar -xf PACS_final_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow.tar -C PACS_final
# rm PACS_final_fake_f_sdxl_floyd_cogview2_sd3_flux_auraflow.tar

# # PACS_final coreset wo flux
# ./gdrive files download 1A6n2Q3IlEnaQVNP7Y_KNDFU_j8mzAwxc # PACS_final_synclr_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1Hev_2GTfMoiOncCLua17JG7dNfqPOFKp # PACS_final_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 14jFxv73fJrlgZ2zOoJhfESK_Yucwqcec # PACS_final_synclr_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1gjmRiR7Wl-Xg2m4pa_XtHkxocso5mfMY # PACS_final_synclr_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1qDRNZudP3MAv420IZvAOxaP2GUK6JGTe # PACS_final_synclr_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1pyyJbI0wnJjbobsFiDYv0eY-s-1f_uK0 # PACS_final_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1k65TS812WBu-YbYMQ3PRSzytCQHQskiQ # PACS_final_synthclip_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1NvgA8OCNfGBZataHFlZjLizkCfmni8lI # PACS_final_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1-IuZky_8iaLs0CkYQaSfSuc5Ipb8rPT2 # PACS_final_synthclip_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1dmE_ba_HqTXxvwLA3JpYOITsT_Swaob5 # PACS_final_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1kZo4rlooPgHeGwsacPoX9-Jk5UOgkSqm # PACS_final_synthclip_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1rbzyRkkaC1H0KzafyovLSF3md9_FxGI9 # PACS_final_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf PACS_final_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# rm PACS_final_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # HCFR coreset
# ./gdrive files download 1JNdC369T4fIuezPtk9i8MqYYLn8uLrei # PACS_final_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1WkcIUXSnHrnO_OKqwlWQAOiVbzNY3spA # PACS_final_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1kxdnX0pWOJJMMGAK8oM0wQZKtfyhgC5c # PACS_final_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1cRMRnNJ3ZjnovvVLz09M-5_xSK-6M0Ns # PACS_final_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1fcKN-4ejZCEGWk-rPT93aJ3tMePpMiI3 # PACS_final_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1NL309wqGB-mYtX3a-eMpzD4yMs2sVkdq # PACS_final_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# rm PACS_final_cot_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_cot_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_cot_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_cot_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_cot_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_cot_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # hcfr, synclr, synthclip moderate
# ./gdrive files download 1Kkk3420TuVVYDqeJ-CdiC957AO_mt05E # PACS_final_hcfr_wo_flux_CLIP_moderate
# ./gdrive files download 1giEKJzkp1Ry9JkWvDHjI5CKez7XGHbKG # PACS_final_synthclip_wo_flux_CLIP_moderate
# ./gdrive files download 1nxSbdwDSewc2-W2B3P9B1ftRQVEQ1O9Y # PACS_final_synclr_wo_flux_CLIP_moderate
# tar -xf PACS_final_hcfr_wo_flux_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_synthclip_wo_flux_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_synclr_wo_flux_CLIP_moderate.tar -C PACS_final
# rm PACS_final_hcfr_wo_flux_CLIP_moderate.tar
# rm PACS_final_synthclip_wo_flux_CLIP_moderate.tar
# rm PACS_final_synclr_wo_flux_CLIP_moderate.tar

# # HCFR equalweight
# ./gdrive files download 1D4IYmnW5_lU9wlJv-X3GPJSRehHhBZUj # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C PACS_final
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar

# # LE, sdbp with or wo equalweight
# ./gdrive files download 1N3wpajXxW2HV5sxkRvgKY-46xYLXIZII # PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1Lspq6-bXcV0VcJQgYydr-zyHv_Z2EpO7 # PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# ./gdrive files download 1lMte9uMfAgL0QDShG1F7MBp-agNbW_vD # PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1Q7UpWPRiTkEUJRmmEMShIdXgZ5maqjEb # PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C PACS_final
# tar -xf PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# tar -xf PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C PACS_final
# rm PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_LE_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar
# rm PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm PACS_final_sdbp_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar

# # PACS_final LE, fake_f, sdbp wo_flux coresets
# ./gdrive files download 1XqPB5jj-sZe9oX-0qd48-HpcE66l9__7 # PACS_final_sdbp_CLIP_moderate
# ./gdrive files download 1rTzjGnUTDjmbmwiUR8bmBhOEADxBimYM # PACS_final_sdbp_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1oUQC62hij4r5JDXMmLA2vgvwDofF5Ecc # PACS_final_sdbp_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1e0OuALhoIoOQUBrMbMNh0i719-ToomlD # PACS_final_sdbp_DINO_base_Glister_10_0.0001
# ./gdrive files download 1ajCssP-szUeC5TnLjyIJZF7PaFJm4uJ3 # PACS_final_sdbp_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1AQwVUvZR0bGJOaW1zK62_etTGd63QF2j # PACS_final_sdbp_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1buk_UGso4AWANjv2simbcuEcnlK_fDhT # PACS_final_sdbp_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 1id8oHEgkhBsolBfSZonAPqA_3_FCujqY # PACS_final_LE_CLIP_moderate
# ./gdrive files download 1uPzbkY9vnOBh-5RcGEbf65Qz8U364PrA # PACS_final_LE_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1ziXaSaq39DNBmIiD7_GhuyR7YjtLrqxQ # PACS_final_LE_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1lldY1UdHV9ORi4Qwuaqtj64BSQ-c2FzC # PACS_final_LE_DINO_base_Glister_10_0.0001
# ./gdrive files download 1PPr6oLvvjp_GTXWUwB9W9Yf5JuKOdON1 # PACS_final_LE_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1tMhqHtXlqeYHamUMfBUMmFW-8xP6QfJt # PACS_final_LE_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1meb5e08IL2o0Ocl0ed5HqUPLwQcxcMPS # PACS_final_LE_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 15xDzx6IKTkbBZVdohhtb57jI33AHklbf # PACS_final_fake_f_CLIP_moderate
# ./gdrive files download 17edlFWUkXm1LezQHLRCb-O1iSHxru7qY # PACS_final_fake_f_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1lws0ETzN-zuvieiHSQ8OB8II7tyAyBFE # PACS_final_fake_f_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1HfFUaKx-hTS_qEsYnbQVInCR-q-dgS3_ # PACS_final_fake_f_DINO_base_Glister_10_0.0001
# ./gdrive files download 12nEd4mJpFXALHvQ_VzbhhrRi4bADl5NR # PACS_final_fake_f_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 170fQMMdspFPrxEgnbjl4dQr-7x7kLZPs # PACS_final_fake_f_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1LNi3qyaGEcpmIRHXwLsgExENj_8n_WYf # PACS_final_fake_f_DINO_base_Uncertainty_10_0.0001
# tar -xf PACS_final_sdbp_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_sdbp_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_LE_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_fake_f_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# rm PACS_final_sdbp_CLIP_moderate.tar
# rm PACS_final_sdbp_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_sdbp_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_sdbp_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_sdbp_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_sdbp_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_sdbp_DINO_base_Uncertainty_10_0.0001.tar
# rm PACS_final_LE_CLIP_moderate.tar
# rm PACS_final_LE_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_LE_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_LE_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_LE_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_LE_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_LE_DINO_base_Uncertainty_10_0.0001.tar
# rm PACS_final_fake_f_CLIP_moderate.tar
# rm PACS_final_fake_f_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_fake_f_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_fake_f_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_fake_f_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_fake_f_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_fake_f_DINO_base_Uncertainty_10_0.0001.tar

# # fake_f_RMD_wo_flux
# ./gdrive files download 1dhUw6a-u5rhULgln41T2uRysSMPpCzsG # PACS_final_fake_f_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf PACS_final_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar -C PACS_final
# rm PACS_final_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar

# # PACS wflux new coresets
# ./gdrive files download 1tBNGIH5hDP1IWHU2ZSDXwP6y06oR6L-w # PACS_final_cot_50_2_wflux_CLIP_moderate
# ./gdrive files download 1sNRucWhUp4RGfz65v6w38BQfdzdWZDi0 # PACS_final_cot_50_2_wflux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 18OfhWjxffsLc_6YIzMSrq7KaYlEWKAL6 # PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1y-Z3cKmvVFdtSC5YVCDxRYsLDn_fJjxt # PACS_final_cot_50_2_wflux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1L8qpPyrADRZDsg5uZ7iucf6Bw2RI9MhT # PACS_final_cot_50_2_wflux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 129_524YzV2x2ywf9L0qzHo5WDieh2Txz # PACS_final_cot_50_2_wflux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1ZG2yCwf3p3kLjqUaTfy6lAIe4qokKrt- # PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_10_0.0001
# tar -xf PACS_final_cot_50_2_wflux_CLIP_moderate.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Adacore_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Glister_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_GradMatch_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Submodular_10_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_10_0.0001.tar -C PACS_final
# rm PACS_final_cot_50_2_wflux_CLIP_moderate.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Adacore_10_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_10_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Glister_10_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_GradMatch_10_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Submodular_10_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_10_0.0001.tar


# # dynamic 50 prompts new
# ./gdrive files download 1aKj7gdh5Bxvc0cKBZ6OHkyGZC2UPWHJe # PACS_final_dynamic_50_new
# tar -xf PACS_final_dynamic_50_new.tar -C PACS_final
# rm PACS_final_dynamic_50_new.tar

# # truncate 95
# ./gdrive files download 1Y9cmge5-xSq_9qmHzagEY1PRhPOMmB7c # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_95
# ./gdrive files download 1mIAmickrA9KDFGGI7bSKmXx7FtZgHfV_ # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow_95
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_95.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow_95.tar -C PACS_final
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_95.tar
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_flux_auraflow_95.tar

# # wflux, woflux 20epoch
# ./gdrive files download 1OZtaA3PC_6IUcGK25DhnpuRdfAzH7SaV # PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_20_0.0001
# ./gdrive files download 18uh7JlsZHRjuLKqia542uVlixkMcM6vD # PACS_final_cot_50_2_wflux_DINO_base_Submodular_20_0.0001
# ./gdrive files download 110Fo_noCs2f425lmwmAaC8tlkLbEbeLI # PACS_final_cot_50_2_wflux_DINO_base_GradMatch_20_0.0001
# ./gdrive files download 1ZeOEY1TaAwksLfMGRcAhqatl5Z3tnHvd # PACS_final_cot_50_2_wflux_DINO_base_Glister_20_0.0001
# ./gdrive files download 1ujBKpdczpdNWVeqDUja-o9T3-K9b1KrT # PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_20_0.0001
# ./gdrive files download 1Ap5DkrY_x_9YY0jBuhWhyVd8VjfmcZD8 # PACS_final_cot_50_2_wflux_DINO_base_Adacore_20_0.0001
# ./gdrive files download 11LQ5kJ-fgSc2EJeNj2cUdW68KKvkyK_l # PACS_final_cot_50_2_woflux_DINO_base_Uncertainty_20_0.0001
# ./gdrive files download 190T74dj-cPDzo_hvXa1eyKFLnl6ss13y # PACS_final_cot_50_2_woflux_DINO_base_Submodular_20_0.0001
# ./gdrive files download 1Oc5jjPvaqt_OEkvpPc10zdyW6y8_m4lS # PACS_final_cot_50_2_woflux_DINO_base_GradMatch_20_0.0001
# ./gdrive files download 1LjqOxucVOiCvJjDkYL7unxC_9kFpgs7T # PACS_final_cot_50_2_woflux_DINO_base_Glister_20_0.0001
# ./gdrive files download 1ffFqEdDOoPAiY9QFzAPASLs78kmrfg-e # PACS_final_cot_50_2_woflux_DINO_base_CurvMatch_20_0.0001
# ./gdrive files download 1B4gJsj32lAYLK1ahkFsZ5c8EalJ34KK0 # PACS_final_cot_50_2_woflux_DINO_base_Adacore_20_0.0001
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Submodular_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_GradMatch_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Glister_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_wflux_DINO_base_Adacore_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_Uncertainty_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_Submodular_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_GradMatch_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_Glister_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_CurvMatch_20_0.0001.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_woflux_DINO_base_Adacore_20_0.0001.tar -C PACS_final
# rm PACS_final_cot_50_2_wflux_DINO_base_Uncertainty_20_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Submodular_20_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_GradMatch_20_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Glister_20_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_CurvMatch_20_0.0001.tar
# rm PACS_final_cot_50_2_wflux_DINO_base_Adacore_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_Uncertainty_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_Submodular_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_GradMatch_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_Glister_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_CurvMatch_20_0.0001.tar
# rm PACS_final_cot_50_2_woflux_DINO_base_Adacore_20_0.0001.tar

# # Web new
# ./gdrive files download 1O65R1A5Mj7GetXDpx3XudPUHP6og6ysn # PACS_final_bing_large_wofilter
# ./gdrive files download 1OMmbCKiTOn6dlcxaDsX2kN5SD9npvI00 # PACS_final_bing_wofilter
# ./gdrive files download 1_v22fW7m-fGKG_5c--S_ulwGQG7evIF0 # PACS_final_flickr_large_wofilter
# ./gdrive files download 1xz49D-B90dVfDu_CPI0OpPIX-I5wZ-Ac # PACS_final_flickr_wofilter
# ./gdrive files download 1_4B2Ww38vnzPUmqbRPUlCVrIfQ9OaTIx # PACS_final_google_large_wofilter
# ./gdrive files download 1HgdrD8nISKkQketyHRs9MhxwKYuL3btt # PACS_final_google_wofilter
# ./gdrive files download 18XBjEHSKZmgv_uOl4sS0-yc481wJN4vA # PACS_final_bing_large_wfilter
# ./gdrive files download 1qoLRcGTK8tiibhHjlmd9gTrBRRJfetAk # PACS_final_bing_wfilter
# ./gdrive files download 15NyPX7bbIglQwWNqkHKrLNGY08N7QjLc # PACS_final_flickr_large_wfilter
# ./gdrive files download 1UDQ68hGjmyTAFInrYovPHaDTa5E-gH9a # PACS_final_flickr_wfilter
# ./gdrive files download 1FbVB5GefoxvZWqsyG5h55cZ13jIFAckI # PACS_final_google_large_wfilter
# ./gdrive files download 18T5JdBrqgxDP9HObO85tdKCddPNx9BtL # PACS_final_google_wfilter
# tar -xf PACS_final_bing_large_wofilter.tar -C PACS_final
# tar -xf PACS_final_bing_wofilter.tar -C PACS_final
# tar -xf PACS_final_flickr_large_wofilter.tar -C PACS_final
# tar -xf PACS_final_flickr_wofilter.tar -C PACS_final
# tar -xf PACS_final_google_large_wofilter.tar -C PACS_final
# tar -xf PACS_final_google_wofilter.tar -C PACS_final
# tar -xf PACS_final_bing_large_wfilter.tar -C PACS_final
# tar -xf PACS_final_bing_wfilter.tar -C PACS_final
# tar -xf PACS_final_flickr_large_wfilter.tar -C PACS_final
# tar -xf PACS_final_flickr_wfilter.tar -C PACS_final
# tar -xf PACS_final_google_large_wfilter.tar -C PACS_final
# tar -xf PACS_final_google_wfilter.tar -C PACS_final
# rm PACS_final_bing_large_wofilter.tar
# rm PACS_final_bing_wofilter.tar
# rm PACS_final_flickr_large_wofilter.tar
# rm PACS_final_flickr_wofilter.tar
# rm PACS_final_google_large_wofilter.tar
# rm PACS_final_google_wofilter.tar
# rm PACS_final_bing_large_wfilter.tar
# rm PACS_final_bing_wfilter.tar
# rm PACS_final_flickr_large_wfilter.tar
# rm PACS_final_flickr_wfilter.tar
# rm PACS_final_google_large_wfilter.tar
# rm PACS_final_google_wfilter.tar

# ./gdrive files download 1HXnpsFGHGkEWRgkZgZzd3d33w5V0c1rj # PACS_final_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight
# tar -xf PACS_final_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar -C PACS_final
# rm PACS_final_wo_cot_wo_hierarchy_50_sdxl_floyd_cogview2_sd3_auraflow_equalweight.tar
# ./gdrive files download 1pIIxaX35vgKThIyuE5odmduscNxon_5v # PACS_final_sdbp_sdxl_50
# tar -xf PACS_final_sdbp_sdxl_50.tar -C PACS_final
# rm PACS_final_sdbp_sdxl_50.tar

# # PACS HIWING + Glide
# ./gdrive files download 1yr9BImD5K16vEedF52A0EeoddY6LO7Q5 # PACS_final_50_2_glide
# tar -xf PACS_final_50_2_glide.tar -C PACS_final
# rm PACS_final_50_2_glide.tar

# # Ours topk, bottomk, inverse
# ./gdrive files download 1Cm7Zi7S2dSutS_PQdZ_moKnSGdMEIqB- # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_topk
# ./gdrive files download 1U2B-dPSLxyNzWCOuRf8MtgBCBpYEcoMG # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_bottomk
# ./gdrive files download 1d4fBE-n21QOoE7Q5sQbfGGYVGu_rJjln # PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_inverse
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_topk.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_bottomk.tar -C PACS_final
# tar -xf PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_inverse.tar -C PACS_final
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_topk.tar
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_bottomk.tar
# rm PACS_final_cot_50_2_sdxl_floyd_cogview2_sd3_auraflow_inverse.tar

# Hierarchy experiments (for rebuttal)
# ./gdrive files download 1aKOamivwrZZsCkLcDlHHhaiftNDdiWuK # PACS_final_tree_2_sdxl
# ./gdrive files download 1w9Lm3d9civYS_Mk0VA4j1_MzeOZhqbw7 # PACS_final_tree_4_sdxl
# tar -xf PACS_final_tree_2_sdxl.tar -C PACS_final
# tar -xf PACS_final_tree_4_sdxl.tar -C PACS_final
# rm PACS_final_tree_2_sdxl.tar
# rm PACS_final_tree_4_sdxl.tar

# # Real-fake baseline
# ./gdrive files download 17OKYRZSYZgVq6c7sXFxtDxRe31FtMyK_ # PACS_final_train_ma_real_fake
# tar -xf PACS_final_train_ma_real_fake.tar -C PACS_final
# rm PACS_final_train_ma_real_fake.tar

# # Real-fake (CL setup)
# ./gdrive files download 1wRlRcAK7aTJO9ejAPNEUiQuDz4Z605Zw # PACS_final_train_ma_real_fake_cl
# tar -xf PACS_final_train_ma_real_fake_cl.tar -C PACS_final
# rm PACS_final_train_ma_real_fake_cl.tar

# # Real fake more parameters
# ./gdrive files download 1anZFYjfZ39A4dKRPUYKQGlqHsCAp9n3G # PACS_final_train_ma_10_real_fake
# ./gdrive files download 1DcAHYKq758sfD01hFpS7I_RVukYEvZvM # PACS_final_train_ma_50_real_fake
# ./gdrive files download 1lsgtL_S-eOA_xs-AIYm04pw_u0LLXic1 # PACS_final_train_ma_200_real_fake
# ./gdrive files download 1fCbv8OBx7dSnAwxTW9I7kKzw7G5x2b_v # PACS_final_train_ma_400_real_fake
# tar -xf PACS_final_train_ma_10_real_fake.tar -C PACS_final
# tar -xf PACS_final_train_ma_50_real_fake.tar -C PACS_final
# tar -xf PACS_final_train_ma_200_real_fake.tar -C PACS_final
# tar -xf PACS_final_train_ma_400_real_fake.tar -C PACS_final
# rm PACS_final_train_ma_10_real_fake.tar
# rm PACS_final_train_ma_50_real_fake.tar
# rm PACS_final_train_ma_200_real_fake.tar
# rm PACS_final_train_ma_400_real_fake.tar

# # DB finetuned
# ./gdrive files download 1T_XQG0xTY_2xZ0CjvhAtBEsVEJ_KOP1G # PACS_final_db_3
# ./gdrive files download 1Brl6_Qy_iIohe-1w1GbfEie-tqQqNoHq # PACS_final_db_5
# ./gdrive files download 1KYzalT65VponuPX_Pp1bTZr2zAQt0BYz # PACS_final_db_10
# tar -xf PACS_final_db_3.tar -C PACS_final
# tar -xf PACS_final_db_5.tar -C PACS_final
# tar -xf PACS_final_db_10.tar -C PACS_final
# rm PACS_final_db_3.tar
# rm PACS_final_db_5.tar
# rm PACS_final_db_10.tar

# PACS_final tree more
./gdrive files download 1cZidV_EHLdoaVG-lOnQPkgNuwdpceRM4 # PACS_final_tree_d2_w5
./gdrive files download 1mJGVfQZe5EUR9ea9xhlxP-ubWeIriN97 # PACS_final_tree_d3_w7
./gdrive files download 14kxNWG3NY_Ktepytqmab2n6NiLHMn-U- # PACS_final_tree_d1_w7
./gdrive files download 1n_3bRIILy4hBJttDwH5eaVbbRtFJ32bo # PACS_final_tree_d2_w10
tar -xf PACS_final_tree_d2_w5.tar -C PACS_final
tar -xf PACS_final_tree_d3_w7.tar -C PACS_final
tar -xf PACS_final_tree_d1_w7.tar -C PACS_final
tar -xf PACS_final_tree_d2_w10.tar -C PACS_final
rm PACS_final_tree_d2_w5.tar
rm PACS_final_tree_d3_w7.tar
rm PACS_final_tree_d1_w7.tar
rm PACS_final_tree_d2_w10.tar