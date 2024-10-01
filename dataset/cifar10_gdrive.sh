./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# # New (0521)
# mkdir -p cifar10
# ./gdrive files download 1VxdSntW9y6tzzSD1aiG_YGaL3jzpJv4g #nc
# ./gdrive files download 16DTKBJjjzU4Z3jmiIYwgEPrXb4Hju3B4 #c
# ./gdrive files download 1jgqUvtGf7Ta1Uq-jMUm6vLmDgjaXzqzu #ma (updated - 0522)
# ./gdrive files download 1TdIXr9uwdCcjDPBmnD_VQ2qv30uNKyxQ #original test set
# tar -xf cifar10_ma.tar -C cifar10
# tar -xf cifar10_nc.tar -C cifar10
# tar -xf cifar10_c.tar -C cifar10
# tar -xf cifar10_original_test.tar -C cifar10
# rm cifar10_ma.tar
# rm cifar10_nc.tar
# rm cifar10_c.tar
# rm cifar10_original_test.tar


# # Base prompt
# gdrive files download 1YKSZtuh4qxl0QUAczZB9Q-EGyxVqj7ek
# tar -xf cifar10_sdxl_base.tar -C cifar10
# rm cifar10_sdxl_base.tar



# # SDBP
# gdrive files download 1QpJ08Ci9f5PSTx96eF1P9geu1MR0hGsx
# tar -xf cifar10_sdbp.tar -C cifar10
# rm cifar10_sdbp.tar



# # sdxl diversified and generated equalweight (0528)
# ./gdrive files download 1VpElBUOtlUK3KNkbrgkdS-VGFqx7k9ww # sdxl diversified
# ./gdrive files download 1GcVfh1r0WVkXZHtcGcULIzw7sll7YOnt # equalweight
# ./gdrive files download 1dmWsy_9vMdJtBu8Kh9pCTeXQTNu-xGtt
# tar -xf cifar10_static_cot_50_sdxl.tar -C cifar10
# tar -xf cifar10_generated_equalweight.tar -C cifar10
# tar -xf CIFAR10_glide.tar -C cifar10 && mv cifar10/CIFAR10_glide cifar10/cifar10_glide # CIFAR10 -> cifar10 (change case)
# rm cifar10_static_cot_50_sdxl.tar
# rm cifar10_generated_equalweight.tar
# rm CIFAR10_glide.tar


# # Compare between 5000 and 10000 (using sdxl_diversified)
# ./gdrive files download 1FXM3vJ0I9ntijCLtD4IxgmPz-M1T5Iop
# ./gdrive files download 1teeqK_nsl1MtSfI9kd9GfbpN-Z9uOFdX
# tar -xf cifar10_sdxl_5000.tar -C cifar10
# tar -xf cifar10_sdxl_10000.tar -C cifar10
# rm cifar10_sdxl_5000.tar
# rm cifar10_sdxl_10000.tar

# # train ma 10000 (0930)
# ./gdrive files download 1Iba892lPEfr-r-lC66IHFkJLjS_G7abT
# tar -xf cifar10_train_ma_10000.tar -C cifar10
# rm cifar10_train_ma_10000.tar

# # cifar10_50_2_sdxl
# ./gdrive files download 1zUMOdboi3XUYUt8kZu8suuDOH5hGyF_C # cifar10_50_2_sdxl
# tar -xf cifar10_50_2_sdxl.tar -C cifar10
# rm cifar10_50_2_sdxl.tar

# # cifar10 50_8, synthclip, fake-f, sdbp sdxl
# ./gdrive files download 1SZ_HfS5ewgUNE0AfvdzwtdtvchBKjmrO # cifar10_50_8_sdxl
# ./gdrive files download 18N8LQVkkji6vZ4xlg4BY6jzo9CVjiZjf # cifar10_sdbp_sdxl
# ./gdrive files download 1NiZlLOxEmwyah9sgszP2TzWLlcQe87JI # cifar10_fake_f_sdxl
# ./gdrive files download 1DhJramdE5H2sYCtFbU24d1Q1scYf5B9X # cifar10_synthclip_sdxl
# tar -xf cifar10_50_8_sdxl.tar -C cifar10
# tar -xf cifar10_sdbp_sdxl.tar -C cifar10
# tar -xf cifar10_fake_f_sdxl.tar -C cifar10
# tar -xf cifar10_synthclip_sdxl.tar -C cifar10
# rm cifar10_50_8_sdxl.tar
# rm cifar10_sdbp_sdxl.tar
# rm cifar10_fake_f_sdxl.tar
# rm cifar10_synthclip_sdxl.tar

# ./gdrive files download 1dwsmH-dMPaZ9DvxhEq9aeLMjBIHAedKu # cifar10_LE_sdxl
# tar -xf cifar10_LE_sdxl.tar -C cifar10
# rm cifar10_LE_sdxl.tar

# ./gdrive files download 1MMwOWMHtdLxncTeC2ZPV7Ua4Qq75XHNo # cifar10_synclr_sdxl
# tar -xf cifar10_synclr_sdxl.tar -C cifar10
# rm cifar10_synclr_sdxl.tar

# ./gdrive files download 1C1OJgJ9pE-iyR9CNlr2vSpTXPSBQTrvK # cifar10_50_3_sdxl
# tar -xf cifar10_50_3_sdxl.tar -C cifar10
# rm cifar10_50_3_sdxl.tar

# ./gdrive files download 1rA8xAJM2xKXcEFskMJmeq8l00KAC5BdM # cifar10_50_4_sdxl
# tar -xf cifar10_50_4_sdxl.tar -C cifar10
# rm cifar10_50_4_sdxl.tar

# ./gdrive files download 12grRkc-7w3C_abmpEQHNYj4S9Cct_yE2 # cifar10_50_1_sdxl
# ./gdrive files download 1Rq8BNQcrceo5jLOZV03SRf9K3dvLhCoC # cifar10_50_7_sdxl
# tar -xf cifar10_50_1_sdxl.tar -C cifar10
# tar -xf cifar10_50_7_sdxl.tar -C cifar10
# rm cifar10_50_1_sdxl.tar
# rm cifar10_50_7_sdxl.tar

# ./gdrive files download 1cIS8y_gdNShk0_2_paLsSu1SBad0dB5B # cifar10_50_2_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf cifar10_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# rm cifar10_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar

# # RMD, prompt ablation 
# ./gdrive files download 1MWF5saCsMnEakqGmWN-QfajJnLJjwWIx # cifar10_fake_f_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1JQc7nhcX4lTtR1yKlfSbLejhdLS8mg9f # cifar10_LE_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 11Ao-Ntvmdb2G9tIBY_H9jQQnOad2i9Uf # cifar10_synclr_sdxl_floyd_cogview2_sd3_auraflow
# ./gdrive files download 1w_k8jsZU1NYkYSR2KC4MAtLcLwRsXId9 # cifar10_synthclip_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf cifar10_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# tar -xf cifar10_LE_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# tar -xf cifar10_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# tar -xf cifar10_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# rm cifar10_fake_f_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm cifar10_LE_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm cifar10_synclr_sdxl_floyd_cogview2_sd3_auraflow.tar
# rm cifar10_synthclip_sdxl_floyd_cogview2_sd3_auraflow.tar

# # Cifar10 Base sdxl
# ./gdrive files download 1QTQMmzBAGhMc0pILa_PYUuzjrXPtooa6 # cifar10_base_sdxl
# tar -xf cifar10_base_sdxl.tar -C cifar10
# rm cifar10_base_sdxl.tar

# # cifar10 sdbp + CONAN
# ./gdrive files download 1EfDHNYFg2TURppGzAiO_HcTRr9iwnoVd # cifar10_sdbp_sdxl_floyd_cogview2_sd3_auraflow
# tar -xf cifar10_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar -C cifar10
# rm cifar10_sdbp_sdxl_floyd_cogview2_sd3_auraflow.tar

# # synclr coresets
# ./gdrive files download 1ad2fDkkK2AmyDOaiI9JASjjJP5r1-APE # cifar10_synclr_wo_flux_CLIP_moderate
# ./gdrive files download 1-NqhcaObYwTlL1vfjw-MOD1xxzb0kXsc # cifar10_synclr_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1Ckq0-aaUe1c95zLShHHxZ6lXegNhHIB8 # cifar10_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1K_C_OQeK8SJMvnvrM6LqznkR-XuRgme8 # cifar10_synclr_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1fi3TaSkLN1cdRwajXnTe44E1fbhlai4l # cifar10_synclr_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1kcCpcMtvwTTcA4oz3aKo6heZ82OIwOy0 # cifar10_synclr_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1QaqSSzfw0Vxt91VOTSlsrF70KOjPxFjz # cifar10_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf cifar10_synclr_wo_flux_CLIP_moderate.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar -C cifar10
# tar -xf cifar10_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C cifar10
# rm cifar10_synclr_wo_flux_CLIP_moderate.tar
# rm cifar10_synclr_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm cifar10_synclr_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm cifar10_synclr_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm cifar10_synclr_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm cifar10_synclr_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm cifar10_synclr_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# # HIWING coresets, synthclip coresets
# ./gdrive files download 1okrnh3Qs7w46AvFma79MWOO1OrZ1HSMm # cifar10_50_2_wo_flux_CLIP_moderate
# ./gdrive files download 1GWHFfEJ7UeXLrZtz4aDMmSS2QTtKU5Go # cifar10_50_2_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1S-ruCE_LiHTPpqHMaUrUCg-biU5S3Kbi # cifar10_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1MuyFmwD5VBWKUxqKxlF3u8sy1cF8B8tE # cifar10_50_2_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 1y3yFmEJk8HCBuhznGSykqFF5MRN_TYKe # cifar10_50_2_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1d3HZ3c6S62HETlBdNUoOWAEcC0qbNjS0 # cifar10_50_2_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1bfmhmRBJXdsNPJqFgtp2UDZt1kDyoIid # cifar10_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001
# ./gdrive files download 15B4Gj0YTVyBZ1yyv03L51LTz294wxY9_ # cifar10_synthclip_wo_flux_CLIP_moderate
# ./gdrive files download 1Kd7M3zQ4BiHEQn2lny5rUo82GFWvoPyu # cifar10_synthclip_wo_flux_DINO_base_Adacore_10_0.0001
# ./gdrive files download 1BcHMxAdS5Obbf11DK73Mi56AsRIKzSP1 # cifar10_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001
# ./gdrive files download 1Zox1iM_wytAiC52TkE6hVtMbOV1QubCU # cifar10_synthclip_wo_flux_DINO_base_Glister_10_0.0001
# ./gdrive files download 15OKpaYFJZhWBLg8wWE7-ZntLfn1GSM3O # cifar10_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001
# ./gdrive files download 1IwctENHswoExxGZqiK0KimnlXownNpEZ # cifar10_synthclip_wo_flux_DINO_base_Submodular_10_0.0001
# ./gdrive files download 1wGMS_jnvVJ-kwaEKaeero9w8fOnWlZ1B # cifar10_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001
# tar -xf cifar10_50_2_wo_flux_CLIP_moderate.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar -C cifar10
# tar -xf cifar10_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_CLIP_moderate.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar -C cifar10
# tar -xf cifar10_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar -C cifar10
# rm cifar10_50_2_wo_flux_CLIP_moderate.tar
# rm cifar10_50_2_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm cifar10_50_2_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm cifar10_50_2_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm cifar10_50_2_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm cifar10_50_2_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm cifar10_50_2_wo_flux_DINO_base_Uncertainty_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_CLIP_moderate.tar
# rm cifar10_synthclip_wo_flux_DINO_base_Adacore_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_DINO_base_CurvMatch_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_DINO_base_Glister_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_DINO_base_GradMatch_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_DINO_base_Submodular_10_0.0001.tar
# rm cifar10_synthclip_wo_flux_DINO_base_Uncertainty_10_0.0001.tar

# Glide-Syn
./gdrive files download 1ktNtmOrAh6ltXwtYHQrxp59y31uNAuCH # cifar10_glide_syn
tar -xf cifar10_glide_syn.tar -C cifar10
rm cifar10_glide_syn.tar