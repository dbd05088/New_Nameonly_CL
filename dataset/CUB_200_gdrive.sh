./gdrive account remove dbd05088@naver.com
./gdrive account import gdrive_export-dbd05088_naver_com.tar
./gdrive account switch dbd05088@naver.com


# Required
mkdir -p CUB_200
./gdrive files download 1VYNnUu5F8InA-Vx2EzU6JBxJzpzDdoqr # test_ma
./gdrive files download 1nrXm3X5lpQw7skouKVwOhtJWEFz6zbWJ # painting

tar -xf CUB_200_test_ma.tar -C CUB_200
tar -xf CUB_200_painting.tar -C CUB_200
rm CUB_200_test_ma.tar
rm CUB_200_painting.tar

# train_ma
./gdrive files download 1dVqhry8jsg7SJh0nEt-uSkOgahyUKGeN
tar -xf CUB_200_train_ma.tar -C CUB_200
rm CUB_200_train_ma.tar

# sdxl, RMD
./gdrive files download 10oIDXVvCwUFERltPqAslBQMuVU4yZHku # CUB_200_50_2_sdxl
./gdrive files download 1tj8fSYdDeUGITqjysPoY8WJFkm43nVxc # CUB_200_50_2_sdxl_floyd_cogview2_sd3_auraflow
tar -xf CUB_200_50_2_sdxl.tar -C CUB_200
tar -xf CUB_200_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar -C CUB_200
rm CUB_200_50_2_sdxl.tar
rm CUB_200_50_2_sdxl_floyd_cogview2_sd3_auraflow.tar

# DB finetuned
./gdrive files download 1Ov_djKBIJqJSEJwBAWKbkZLqR1VYpL13 # CUB_200_db
tar -xf CUB_200_db.tar -C CUB_200
rm CUB_200_db.tar

# DB with 3 examples
./gdrive files download 1sO_qrIvDsTg9ZY6ajSCOzNtKFNIfZWr- # CUB_200_db_3
tar -xf CUB_200_db_3.tar -C CUB_200
rm CUB_200_db_3.tar