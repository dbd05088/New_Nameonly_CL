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
