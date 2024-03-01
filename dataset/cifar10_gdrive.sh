#nohup gdrive files download 179g6v9hOAwDKXrgQ7Gs3-5hifjVt0UAu #web_10
gdrive files download 1gTJP9ciBwg-bpqdtvlbSDWEjNBQ4x4bu #web
gdrive files download 1W6YRfL-u9ySIdXWkHiQ0Whq9m8EtzQhb #generated
gdrive files download 1VxdSntW9y6tzzSD1aiG_YGaL3jzpJv4g #nc
gdrive files download 16DTKBJjjzU4Z3jmiIYwgEPrXb4Hju3B4 #c
gdrive files download 1JsokoJg1ws0Pbz2HMiosBSpPh4gq7KWL #ma
gdrive files download 1TdIXr9uwdCcjDPBmnD_VQ2qv30uNKyxQ #original test set
gdrive files download 1WIbwg_wV7DIQvGbsctqZy6uBA4yeDcIN #sdxl_diversified

tar -xvf cifar10_MA.tar
tar -xvf cifar10_generated.tar
tar -xvf cifar10_web.tar
tar -xvf cifar10_web_10.tar
tar -xvf cifar10_original_test.tar
tar -xvf cifar10_nc.tar
tar -xvf cifar10_c.tar
tar -xvf cifar10_sdxl_diversified.tar

mkdir cifar10
mv cifar10_MA cifar10
mv cifar10_generated cifar10
mv cifar10_web cifar10
mv cifar10_web_10 cifar10
mv cifar10_original_test cifar10
mv cifar10_nc cifar10
mv cifar10_c cifar10
mv cifar10_sdxl_diversified cifar10

rm cifar10_MA.tar
rm cifar10_generated.tar
rm cifar10_original_test.tar
rm cifar10_web.tar
rm cifar10_web_10.tar
rm cifar10_nc.tar
rm cifar10_c.tar
rm cifar10_sdxl_diversified.tar
