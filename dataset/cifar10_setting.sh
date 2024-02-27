gdown --id "1W6YRfL-u9ySIdXWkHiQ0Whq9m8EtzQhb"
gdown --id "16DTKBJjjzU4Z3jmiIYwgEPrXb4Hju3B4"
gdown --id "1VxdSntW9y6tzzSD1aiG_YGaL3jzpJv4g"
gdown --id "1JsokoJg1ws0Pbz2HMiosBSpPh4gq7KWL"
gdown --id "1gTJP9ciBwg-bpqdtvlbSDWEjNBQ4x4bu"
gdown --id "179g6v9hOAwDKXrgQ7Gs3-5hifjVt0UAu"
gdown --id "1TdIXr9uwdCcjDPBmnD_VQ2qv30uNKyxQ"

tar -xvf cifar10_MA.tar
tar -xvf cifar10_generated.tar
tar -xvf cifar10_web.tar
tar -xvf cifar10_web_10.tar
tar -xvf cifar10_original_test.tar
tar -xvf cifar10_nc.tar
tar -xvf cifar10_c.tar

mkdir cifar10
mv cifar10_MA cifar10
mv cifar10_generated cifar10
mv cifar10_web cifar10
mv cifar10_web_10 cifar10
mv cifar10_original_test cifar10
mv cifar10_nc cifar10
mv cifar10_c cifar10

rm cifar10_MA.tar
rm cifar10_generated.tar
rm cifar10_original_test.tar
rm cifar10_web.tar
rm cifar10_web_10.tar
rm cifar10_nc.tar
rm cifar10_c.tar
