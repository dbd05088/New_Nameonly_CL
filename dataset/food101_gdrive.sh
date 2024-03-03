gdrive files download 1fSoOHT5eK56skulpmTijboEGw7Zjrob8 
gdrive files download 1wVbqPN5gIDLiiMFurbE80tXeD0xjNiiH 
gdrive files download 1pG6YA-q4kyzVs6By4ug6qA0704ApWcgi 
#gdrive files download 1kkwDnC9oQ-5JvktC7uC_2gNCiCrZVEJ7
gdrive files download 1Ni8GD7awuQaIhC23TNFYSIX4XNJ-7Eix
gdrive files download 1tILVlNQ3EpvMkBju6s3_0FpTJoQrteKP

tar -xvf food101_test_ma.tar
tar -xvf food101_train_ma.tar
tar -xvf food101_web.tar
#tar -xvf food101_web_10.tar
tar -xvf food101_sdxl_diversified.tar
tar -xvf food101_sdxl_diversified_newprompt.tar

mkdir food101
mv food101_train_ma food101
mv food101_test_ma food101
mv food101_web food101
#mv food101_web_10 food101
mv food101_sdxl_diversified food101
mv food101_sdxl_diversified_newprompt food101

rm food101_test_ma.tar
rm food101_train_ma.tar
rm food101_web.tar
#rm food101_web_10.tar
rm food101_sdxl_diversified.tar
rm food101_sdxl_diversified_newprompt.tar

