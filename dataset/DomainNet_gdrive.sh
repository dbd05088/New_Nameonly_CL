gdrive files download 1eCHHO8eheuM-FT0t13LbmOoTGxM0e-jO
gdrive files download 1Gnqna2AjQk-wVIwogGWS04MCvYnTmkdi
gdrive files download 1UDKD-volVNLmujzmdBGIBDhbc2o1-wsf
gdrive files download 1VNIz2xPm6ICKzp8i7KhhyN2VYv4fkQZY
gdrive files download 1I5W7bL46cduMMrNq4gH5KsUOonLejMuZ

tar -xvf DomainNet_train_ma.tar
tar -xvf DomainNet_test_ma.tar
tar -xvf DomainNet_sdxl_diversified.tar
tar -xvf DomainNet_web.tar
tar -xvf DomainNet_MA.tar

mkdir DomainNet
mv DomainNet_MA DomainNet
mv DomainNet_train_ma DomainNet
mv DomainNet_test_ma DomainNet
mv DomainNet_web DomainNet
mv DomainNet_sdxl_diversified DomainNet

rm DomainNet_MA.tar
rm DomainNet_sdxl_diversified.tar
rm DomainNet_train_ma.tar
rm DomainNet_test_ma.tar
rm DomainNet_web.tar
