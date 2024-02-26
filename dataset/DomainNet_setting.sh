gdown --id "1G_aILsxwKAhfFPwSuc2ecnuhE9JH1RH-"
gdown --id "1I5W7bL46cduMMrNq4gH5KsUOonLejMuZ"
gdown --id "1UDKD-volVNLmujzmdBGIBDhbc2o1-wsf"
tar -xvf DomainNet_MA.tar
tar -xvf DomainNet_generated.tar
tar -xvf DomainNet_web.tar

mkdir DomainNet
mv DomainNet_MA DomainNet
mv DomainNet_generated DomainNet
mv DomainNet_web DomainNet

rm DomainNet_generated.tar
rm DomainNet_MA.tar
rm DomainNet_web.tar
