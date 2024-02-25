gdown --id "1G_aILsxwKAhfFPwSuc2ecnuhE9JH1RH-"
gdown --id "1I5W7bL46cduMMrNq4gH5KsUOonLejMuZ"

tar -xvf DomainNet_MA.tar
tar -xvf DomainNet_generated.tar

mkdir DomainNet
mv DomainNet_MA DomainNet
mv DomainNet_generated DomainNet

rm DomainNet_generated.tar
rm DomainNet_MA.tar
