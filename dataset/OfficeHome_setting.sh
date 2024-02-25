gdown --id "1mXLknDb--GavWYD3vlEmdlU7E-5se0ER"
gdown --id "1dqP7fEr8z5FvTw_rnlkxBM-eoJK0IcON"
gdown --id "1U3oFKEeVI7gXORCVz1Tij56b4vebN55Q"

tar -xvf OfficeHome_MA.tar
tar -xvf OfficeHome_generated.tar
tar -xvf OfficeHome_web.tar

mkdir OfficeHome
mv OfficeHome_MA OfficeHome
mv OfficeHome_generated OfficeHome
mv OfficeHome_web OfficeHome

rm OfficeHome_generated.tar
rm OfficeHome_MA.tar
rm OfficeHome_web.tar
