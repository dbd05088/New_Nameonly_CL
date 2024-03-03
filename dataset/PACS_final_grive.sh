gdrive files download 19V07TMALoNa6Dzdezpkmo3aXhTrLB0LE 
gdrive files download 1E46aRn2KGKQM3iVaEdO6a0MXl0CHLlP_ 
gdrive files download 11zV8OnLYqQ9JQZVkxv_1OTWdtbscBsUn 
gdrive files download 1Q-leEeSjgWZNQUbapU1GmHKFr947_-K- 
gdrive files download 1pFiNMTBHV5KdDIX0eh7a6zy_afnIK2Rg
gdrive files download 1LfcRv1xFlLbT9Etx71JbpWzKSQiouxjO
gdrive files download 1bpMSWOmu59LksTThsDnVo2RE2l1epKAi # PACS_both_ensembled
gdrive files download 1zLPhg_Yg685fZbkNE_lZE0S4RUp9yC8n

tar -xvf PACS_final_train_ma.tar
tar -xvf PACS_final_test_ma.tar
tar -xvf PACS_final_web.tar
tar -xvf PACS_MA.tar
tar -xvf PACS_final_web_10.tar
tar -xvf PACS_final_sdxl_diversified.tar
tar -xvf PACS_both_ensembled.tar
tar -xvf PACS_generated.tar

mkdir PACS_final
mv PACS_MA PACS_final
mv PACS_final_train_ma PACS_final
mv PACS_final_test_ma PACS_final
mv PACS_final_web PACS_final
mv PACS_final_web_10 PACS_final
mv PACS_final_sdxl_diversified PACS_final
mv PACS_both_ensembled PACS_final
mv PACS_generated PACS_final

rm PACS_MA.tar
rm PACS_final_train_ma.tar
rm PACS_final_test_ma.tar
rm PACS_final_web.tar
rm PACS_final_web_10.tar
rm PACS_final_sdxl_diversified.tar
rm PACS_both_ensembled.tar
rm PACS_generated.tar