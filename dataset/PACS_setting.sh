gdown --id "1zLPhg_Yg685fZbkNE_lZE0S4RUp9yC8n"
gdown --id "1LfcRv1xFlLbT9Etx71JbpWzKSQiouxjO"
gdown --id "1szrba3mR9ZjXOQg7ABRj6T-GFm1PY159"
gdown --id "1eimtHK_-h_ynWQ_LbLFO7RJ8ZD8d8zIJ"

tar -xvf PACS_MA.tar
tar -xvf PACS_generated.tar
tar -xvf PACS_web.tar
tar -xvf PACS_prompt_ensemble.tar

mkdir PACS
mv PACS_MA PACS
mv PACS_generated PACS
mv PACS_web PACS
mv PACS_prompt_ensemble PACS

rm PACS_generated.tar
rm PACS_MA.tar
rm PACS_web.tar
rm PACS_prompt_ensemble.tar