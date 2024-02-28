gdrive files download 1wVzuRPp889hkoTnZA9m37M1VBG94gJck # PACS_web_10
gdrive files download 1LfcRv1xFlLbT9Etx71JbpWzKSQiouxjO # PACS_MA
gdrive files download 1zLPhg_Yg685fZbkNE_lZE0S4RUp9yC8n # PACS_generated
gdrive files download 1eimtHK_-h_ynWQ_LbLFO7RJ8ZD8d8zIJ # PACS_web

tar -xvf PACS_MA.tar
tar -xvf PACS_generated.tar
tar -xvf PACS_web.tar
tar -xvf PACS_web_10.tar

mkdir PACS
mv PACS_MA PACS
mv PACS_generated PACS
mv PACS_web PACS
mv PACS_web_10 PACS

rm PACS_generated.tar
rm PACS_MA.tar
rm PACS_web.tar
rm PACS_web_10.tar
