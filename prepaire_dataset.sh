mkdir dataset
echo "Downloading dataset"
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
gzip reviews_Cell_Phones_and_Accessories_5.json.gz
mv reviews_Cell_Phones_and_Accessories_5.json.gz dataset/data.json
echo "Unpack finished, work with data.json file"
