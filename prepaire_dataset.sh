mkdir dataset
echo "Downloading dataset !!!"
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
mv reviews_Cell_Phones_and_Accessories_5.json.gz dataset/
echo "Downloading finished !!!"
wget http://mattmahoney.net/dc/enwik8.zip

unzip enwik8.zip
perl main_.pl enwik8 > dataset/enwik8.txt

rm enwik8.zip
rm enwik8