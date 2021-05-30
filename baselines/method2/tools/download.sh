
## Script for downloading data

# GloVe Vectors
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

# Annotations
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip

# Image Features
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/test2014.zip

unzip data/trainval.zip -d data
unzip data/test2014.zip -d data
rm data/trainval.zip
rm data/test2014.zip

wget -P data https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip
unzip data/test2015.zip -d data
rm data/test2015.zip


wget -P data http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip data/train2014.zip -d data/
rm data/train2014.zip

wget -P data http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip data/val2014.zip -d data/
rm data/val2014.zip

wget -P data http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip data/test2015.zip -d data/
rm data/test2015.zip

# Download Pickle caches for the pretrained model from
# anonymous link
# and extract pkl files under data/cache/.

mkdir -p data/cache
wget --no-check-certificate 'anonymous link' -O data/cache/cache.pkl.zip
unzip data/cache/cache.pkl.zip -d data/cache/
rm data/cache/cache.pkl.zip

# anonymous link

