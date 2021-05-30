# PathVQA
The code for the paper Pathological Visual Question Answering. 

## Dataset
### Prepare PathVQA dataset:  
Google Drive Link: https://drive.google.com/file/d/1utnisF_HJ8Yk9Qe9dBe9mxuruuGe7DgW/view?usp=sharing

### Prepare [BCCD dataset](https://public.roboflow.com/object-detection/bccd/1), which is used to finetune Faster RCNN. 
Choose Pascal VOC XML format. Name the folder bccd and place it somewhere. The run bccd/prep_bccd_voc.sh outside the bccd folder. 
```
bash prep_bccd_voc.sh
```

## Method 1
Data and environment preparation:
```bash
cd method1
mkdir data
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/

pip install -r requirements.txt
```

Then run the bash scripts in run/pvqa_img folder. 
The difference between run/pvqa_img and run/pvqa_img3 is that, the image features used are different. 
Explanation of bash files: 
* pvqa_seeds_scratch.sh: Method 1 on PathVQA. 
* pvqa_seeds_qabl_scratch.sh: Method 1 without image on PathVQA. 
* pvqa_seeds_vq.sh: Method 1 with CMSSL-IQ on PathVQA. 
* pvqa_seeds_qa_vq_va_vqa.sh: Method 1 with joint pretraining on PathVQA. 
* pvqa_finetune_ignore.sh: Method 1 with ignoring on PathVQA. 
Example usage:
```bash
bash run/pvqa_img/pvqa_seeds_scratch.sh
```

## Method 2
Data preparation: 
```
cd method2
mkdir data
ln -s path/to/pvqa/ data/pvqa
```
The naming of bash scripts is similar to LXMERT. Example usage:
```bash
bash run/pvqa_img3/pvqa_run_scratch.sh
bash run/pvqa_img3/pvqa_finetune_ignore.sh
```

## Faster-rcnn finetune on Visual Genome and BCCD
Preparation:
download Faster R-CNN model pretrained on VG from https://drive.google.com/file/d/1UtySeaz1klQ2WoPfP7NIUj0ft-J5pdpy/view?usp=sharing, 
and place it in folder faster-rcnn.pytorch-pytorch-1.0
```bash
cd faster-rcnn.pytorch-pytorch-1.0
pip install -r requirements.txt
cd lib 
python setup.py build develop
cd ..
mkdir data
ln -s path/to/bccd data/bccd
bash run/train_bccd.sh
```
Copy the trained faster-rcnn model to folder Faster-R-CNN-with-model-pretrained-on-Visual-Genome-master. 
Rename the model file 'faster_rcnn_res101_bccd.pth'. 
Then generate image features:
```bash
cd  Faster-R-CNN-with-model-pretrained-on-Visual-Genome-master
pip install -r requirements.txt
cd lib 
python setup.py build develop
cd ..
mkdir data
ln -s path/to/pvqa data/pvqa
ln -s path/to/bccd data/bccd
bash run_bccd.sh
```

The new image features will be in data/pvqa. 
Run the bash scripts in method 1 or method2 to use the new image features. 


## Ignoring
```bash
cd method1
mkdir data
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/
bash run/pvqa_VG_BCCD/pvqa_finetune_ignore.sh
```

```bash
cd method2
mkdir data
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/
bash run/pvqa_img3/pvqa_finetune_ignore.sh
```
