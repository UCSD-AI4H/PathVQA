import  PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter
import time
import re
import os
import fitz

import sys;sys.path.append('/Users/jkooy/research/caption_image_extract')
import PyPDF2

from PIL import Image

import sys
from os import path
import PyPDF2

from PIL import Image, ImageOps

import sys
import struct
from os import path
import warnings
import io
from collections import namedtuple
warnings.filterwarnings("ignore")

import os

import fitz  # PyMuPDF

import PyPDF2
import cv2
import numpy as np


# #############################################   extract text
import sys
import importlib
importlib.reload(sys)
sys.path.append("/Users/jkooy/research/caption_image_extract/pdfminer.six-develop")

from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams, LTImage
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed



def save_image (lt_image, page_number, images_folder):
    result = None
    if lt_image.stream:
        file_stream = lt_image.stream.get_rawdata()
        file_ext = determine_image_type(file_stream[0:4])
    if file_ext:
        file_name = ''.join([str(page_number), '_', lt_image.name, file_ext])
    if write_file(images_folder, file_name, lt_image.stream.get_rawdata(), flags='wb'):
        result = file_name
    return result

def determine_image_type (stream_first_4_bytes):
#Find out the image file type based on the magic number comparison of the first 4 (or 2) bytes#
    file_type = None
    bytes_as_hex = b2a_hex(stream_first_4_bytes)
    if bytes_as_hex.startswith('ffd8'):
        file_type = '.jpeg'
    elif bytes_as_hex == '89504e47':
        file_type = ',png'
    elif bytes_as_hex == '47494638':
        file_type = '.gif'
    elif bytes_as_hex.startswith('424d'):
        file_type = '.bmp'
    return file_type


def write_file (folder, filename, filedata, flags='w'):
#Write the file data to the folder and filename combination
#(flags: 'w' for write text, 'wb' for write binary, use 'a' instead of 'w' for append)#
    result = False
    if os.path.isdir(folder):
        try:
            file_obj = open(os.path.join(folder, filename), flags)
            file_obj.write(filedata)
            file_obj.close()
            result = True
        except IOError:
            pass
    return result

path = r'/Users/jkooy/research/caption_image_extract/Basic_Pathology.pdf'


fp = open(path, 'rb')
praser = PDFParser(fp)
doc = PDFDocument()
praser.set_document(doc)
doc.set_parser(praser)


doc.initialize()

if not doc.is_extractable:
    raise PDFTextExtractionNotAllowed
else:
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    print('pdf')
    text_content = []
    for page in doc.get_pages():
        interpreter.process_page(page)
        layout = device.get_result()
        for x in layout:
            if (isinstance(x, LTTextBoxHorizontal)):
                with open(r'/Users/jkooy/research/caption_image_extract/1.txt', 'a') as f:
                    results = x.get_text()
                    # print(results)
                    f.write(results + '\n')
                    print('text')
            elif isinstance(x, LTImage):
                saved_file = save_image(x, page,'/Users/jkooy/research/caption_image_extract/'.txt)
                if saved_file:
                    text_content.append('<img src="' + os.path.join('/Users/jkooy/research/caption_image_extract/', saved_file) + '" />')
                    print('image')
                else:
                    print >> sys.stderr, "Error saving image on page"

#
############################ PyPDF2  ########################
from PyPDF2.pdf import PdfFileReader
import pandas as pd


# if __name__ == '__main__':
#     input1 = PyPDF2.PdfFileReader(open("/Users/jkooy/research/caption_image_extract/Basic_Pathology.pdf", "rb"))
#     pages = input1.getNumPages()
#     pic_list = []
#
#     # content = ""
#     #
#     # page_i = input1.getPage(46)
#     # xObject = page_i['/Resources']
#     # extractedText = page_i.extractText()
#     # print(extractedText)
#     #
#     #
#     # for i in range(pages):
#     #     page_i = input1.getPage(i)
#     #     xObject = page_i['/Resources']
#     #     extractedText = page_i.extractText()
#     #     content += extractedText + "\n"
#     #     print(content)
#
#     for i in range(pages):
#         page_i = input1.getPage(i)
#         xObject = page_i['/Resources']['/XObject'].getObject()
#         for obj in xObject:
#             if xObject[obj]['/Subtype'] == '/Image':
#                 size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
#                 data = xObject[obj].getData()
#                 if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
#                     mode = "RGB"
#                 else:
#                     mode = "P"
#
#                 if xObject[obj]['/Filter'] == '/FlateDecode':
#                     #                    img = Image.frombytes(mode, size, data)
#                     #                    img.save(obj[1:] + ".png")
#                     img = np.fromstring(data, np.uint8)
#                     img = img.reshape(size[1], size[0])
#                     #                     pdb.set_trace()
#                     img = 255 - img
#                     img = img + 1
#                     # cut img
#                     img = img[:, 220:-225]
#                     cv2.imwrite("/Users/jkooy/research/caption_image_extract/image/" + "obj" + obj[1:] + ".png", img)
#                     # pic_list.append("page" + "%05ui" % i + "obj" + obj[1:] + ".png")
#                     # pass
#
#                 elif xObject[obj]['/Filter'] == '/DCTDecode':
#                     #                    img = open(obj[1:] + ".jpg", "wb")
#                     #                    img.write(data)
#                     #                    img.close()
#                     img = np.fromstring(data, np.uint8)
#                     cv2.imwrite("/Users/jkooy/research/caption_image_extract/image/"+obj[1:] + ".jpg",img)
#                 elif xObject[obj]['/Filter'] == '/JPXDecode':
#                     #                    img = open(obj[1:] + ".jp2", "wb")
#                     #                    img.write(data)
#                     #                    img.close()
#                     img = np.fromstring(data, np.uint8)
#                     cv2.imwrite("/Users/jkooy/research/caption_image_extract/image/"+obj[1:] + ".jp2", img)
#                     # pass
#
#
#
# def pdf2pic(path, pic_path):
#     '''
#     :param path:
#     :param pic_path:
#     :return:
#     '''
#     t0 = time.clock()
#     checkXO = r"/Type(?= */XObject)"
#     checkIM = r"/Subtype(?= */Image)"
#
#     doc = fitz.open(path)
#     imgcount = 0
#     lenXREF = doc._getXrefLength()
#
#
#     for i in range(1, lenXREF):
#         text = doc._getXrefString(i)
#         isXObject = re.search(checkXO, text)
#         isImage = re.search(checkIM, text)
#         if not isXObject or not isImage:
#             continue
#         imgcount += 1
#         pix = fitz.Pixmap(doc, i)
#         new_name = path.replace('\\', '_') + "_img{}.png".format(imgcount)
#         new_name = new_name.replace(':', '')
#
#         if pix.n < 5:
#             pix.writePNG(os.path.join(pic_path, new_name))
#         else:
#             pix0 = fitz.Pixmap(fitz.csRGB, pix)
#             pix0.writePNG(os.path.join(pic_path, new_name))
#             pix0 = None
#         pix = None
#         t1 = time.clock()
#
# if __name__ == '__main__':
#     # pdf路径
#     path = r'/Users/jkooy/research/caption_image_extract/Basic_Pathology.pdf'
#     pic_path = r'/Users/jkooy/research/caption_image_extract/image'
#     if os.path.exists(pic_path):
#         raise SystemExit
#     else:
#         os.mkdir(pic_path)
#     m = pdf2pic(path, pic_path)
#
# readFile = "/Users/jkooy/research/caption_image_extract/Pathology.pdf"

# Read In Text
fileName = "/Users/jkooy/research/caption_image_extract/o.txt"
pdfTextfile = open(fileName, "r")
pdfText = pdfTextfile.read()

# Split text into blocks separated by double line break.
blocks = pdfText.split("\n\n")

# Remove all new lines within blocks to remove arbitary line breaks
captions = map(lambda x : x.replace("\n", ""), blocks)

# Which blocks are figure captions?
captionlist = []
for cap in captions:
    if re.search('^fig', cap, re.IGNORECASE):
        captionlist.append(cap)

# Done!
for caption in captions:
    f = open('/Users/jkooy/research/caption_image_extract/caption.txt', 'a+')
    f.write(caption)
    f.write('\n\n')
    f.close()

pdfFileReader =getPdfContent(readFile)

pdfFileReader = PdfFileReader(readFile)
pdf = pdfFileReader.getPage(30)

# import numpy as np
# import pandas as pd
#
# from stanfordcorenlp import StanfordCoreNLP
# import stanfordnlp
# #
# # np.load(".../self_dataset/Images/captions/captions.npy")
# original_data=np.load("self_dataset/Images/captions/captions.npy")
# #
# # image_list=[20,21,98,109,115,118,134,137,140,148]
# # image_pick=np.zeros((10,2))
# #
# #
# en_model = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27', lang='en')
# #
# df = pd.DataFrame(index = [20,21,98,109,115,118,134,137,140,148],columns = ['Caption','POS tagging'])
# # df.to_("tagging.csv")
# #
# # for image in image_list:
# #     # frame[f'{image}'] = original_data[image]
# #
# #     sentence=original_data[image]
# #     df.loc[image,'Caption'] = sentence
# #     sentence_tagging=en_model.pos_tag(sentence)
# #     df.loc[image, 'POS tagging'] = sentence_tagging
# #
# #     df.to_csv("tagging.csv")
#


# ######################################### deal with caption
# import numpy as np
# import re
#
# input_data = np.load("/Users/jkooy/research/self_dataset/Images 2/captions/captions.npy")
# print(input_data.shape)
# data = input_data.reshape(-1,1)
# # data = input_data
# print(data.shape)
#
# for index,i in enumerate(data):
#     print(i[0])
#     print(i[0].tolist())
#     i[0] = re.sub("\n", " ", i[0])
#     data[index]=i[0]+"\n\n"
# print(data)
#
# np.savetxt(r"/Users/jkooy/research/self_dataset/Images/captions/caption6.txt",data,fmt='%s')
#
# for index,i in enumerate(data):
#     f = open('/Users/jkooy/research/self_dataset/Images/captions/caption5.txt','w+')
#     i.strip('[')
#     i.strip(']')
#     f.write('\n\n\n'+str(i))
#     f.close()
# #
# df = pd.DataFrame(data)
# # df.to_csv("/Users/jkooy/research/self_dataset/Images/captions/test2.csv", index=False, header=True)
# # np.savetxt("/Users/jkooy/research/self_dataset/Images/captions/caption.txt",data,fmt='%s',delimiter="\n",newline='\n')
# #
# #
# # # stanfordnlp.download('en')
# #
# #
# # print ()



