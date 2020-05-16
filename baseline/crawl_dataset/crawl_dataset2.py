# from requests_html import HTMLSession
#
# session = HTMLSession()
# url = 'http://peir.path.uab.edu/library/index.php?/category/2'
#
# r = session.get(url)
# print(r.html.text)
#
# r.html.absolute_links

import requests
from bs4 import BeautifulSoup
import sys
from urllib import  request
from bs4 import BeautifulSoup
import os


def geturl(page):
    soup = BeautifulSoup(page, 'html.parser')
    suburls = []

    # uls3 = soup.find_all("div")
    # uls4 = soup.find_all('span', attrs={"class":"ui-li-count"})
    # uls2 = soup.find_all('span',attrs={"class":"wrap2"})

    uls = soup.find_all('li',attrs={"class":"liVisible"})
    print(uls,"\n")
    print(len(uls), "\n")
    for ul in uls:
        tags=ul.find_all('a')
        # print(tags,"\n")
        for tag in tags:
            suburls.append('http://peir.path.uab.edu/library/' +tag.get('data-picture-url'))

        # try:
        #     links.append(dd.get('href'))
        # except AttributeError:
        #     print (dd)
        # if dd is not NoneType:
        #     links.append(dd.get('href'))
    print(suburls)
    print(len(suburls), "\n")
    return suburls
def getHtmlCode(url):
    headers = {
        # huipu laptop
        'User-Agent' : 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko)  Chrome/56.0.2924.87 Mobile Safari/537.36'
    }
    url1 = request.Request(url, headers = headers)
    page = request.urlopen(url1).read().decode()
    return page
def getImg(page, count, folder):
    soup = BeautifulSoup(page,'html.parser')
    book_list_name = soup.find_all(attrs={"class":"imageComment"})
    for book_one in book_list_name:
        print(book_one.string)
        f = open('/Users/jkooy/research/self_dataset/crawled_images/caption_crawled.txt', 'a+')
        f.write(str(count)+'\t'+book_one.string)
        f.write('\n\n\n')
        f.close()
    book_list_img = soup.find_all('img', attrs={'id':'theMainImage' })

    for book_one in book_list_img:
        book_img_url = book_one.get('src')
        book_img_url='http://peir.path.uab.edu/library/'+book_img_url
        print(book_img_url)

        folder_exist = os.path.exists('/Users/jkooy/research/self_dataset/crawled_images/%s'%folder)
        if not folder_exist:
            os.makedirs('/Users/jkooy/research/self_dataset/crawled_images/%s'%folder)
            print
            "---  new folder...  ---"
            print
            "---  OK  ---"

        else:
            print
            "---  There is this folder!  ---"
        request.urlretrieve(book_img_url, '/Users/jkooy/research/self_dataset/crawled_images/%s/%s.jpg' %(folder, count))

url = 'http://peir.path.uab.edu/library/index.php?/category/2'
page = getHtmlCode(url)
soup = BeautifulSoup(page, 'html.parser')
links = []
h3 = soup.find_all("div")
uls1 = soup.find_all("div",attrs={"data-role":"collapsible"})
uls = soup.find_all("ul",attrs = {"data-role":"listview"})
suburls=[]
for ul in uls:
    tags = ul.find_all('a')
    # print(tags,"\n")
    for tag in tags:
        suburls.append('http://peir.path.uab.edu/library/' + tag.get('href'))
print(suburls)
print(len(suburls))
newsuburls = suburls[7:30]
print(len(newsuburls))

count = 0
folder = 0
for url in newsuburls:
    print(url)
    # url = 'http://peir.path.uab.edu/library/index.php?/category/3'
    page = getHtmlCode(url)
    soup = BeautifulSoup(page, 'html.parser')
    links = []
    uls = soup.find_all("div",attrs={"data-role":"content"})
    suburls=[]
    for ul in uls:
        tags = ul.find_all('a')
        # print(tags,"\n")
        for tag in tags:
            suburls.append('http://peir.path.uab.edu/library/' + tag.get('href'))
    print(suburls)


    for url in suburls:
        page = getHtmlCode(url)
        sub_url_list = geturl(page)
        for sub_url in sub_url_list:
            each_image_page = getHtmlCode(sub_url)
            getImg(each_image_page, count, folder)
            count += 1
        folder += 1







