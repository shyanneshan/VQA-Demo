import argparse
import csv
import os
import shutil
import zipfile

from generateQA import generateQA
from generateVQA import generateVQA, insertInformation
import mysqlConnector

def organizeOriginalData(src_path):
    with open(src_path + '/relationship.csv', 'r') as f:
        reader = csv.reader(f)
        print(type(reader))
        img_txt_dic = {}
        for row in reader:
            print(row)
            if row[0] == 'txtid':
                continue
            img_txt_dic[row[1]] = row[0]

        print(img_txt_dic)
        if not os.path.exists(src_path + '/organizedData'):
            os.mkdir(src_path + '/organizedData')

        for key, value in img_txt_dic.items():
            source_img_path = src_path + '/img/' + key + '.jpg'
            source_txt_path = src_path + '/txt/' + value + '.txt'
            new_dir_path = src_path + '/organizedData/' + key
            os.mkdir(new_dir_path)
            target_img_path = new_dir_path + '/' + key + '.jpg'
            target_txt_path = new_dir_path + '/' + value + '.txt'

            shutil.copy(source_img_path, target_img_path)
            shutil.copy(source_txt_path, target_txt_path)

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print(zip_src,'This is not zip')

def copyImgToImgpath(zip_dest,imgDest):
    imgSrcPath = zip_dest + '/img'
    if not os.path.exists(imgDest +'/'+ zip_dest.split('/')[-1]):
        os.mkdir(imgDest +'/'+ zip_dest.split('/')[-1])
    for i in os.listdir(imgSrcPath):
        src = imgSrcPath +'/'+ i

        dest = imgDest +'/'+ zip_dest.split('/')[-1] + '/' + i
        shutil.copy(src, dest)


def main(src,imgpath):


    zip_dest=src[:-4]

    dataset_name = zip_dest.split("/")[-1]
    train, valid, test = mysqlConnector.getPro(dataset_name)
    unzip_file(src, zip_dest)

    organizeOriginalData(zip_dest)
    copyImgToImgpath(zip_dest,imgpath)

    QApath = zip_dest + "/QA/"
    VQApath = zip_dest + "/VQA/"
    generateQA(zip_dest+"/organizedData/", QApath, train, valid, test)
    insertInformation(zip_dest+"/organizedData/", dataset_name, VQApath, train, valid, test)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test')
    parser.add_argument('--imgpath')

    args = parser.parse_args()
    print(args.name)
    main(args.name,args.imgpath)

