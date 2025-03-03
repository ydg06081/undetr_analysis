#coding=gb2312
import os
import json
import cv2
import random
import time
from PIL import Image

coco_format_save_path='/data/lhm/UN_DETR/'   #Ҫ���ɵı�׼coco��ʽ��ǩ�����ļ���
yolo_format_classes_path='/data/lhm/UN_DETR/eval/class.txt'     #����ļ���һ��һ����
yolo_format_annotation_path='/data/lhm/UN_DETR/pesudo_label/Voc_txt_result'  #yolo��ʽ��ǩ�����ļ���
img_pathDir='/data/lhm/VOC2COCO/val2017'    #ͼƬ�����ļ���

LABLE_DICT = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog":5, "horse": 6, "sheep": 7, "airplane": 8, "bicycle": 9, "boat": 10, "bus": 11, 
              "car": 12, "motorcycle": 13, "train": 14, "bottle": 15, "chair": 16, "dining table": 17, "potted plant": 18, "couch": 19, "tv": 20}

with open(yolo_format_classes_path,'r') as fr:                               #�򿪲���ȡ����ļ�
    lines1=fr.readlines()
# print(lines1)
categories=[]                                                                 #�洢�����б�
for j,label in enumerate(lines1):
    label=label.strip()
    if label == 'unknow object':
        categories.append({'id':81,'name':label,'supercategory':'None'}) 
    else:
        categories.append({'id':j+1,'name':label,'supercategory':'None'})         #�������Ϣ��ӵ�categories��
# print(categories)

write_json_context=dict()                                                      #д��.json�ļ��Ĵ��ֵ�
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '����ss', 'date_created': '2022-07-8'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

#�������Ĵ�����Ҫ���'images'��'annotations'��keyֵ
imageFileList=os.listdir(img_pathDir)                                           #�������ļ����µ������ļ������������ļ�����ӵ��б���
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                             #��ȡͼƬ�ľ���·��
    image = Image.open(imagePath)                                               #��ȡͼƬ��Ȼ���ȡͼƬ�Ŀ�͸�
    W, H = image.size

    img_context={}                                                              #ʹ��һ���ֵ�洢��ͼƬ��Ϣ
    #img_name=os.path.basename(imagePath)                                       #����path�����ļ��������path��/��\��β����ô�ͻ᷵�ؿ�ֵ
    img_context['file_name']=imageFile
    
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2022-07-8'
    img_context['id']= int(imageFile.split('.')[0])                                                      #��ͼƬ��id
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)                            #����ͼƬ��Ϣ��ӵ�'image'�б���


    txtFile=imageFile.split('.')[0]+'.txt'                                               #��ȡ��ͼƬ��ȡ��txt�ļ�
    with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
        lines=fr.readlines()                                                   #��ȡtxt�ļ���ÿһ�����ݣ�lines2��һ���б�������һ��ͼƬ�����б�ע��Ϣ
    for j,line in enumerate(lines):

        bbox_dict = {}                                                          #��ÿһ��bounding box��Ϣ�洢�ڸ��ֵ���
        # line = line.strip().split()
        # print(line.strip().split(' '))

        class_id,x,y,w,h, Uscore=line.strip().split(' ')                                          #��ȡÿһ����ע�����ϸ��Ϣ
        
        class_id,x, y, w, h, Uscore = int(class_id), float(x), float(y), float(w), float(h), float(Uscore)       #���ַ�������תΪ�ɼ����int��float����
        # if class_id == 81:
        #     class_id = 1
   
        
        xmin=(x-w/2)*W                                                                    #����ת��
        ymin=(y-h/2)*H
        xmax=(x+w/2)*W
        ymax=(y+h/2)*H
        w=w*W
        h=h*H

        bbox_dict['id']=i*10000+j                                                         #bounding box��������Ϣ
        bbox_dict['image_id']= int(imageFile.split('.')[0])
        bbox_dict['category_id']=class_id                                               #ע��Ŀ�����Ҫ��һ
        bbox_dict['iscrowd']=0
        height,width=abs(ymax-ymin),abs(xmax-xmin)
        bbox_dict['area']=height*width
        bbox_dict['bbox']=[xmin,ymin,w,h]
        bbox_dict['score'] = Uscore
        bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
        write_json_context['annotations'].append(bbox_dict)                               #��ÿһ�����ֵ�洢��bounding box��Ϣ��ӵ�'annotations'�б���

name = os.path.join(coco_format_save_path,"train"+ '.json')
with open(name,'w') as fw:                                                                #���ֵ���Ϣд��.json�ļ���
    json.dump(write_json_context,fw,indent=2)

