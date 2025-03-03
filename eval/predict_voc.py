#coding=gb2312
import cv2
from PIL import Image
import numpy as np
import os
import time
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model
 
torch.set_grad_enabled(False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def dnms(bboxes, scores, threshold=0.6):
    x1 = torch.tensor(bboxes[:, 0])
    y1 = torch.tensor(bboxes[:, 1])
    x2 = torch.tensor(bboxes[:, 2])
    y2 = torch.tensor(bboxes[:, 3])
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    order = order.cpu()
    
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]

        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2

        diagonal_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

        # 计算enclose框的左上角和右下角坐标
        enclose_x1 = torch.min(x1[i], x1[order[1:]])
        enclose_y1 = torch.min(y1[i], y1[order[1:]])
        enclose_x2 = torch.max(x2[i], x2[order[1:]])
        enclose_y2 = torch.max(y2[i], y2[order[1:]])

        # 计算enclose框的宽度和高度
        enclose_width = enclose_x2 - enclose_x1
        enclose_height = enclose_y2 - enclose_y1

        # 计算enclose框的对角线距离的平方
        enclose_diagonal_dist = enclose_width ** 2 + enclose_height ** 2

        # 计算DIOU
        diou = iou - (diagonal_dist / enclose_diagonal_dist)
        iou = torch.squeeze(diou)
        idx = np.nonzero(iou <= threshold)
        if idx.numel() == 0:
            break
           
        order = order[idx+1]
      
    return torch.LongTensor(keep)   

def save_result(img_name, boxes, ori_prob,  img_size):
    
    img_name = img_name.split('.')[0]
    
    output_dir = './pesudo_label/Voc_txt_result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    

    with open(os.path.join(output_dir, img_name + '.txt'), 'a') as f:
    # img_size [h ,w, c]
    
        if len(boxes.shape) == 1:
         
            xmin, ymin, xmax, ymax = boxes
            width, height = (xmax - xmin), (ymax - ymin)
            x_center, y_center = (xmax - width / 2), (ymax - height / 2)

            scale_x, scale_y, scale_width, scale_height = (x_center / img_size[1]), (y_center / img_size[0]),\
                (width / img_size[1]), (height / img_size[0])

            f.write(str(np.argmax(ori_prob.cpu().numpy(), 1)[0]) + ' ' + str(scale_x) + ' ' + str(scale_y) + ' ' + str(scale_width) + ' ' + str(scale_height) + ' ' + str(ori_prob[0][ori_prob[0].argmax()].item()) +'\n')
        else:
            for index, coordinate in enumerate(boxes):
                
                xmin, ymin, xmax, ymax = coordinate
                if (xmax - xmin) / img_size[1] > 0.9 and (ymax - ymin) / img_size[0] > 0.9:
                    continue
                width, height = (xmax - xmin), (ymax - ymin)
                x_center, y_center = (xmax - width / 2), (ymax - height / 2)

                scale_x, scale_y, scale_width, scale_height = (x_center / img_size[1]), (y_center / img_size[0]),\
                    (width / img_size[1]), (height / img_size[0])
                
                       
                f.write(str(np.argmax(ori_prob.cpu().numpy(), 1)[index]) + ' ' + str(scale_x) + ' ' + str(scale_y) + ' ' + str(scale_width) + ' ' + str(scale_height) + ' ' + str(ori_prob[index][ori_prob[index].argmax()].item()) +'\n')
        f.close()

# plot box by opencv
def plot_result(pil_img, prob, boxes, objs, save_name=None, imshow=False, imwrite=False, save_txt=False):    
    judge_score = torch.tensor(np.amax(prob, axis=1, keepdims=True))
    keep = dnms(boxes, judge_score, 0.4)
    
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    prob = torch.tensor(prob)
    prob = prob[keep]
    boxes = boxes[keep]
    
    if save_txt:
        img_size = opencvImage.shape
        save_result(save_name, boxes, prob, img_size)    
        imshow = False
        imwrite = False
    
    LABEL =['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    if prob.shape[0] == 1:
        boxes = torch.tensor(boxes).unsqueeze(0)
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl].item() * 100, 2))
 
        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)
    
    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)
 
    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)
 
# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b
 
def load_model(model_path , args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path) # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model
 
# 图像的推断
def detect(im, model, transform, prob_threshold=0.45):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
 
 
    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)
   
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :]
    objs = outputs['obj'][0]
    
    keep = probas.max(-1).values > prob_threshold
 
    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
 
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, objs[keep], end - start
 
 
if __name__ == "__main__":
    
    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('/data/lhm/DETR-test/exps/best.pth',main_args) # <--修改为自己加载模型的路径
 
    files = os.listdir("/data/lhm/VOC2COCO/val2017") # <--修改为待预测图片所在文件夹路径
 
    cn = 0
    waste=0
    for file in files:
        img_path = os.path.join("/data/lhm/VOC2COCO/val2017", file) # <--修改为待预测图片所在文件夹路径
        im = Image.open(img_path)
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        scores, boxes, objs, waste_time  = detect(img, dfdetr, transform)
        plot_result(im, scores, boxes, objs, save_name=file, imshow=False, imwrite=True, save_txt=False)
 
        cn+=1 
        waste+=waste_time
        waste_avg = waste/cn
        print(waste_avg)