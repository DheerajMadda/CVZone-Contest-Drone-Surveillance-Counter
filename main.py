import sys
sys.path.insert(0, './packages')
sys.path.insert(0, './packages/yolov5')
import os
import time
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized

from classes.yolov5_class import Yolov5
from classes.deep_sort_class import Deep_Sort
from classes.util_class import  draw_boxes, xyxy_to_xywh, xyxy_to_tlwh

def run(opt):

    # 1. Variables
    source = opt.source
    yolo_weights = opt.weights
    save_txt = opt.save_txt
    
    object_confidence_threshold = 0.7
    IOU_threshold_for_NMS = 0.5

    output = os.path.join(os.getcwd(),'outputs')
    if not os.path.exists(output):
        os.makedirs(output)

    # 2. Initilaize and Load Deep_Sort
    deep_sort_weights = 'packages/deep_sort_pytorch/deep_sort/deep/checkpoint'
    config_deepsort = 'packages/deep_sort_pytorch/configs/deep_sort.yaml'
    ds = Deep_Sort(deep_sort_weights, config_deepsort)
    deepsort = ds.load()

    # 3. Load Yolov5 Model
    device = select_device('cuda:0')
    yolov5 = Yolov5(yolo_weights, device)
    model = yolov5.load()
    # 3.1 get class names
    names = model.module.names if hasattr(model, 'module') else model.names
    # 3.2 Load source
    dataset = yolov5.load_images(source)

    # 4. Initilize cv2 Writer & text writer
    vid_name = source.split('/')[-1].split('.')[0] + '.avi'
    vid_path = output + '/' + vid_name
    txt_name = source.split('/')[-1].split('.')[0] + '.txt'
    txt_path = output + '/' + txt_name

    vid = cv2.VideoCapture(source)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(vid_path, codec, vid_fps, (vid_width, vid_height))

    start_time = time.time()
	
    labelStoreVec = []
    dissapearedStoreVec = []

    # 5. Iterate dataset
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if yolov5.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)  # 3 dim to 4 dim

        t1 = time_synchronized()
        # 5.1 yolov5 inference
        pred = model(img, augment=False)[0]
        
        # 5.2 Apply NMS
        pred = non_max_suppression(pred, 
                                object_confidence_threshold,
                                IOU_threshold_for_NMS,
                                classes=None,
                                agnostic=False
                            )
        t2 = time_synchronized()
        ids = []
        # 5.3 Process detections
        for i, det in enumerate(pred):  # detections per image
            p, im0 = path, im0s
            s = ""

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                print("det:", det[:,:4])
            
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
            
                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]

                    idx = 0

                    for ID_stored in labelStoreVec:
                        if ID_stored not in identities:
                            if ID_stored not in dissapearedStoreVec:
                                dissapearedStoreVec.append(ID_stored)


                    for ID in identities:

                        if ID not in labelStoreVec:
                            if frame_idx>10:
                                if bbox_xyxy[idx][1]/im0.shape[0]>0.5:
                                    labelStoreVec.append(ID)
                            else:
                                labelStoreVec.append(ID)
                        else:
                            for k in range(len(dissapearedStoreVec)):
                                if ID == dissapearedStoreVec[k]:
                                    identities[idx] = ID*100

                        idx+=1


                    draw_boxes(im0, bbox_xyxy, identities)

                    # Write MOT compliant results to file
                    if save_txt:
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy) # to MOT format
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
							#ids.append(identity)
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                
            else:
                deepsort.increment_ages()

        cv2.putText(im0, "DHEERAJ MADDA", (850,90), 4, 1.5, (0,0,255), 2)
        cv2.putText(im0, "{}".format(len(labelStoreVec)), (1750,90), 4, 1.5, (0,0,255), 2)
        out.write(im0)

    out.release()
    end_time = time.time() - start_time
    print('\n'*2 +'[Done]')
    print('[Time Taken] : {:.2f} seconds.'.format(end_time))
    print('Results saved in outputs folder')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Path to the source")
    parser.add_argument("-w", "--weights", help="Path to the Yolo weights")
    parser.add_argument("-st", "--save-txt", help="Saves the Tracking Details to a txt file", action="store_true")
    args = parser.parse_args()
    run(args)
