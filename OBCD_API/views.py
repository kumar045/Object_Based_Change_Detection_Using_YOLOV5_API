from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import status
from .models import *
from .serializers import OBCDSerializer
from os import walk
from os import listdir
import random
import pdfkit
import argparse
import cv2
import numpy as np
import os
import pandas as pd 
import glob
import csv
import os
from PIL import Image, ImageDraw
import math
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os.path
from os import path
import random 
import pdfkit 
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import dataframe_image as dfi
import sys
sys.argv=['']
del sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Create your views here.
class  OBCDAPIView(CreateAPIView):
    serializer_class = OBCDSerializer
    queryset = OBCD.objects.all()
    def create(self, request, format=None):
        """
                Takes the request from the post and then processes the algorithm to extract the data and return the result in a
                JSON format
                :param request:
                :param format:
                :return:
                """

        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():
            tif_images_folder_path=self.request.data['tif_images_folder_path']

            content = []

            tif_images_folder_path= "C:\\Users\\Shivam\\Pictures\\Photos\\" + str(tif_images_folder_path) 

            print("tif_images_folder_path:::::",tif_images_folder_path)

            list_for_time_series_data=[]
            for name in glob.glob(str(tif_images_folder_path)+"\\*.jpg"):
                print(name)

                print("main_image_url:::::",name)


                old = self.old_slice_image_path(name)
                clear_imgs, unclear_imgs, a=self.load_images('C:\\Users\\Shivam\\Pictures\\Photos\\split_images')

                print(clear_imgs, unclear_imgs, a)


                list_of_files = glob.glob('C:\\Users\\Shivam\\Pictures\\Photos\\runs\\detect/*') # * means all if need specific format then *.csv
                latest_file = max(list_of_files, key=os.path.getctime)
                parser = argparse.ArgumentParser()
                parser.add_argument('--weights', nargs='+', type=str, default='C:\\Users\\Shivam\\Pictures\\Photos\\last.pt', help='model.pt path(s)')
                parser.add_argument('--source', type=str, default='C:\\Users\\Shivam\\Pictures\\Photos\\split_images', help='source')  # file/folder, 0 for webcam
                parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
                parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
                parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
                parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                parser.add_argument('--view-img', action='store_true', help='display results')
                parser.add_argument('--save-txt', type=bool,default=1, help='save results to *.txt')
                parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
                parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
                parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
                parser.add_argument('--augment', action='store_true', help='augmented inference')
                parser.add_argument('--update', action='store_true', help='update all models')
                parser.add_argument('--project', default='C:\\Users\\Shivam\\Pictures\\Photos\\runs/detect', help='save results to project/name')
                parser.add_argument('--name', default='exp', help='save results to project/name')
                parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
                opt = parser.parse_args()
                print(opt)
                source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
                device=opt.device
                project=opt.project
                exist_ok=opt.exist_ok
                name=opt.name
                conf_thres=opt.conf_thres
                iou_thres=opt.iou_thres
                classes=opt.classes
                agnostic_nms=opt.agnostic_nms
                save_conf=opt.save_conf
                augment=opt.augment

                with torch.no_grad():
                    if opt.update:  # update all models (to fix SourceChangeWarning)
                        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                            self.detect(source, weights, view_img, save_txt, imgsz,device,project,exist_ok,name,conf_thres,iou_thres,classes,agnostic_nms,save_conf,augment)
                            strip_optimizer(opt.weights)
                    else:
                        self.detect(source, weights, view_img, save_txt, imgsz,device,project,exist_ok,name,conf_thres,iou_thres,classes,agnostic_nms,save_conf,augment)



                self.txt_reader_func()

                
                
                [number_of_building,number_of_aircraft,number_of_helicopter,number_of_large_vehicle,number_of_small_vehicle,number_of_ship]=self.total_count_of_object_of_all_spilted_images()
                list_for_time_series_data.append([number_of_building,number_of_aircraft,number_of_helicopter,number_of_large_vehicle,number_of_small_vehicle,number_of_ship])
                print([number_of_building,number_of_aircraft,number_of_helicopter,number_of_large_vehicle,number_of_small_vehicle,number_of_ship])
                print(list_for_time_series_data)

            self.plot(list_for_time_series_data)
            self.csv_and_table(list_for_time_series_data)

                
            # add result to the dictionary and revert as response
            mydict = {
                'status': True,
                'response':
                    {

                        'Image_Information':"Image_Information",

                    }
            }
            content.append(mydict)

            return Response(content, status=status.HTTP_200_OK)
        errors = serializer.errors

        response_text = {
            "status": False,
            "response": errors
        }
        return Response(response_text, status=status.HTTP_400_BAD_REQUEST)


    def total_count_of_object_of_all_spilted_images(self):
        path="C:\\Users\\Shivam\\Pictures\\Photos\\split_csv_file\\*.csv"
        number_of_building=0
        number_of_aircraft=0
        number_of_helicopter=0
        number_of_large_vehicle=0
        number_of_small_vehicle=0
        number_of_ship=0

        for i in glob.glob(path):
            data_frame=pd.read_csv(i)
            print(data_frame.head)
            data_frame.iloc[:,1]
            print(data_frame.iloc[:,1])
            print(data_frame.iloc[:,2])
            Building=data_frame.iloc[0:1,2].values
            Aircraft=data_frame.iloc[1:2,2].values
            Helicopter=data_frame.iloc[2:3,2].values
            large_vehicle=data_frame.iloc[3:4,2].values
            Small_Vehicle=data_frame.iloc[3:4,2].values
            ship=data_frame.iloc[4:5,2].values 
            print(Building[0],Aircraft[0],Helicopter[0],large_vehicle[0],Small_Vehicle[0],ship[0])
            number_of_building=number_of_building+Building[0]
            number_of_aircraft=number_of_aircraft+Aircraft[0]
            number_of_helicopter=number_of_helicopter+Helicopter[0]
            number_of_large_vehicle=number_of_large_vehicle+large_vehicle[0]
            number_of_small_vehicle=number_of_small_vehicle+Small_Vehicle[0]
            number_of_ship=number_of_ship+ship[0]
        return number_of_building,number_of_aircraft,number_of_helicopter,number_of_large_vehicle,number_of_small_vehicle,number_of_ship

                

    def old_slice_image_path(self,old_path):
        img = cv2.imread(old_path)
        i = 0
        img_1_rows = 0
        img_1_col = 0

        for r in range(0, img.shape[0], 750):
            img_1_rows = img_1_rows + 1
            # print("img_1_rows ::", img_1_rows)
            for c in range(0, img.shape[1], 750):
                img_1_col = img_1_col + 1
                # print("img_1_col ::", img_1_col)

                # cv2.imwrite("old/" + f"img{r}_{c}.png", img[r:r + 1000, c:c + 1000, :])
                path="C:\\Users\\Shivam\\Pictures\\Photos\\split_images/" + str(i) +".jpg"
                cv2.imwrite(path,
                            img[r:r + 750, c:c + 750, :])
                i += 1
        return img_1_rows, img_1_col

    def load_images(self,folder):
        images = []
        a = 0
        unclear_imgs = 0
        clear_imgs = 0
        for filename in os.listdir(folder):
            a = a + 1
            print(os.path.join(folder, filename))
            # img = cv2.imread(os.path.join(folder,filename))
            img = np.array(Image.open(os.path.join(folder, filename)))
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F, ksize=1).var()
            print(laplacian_var)

            if (laplacian_var >= 90) & (laplacian_var <= 4500):
                print("Image Clear")
                path="C:\\Users\\Shivam\\Pictures\\Photos\\splited_clear_images"
                cv2.imwrite(os.path.join(path,str(filename)+".jpg"),img)
                clear_imgs = clear_imgs + 1
            else:
                print("Image unclear")
                unclear_imgs = unclear_imgs + 1
            if img is not None:
                images.append(img)

        print("Total sub images = ", a)
        # print("Total sub-images detected clear", ci)

        return clear_imgs, unclear_imgs, a




    def detect(self,source,weights,view_img,save_txt,imgsz,device,project,exist_ok,name,conf_thres,iou_thres,classes,agnostic_nms,save_conf,augment,save_img=False):

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        colors = [[random.randint(200, 255) for _ in range(3)] for _ in names]


        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                #if view_img:
                    #cv2.imshow(str(p), im0)
                    #if cv2.waitKey(1) == ord('q'):  # q to quit
                        #raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


    def txt_reader_func(self):
        list_of_files = glob.glob('C:\\Users\\Shivam\\Pictures\\Photos\\runs\\detect/*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)

        for name1 in glob.glob(str(latest_file)+"/labels/*.txt"):
            name_of_file=os.path.splitext(os.path.basename(name1))[0]
            file1 = open(name1, 'r')
            Lines = file1.readlines()
            count=0
            building = 0
            aircraft=0
            helicopter=0
            large_vehicle=0
            Ship=0
            Small_Vehicle=0
            arr=[]
            ar=[]
            a=[]
            for line in Lines:
                arr.append(line)
                count=count+1
            for j in arr:
                ar.append(j.split("\n"))
            for i in range(count):
                if((str(ar[i][0]).split()[0])=="0"):
                    building=building+1
                elif((str(ar[i][0]).split()[0])=="1"):
                    aircraft=aircraft+1
                elif(((str(ar[i][0]).split()[0])=="2")):
                    helicopter=helicopter+1
                elif(((str(ar[i][0]).split()[0])=="3")):
                    large_vehicle= large_vehicle+1
                elif(((str(ar[i][0]).split()[0])=="5")):
                    Ship= Ship+1  
                elif(((str(ar[i][0]).split()[0])=="4")):
                    Small_Vehicle= Small_Vehicle+1    
            print(building,aircraft,helicopter,large_vehicle,Ship,Small_Vehicle)
            # initialize list of lists 
            data = [["Building",building],["Aircraft",aircraft],["Helicopter",helicopter],["Large-Vehicle",large_vehicle],["Small-Vehicle",Small_Vehicle],["Ship",Ship]] 
            df = pd.DataFrame(data, columns = ['Object Name', 'Object Count'])
            df.index = df.index + 1
            # saving dataframe to csv format
            df.to_csv("C:\\Users\\Shivam\\Pictures\\Photos\\split_csv_file/csv_file"+str(name_of_file)+".csv",index_label="S.No.")
            df = pd.read_csv('C:\\Users\\Shivam\\Pictures\\Photos\\split_csv_file/csv_file'+str(name_of_file)+'.csv',index_col=0)
            f = open('C:\\Users\\Shivam\\Pictures\\Photos\\split_html_file/csv_file'+str(name_of_file)+'.html','w')
            a = df.to_html()
            f.write(a)
            f.close()
            print(df)
            config = pdfkit.configuration(wkhtmltopdf='C://Users//Shivam//anaconda3//envs//gputest//Lib//site-packages//wkhtmltopdf//wkhtmltopdf.exe')

            pdfkit.from_file('C:\\Users\\Shivam\\Pictures\\Photos\\split_html_file/csv_file'+str(name_of_file)+'.html', 'C:\\Users\\Shivam\\Pictures\\Photos\\split_pdf_file/csv_file'+str(name_of_file)+'.pdf',configuration=config)



            keyboard = np.zeros((180, 913, 3), np.uint8)

            keys_set_1 = {0: "Object Name", 1: "Building", 2: "Aircraft", 3: "Helicopter", 4:str(helicopter),
                        5: "Object Count", 6: str(building), 7:str(aircraft),8:"Large-Vehicle",9:"Ship",10:"Small-Vehicle",11:str(large_vehicle),
                        12:str(Ship),13:str(Small_Vehicle) }

            def letter(letter_index, text, letter_light):
                # Keys
                if letter_index == 0:
                    x = 0
                    y = 0
                elif letter_index == 1:
                    x = 152
                    y = 0
                elif letter_index == 2:
                    x = 304
                    y = 0
                elif letter_index == 3:
                    x = 456
                    y = 0
                elif letter_index == 4:
                    x = 456
                    y = 60
                elif letter_index == 5:
                    x = 0
                    y = 60
                elif letter_index == 6:
                    x = 152
                    y = 60
                elif letter_index == 7:
                    x = 304
                    y = 60
                elif letter_index == 8:
                    x = 608
                    y = 0
                elif letter_index == 9:
                    x = 760
                    y = 0 
                elif letter_index == 10:
                    x = 912
                    y = 0     
                elif letter_index == 11:
                    x = 608
                    y = 60 
                elif letter_index == 12:
                    x = 760
                    y = 60
                elif letter_index == 13:
                    x = 912
                    y = 60     
                width = 152
                height = 60
                th = 3 # thickness
                cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)

                # Text settings
                font_letter = cv2.FONT_HERSHEY_SIMPLEX 
                font_scale = 0.5
                font_th = 1
                text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
                width_text, height_text = text_size[0], text_size[1]
                text_x = int((width - width_text) / 2) + x
                text_y = int((height + height_text) / 2) + y
                cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)



            for i in range(14):
                if i == 5:
                    light = True
                else:
                    light = False
                letter(i, keys_set_1[i], light)
            k=random.randint(0,500)
            path1="C:\\Users\\Shivam\\Pictures\\Photos\\table"
            cv2.imwrite(os.path.join(path1,str(name_of_file)+".jpg"), keyboard)
            table_image=cv2.imread("C:\\Users\\Shivam\\Pictures\\Photos\\table/"+str(name_of_file)+".jpg")
            detected_image=cv2.imread(str(latest_file)+"/"+str(name_of_file)+".jpg")
            # dsize
            dsize = (913, 705)

            # resize image
            detected_image = cv2.resize(detected_image, dsize,interpolation = cv2.INTER_AREA)

            vertical_Appended_Image_and_Table_Count = np.vstack((detected_image,table_image))
            path2="C:\\Users\\Shivam\\Pictures\\Photos\\image_and_table"
            cv2.imwrite(os.path.join(path2,str(name_of_file)+".jpg"), vertical_Appended_Image_and_Table_Count)

    def plot(self,list_for_time_series_data):
                
        #list_for_time_series_data=[[290, 56, 8, 2, 2, 0], [430, 65, 9, 4, 4, 0],[200, 65, 5, 10, 4, 0]]
        fig=figure(num=None, figsize=(20, 7), dpi=80, facecolor='w', edgecolor='k')
        #plot1
        name = ('building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship')
        count1 = [list_for_time_series_data[0][0],list_for_time_series_data[0][1],list_for_time_series_data[0][2],list_for_time_series_data[0][3],list_for_time_series_data[0][4],list_for_time_series_data[0][5]]
        y_pos = np.arange(len(name))

        plt.ylabel('Number Of Object')
        plt.subplot(1, 3, 1)
        plt.bar(y_pos, count1, align='center',color='b')
        for index1, value1 in enumerate(count1):
            plt.text(index1, value1, str(value1),color='r', fontweight='bold')
        plt.title('Count at time_stamp T1')
        plt.xticks(y_pos, name)
        #plot2
        name = ('building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship')
        count = [list_for_time_series_data[1][0],list_for_time_series_data[1][1],list_for_time_series_data[1][2],list_for_time_series_data[1][3],list_for_time_series_data[1][4],list_for_time_series_data[1][5]]
        y_pos = np.arange(len(name))

        plt.ylabel('Number Of Object')
        plt.subplot(1, 3, 2)
        plt.bar(y_pos, count, align='center',color='y')
        for index, value in enumerate(count):
            plt.text(index, value, str(value),color='r', fontweight='bold')
        plt.title('Count at time_stamp T2')
        plt.xticks(y_pos, name)
        #plot3
        name = ('building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship')
        count2 = [list_for_time_series_data[2][0],list_for_time_series_data[2][1],list_for_time_series_data[2][2],list_for_time_series_data[2][3],list_for_time_series_data[2][4],list_for_time_series_data[2][5]]
        y_pos = np.arange(len(name))

        plt.ylabel('Number Of Object')
        plt.subplot(1, 3, 3)
        plt.bar(y_pos, count2, align='center',color='y')
        for index2, value2 in enumerate(count2):
            plt.text(index2, value2, str(value2),color='r', fontweight='bold')    
        plt.title('Count at time_stamp T3')
        plt.xticks(y_pos, name)    
        #plt.show()
        fig.savefig('C:\\Users\\Shivam\\Pictures\\object_based_change_detection\\object_based_change_detection\\OBCD_API\\result\\plot.png')
    
    def csv_and_table(self,list_for_time_series_data):
        df = pd.DataFrame(list_for_time_series_data, columns = ['building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship'])
        df.index = df.index + 1
        df.to_csv("C:\\Users\\Shivam\\Pictures\\object_based_change_detection\\object_based_change_detection\\OBCD_API\\result\\time_series_object_based_change_detection.csv",index_label="T")
        array=np.transpose(list_for_time_series_data)
        list1 = array.tolist()
        dfObj = pd.DataFrame(list1, columns = ["T1","T2","T3"], index=['building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship'])
        print(dfObj)
        dfObj.to_csv("C:\\Users\\Shivam\\Pictures\\object_based_change_detection\\object_based_change_detection\\OBCD_API\\result\\object_based_change_detection.csv")
        df_styled = dfObj.style.background_gradient()
        dfi.export(df_styled,"C:\\Users\\Shivam\\Pictures\\object_based_change_detection\\object_based_change_detection\\OBCD_API\\result\\mytable.png")
        dict={}
        helping_list=[]
        
        total_number_of_loops=len(list_for_time_series_data)
        for i in range(total_number_of_loops):
            dict[i]="T"+str(i+1)
            i=i+1
        
        number_of_loops=len(list_for_time_series_data)
        resultant=[]
        time_stamps=[]
        for j in range(number_of_loops):
            if(j==0):
                
                res_list = [list_for_time_series_data[j-1][i] - list_for_time_series_data[j][i] for i in range(len(list_for_time_series_data[j]))]
                print(list_for_time_series_data[j-1])
                resultant.append(res_list)
                time_stamps.append(["Changes_Between_Time_Stamps",dict[number_of_loops-1],dict[j]])
                
            else:    
                
                res_list = [list_for_time_series_data[j][i] - list_for_time_series_data[j-1][i] for i in range(len(list_for_time_series_data[j]))]
                resultant.append(res_list)
                time_stamps.append(["Changes_Between_Time_Stamps",dict[j],dict[j-1]])
        print(resultant)
        print(time_stamps)
        for i in time_stamps:
            helping_list.append(' '.join(i))
        dfObj = pd.DataFrame(resultant, columns = ['building', 'aircraft', 'helicopter', 'L_vehicle','S_vehicle','ship'],index=helping_list)
        print(dfObj)
        df_styled = dfObj.style.background_gradient()
        dfi.export(df_styled,"C:\\Users\\Shivam\\Pictures\\object_based_change_detection\\object_based_change_detection\\OBCD_API\\result\\mytable12.png")