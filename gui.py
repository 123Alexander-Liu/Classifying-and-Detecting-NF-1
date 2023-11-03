
import argparse
import logging

import sys
from PyQt5.QtWidgets import QApplication, QWidget ,QLabel,QMessageBox
 
from PyQt5 import QtWidgets, QtCore,QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.Qt import QLineEdit

from  predict_color import  get_args , inference_one
from config import UNetConfig

cfg = UNetConfig()

import torch , os
from unet import NestedUNet
from PIL import Image
import cv2
from utils.colors import get_colors

import numpy as np


def calAcc(pred, label , mask_name): 
    # How many of the predicted non-zero pixel values were correctly predicted
    colos = get_colors(n_classes=3)  #black is background, red is nf1 and white is healthy skin 0,1,2
    
    #First calculate how many non-zero pixel values were predicted
    if len(  pred.shape )==3: # prediction is three channels
        pred =cv2.cvtColor( pred , cv2.COLOR_BGRA2GRAY )
     
        
    gray = pred.copy().reshape(-1)
    gray[gray>0.5]=1
    nozero_pix_n = np.sum(  gray )
    
    #How many of the predictions were right
    label = label.reshape(-1) 
    resha_pred = pred.copy().reshape(-1)
    if "nf1" in mask_name:   
         
        resha_pred[ resha_pred ==255 ]=0
        resha_pred[ resha_pred != 0 ] = 255
    else:
        label[ label == 2 ] = 1
        
        resha_pred[ resha_pred < 255 ] = 0
    
    resha_pred[resha_pred==255] = 1
    
        
  
    redu =  label + resha_pred 
    redu[redu !=2 ]=0
    redu[redu == 2 ]=1
    
    return    np.sum( redu ) / nozero_pix_n 

def calAcc_2(pred, label , mask_name): 
    
    
    # How many of the predicted non-zero pixel values were correctly predicted
    colos = get_colors(n_classes=3) # white is skin, red is nf1 and black is background 0,1,2
    
    #First calculate how many non-zero pixel values were predicted
    if len(  pred.shape )==3: # prediction is 3 channels
        pred =cv2.cvtColor( pred , cv2.COLOR_BGRA2GRAY )
     
        
    gray = pred.copy().reshape(-1)
    gray[gray>0.5]=1
    nozero_pix_n = np.sum(  gray )
    
    #How many of the predictions were right
    label = label.reshape(-1) 
    resha_pred = pred.copy().reshape(-1)
    if "nf1" in mask_name:   
         
        resha_pred[ resha_pred ==255 ]=0
        resha_pred[ resha_pred != 0 ] = 255
    else:
        label[ label == 2 ] = 1
        
        resha_pred[ resha_pred < 255 ] = 0
    
    resha_pred[resha_pred==255] = 1
     
    redu =  np.abs( label  -  resha_pred )
   
    
    return    len( np.where( redu==0 )[0] ) / redu.size   
 


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow,self).__init__()
        self.open_btn = QtWidgets.QPushButton(self)
        self.open_btn.setObjectName("open_btn")
        self.open_btn.setText("open")
        self.open_btn.setGeometry(QtCore.QRect(10, 70, 70, 40))
        self.open_btn.clicked.connect(self.msg) #Binding Response Functions


        self.run_btn = QtWidgets.QPushButton(self)
        self.run_btn.setObjectName("open_btn")
        self.run_btn.setText("run")
        self.run_btn.setGeometry(QtCore.QRect(10, 120, 70, 40))
        self.run_btn.clicked.connect(self.msgRun) #Run button click
        
        self.textEdit= QLineEdit(self)
        self.textEdit.setObjectName("myTextEdit")
        self.textEdit.move(90, 20)
        self.textEdit.resize(1030, 40) # w,h


        self.label_0 = QLabel(self) #pathway
        self.label_0.move(15, 30)
        self.label_0.setText("image path：")    
        
        #First, set the display position of the original image and the baseline segmentation map.
        W,H = 330,330 # Image width and height
        x = 90
        self.label_1 = QLabel(self)
        self.label_1.setFixedSize(W,H)
        self.label_1.move(x, 70)
        self.label_1.setStyleSheet("QLabel{background-color:rgb(200, 200, 200);}"  )
        self.label_2 = QLabel(self)
        self.label_2.move(x+140, 420) # (x,y)
        self.label_2.setText("original image")
        
        self.label_3 = QLabel(self)
        self.label_3.setFixedSize(W,H)
        self.label_3.move(x+W+20, 70)
        self.label_3.setStyleSheet("QLabel{background-color:rgb(200, 200, 200);}"  )
        self.label_4 = QLabel(self)
        self.label_4.move(x+490, 420 ) # (x,y)
        self.label_4.setText("Groundtruth")  
        
        #Re-display the Unet segmentation graph and the Vnet segmentation graph
        x = x+W+20+W+20
        self.label_5 = QLabel(self)
        self.label_5.setFixedSize(W,H)
        self.label_5.move(x, 70)
        self.label_5.setStyleSheet("QLabel{background-color:rgb(200, 200, 200);}"  )
        self.label_6 = QLabel(self)
        self.label_6.move(x+140, 420) # (x,y)
        self.label_6.resize(200, 20)
        self.label_6.setText("Pren Acc") 
        
        self.label_7 = QLabel(self)
  
 
      
        self.label_8 = QLabel(self)
        self.label_8.move(x+140, H+490) # (x,y)
       
        
        #以下是模型初始化
        
        args = get_args()
        print(args) 
      
        self.net = eval(cfg.model)(cfg)  #initie the NestedUnet category
        logging.info("Loading model {}".format(args.model))
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(args.model, map_location=self.device)) # run the trained model file
    
        logging.info("Model loaded !")

    def warnningBox(self,strs):
        msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 
                                      strs)
        msg_box.move(450, 450)
        msg_box.exec_()    
         

    def msg(self): #点击打开的响应函数
        self.label_1.setPixmap( QtGui.QPixmap()) #1 display original picture, 3 display prediction picture
        self.label_3.setPixmap( QtGui.QPixmap())
        self.label_5.setPixmap( QtGui.QPixmap()) #5是unet,7是vnet
        self.label_7.setPixmap( QtGui.QPixmap())
        self.label_6.setText("Unet分割图 Iou")
        self.label_8.setText("Vnet分割图 Iou")
        self.label_7.repaint()
        self.label_8.repaint()
            
        file_path, filetype = QFileDialog.getOpenFileName(self,
                  "choose file",
                  "./",
                  "All Files (*);;Text Files (*.txt)")  #设置文件扩展名过滤,注意用双分号间隔
        self.textEdit.setText(file_path )
        self.file_path =file_path
        
        self.jpg_qpx = QtGui.QPixmap(file_path)
        if not self.jpg_qpx.isNull():
            jpg = self.jpg_qpx.scaled(self.label_1.width(), self.label_1.height())
            self.label_1.setPixmap(jpg)
            
             
            masks_path =  os.path.join("data/labels"  ,   os.path.basename(  file_path  ).replace("jpg" , "png")    )    
            mask_qpx = QtGui.QPixmap(masks_path)
            
            self.masks_path = None
            if "images" in file_path and not mask_qpx.isNull():
                mask = mask_qpx.scaled(self.label_3.width(), self.label_3.height())
                self.label_3.setPixmap(mask)
                self.masks_path = masks_path
            else:
                self.warnningBox('if you cannot find groundtruth mask images, please ensure the original figures to save context of data/labels！')               
        else:
            self.warnningBox('please ensure to open a image file!')
  
    #点击运行按钮响应函数    
    def msgRun(self):
        if not self.jpg_qpx.isNull(): 
            #读取到原始图像后开始做分割
            self.label_5.setPixmap( QtGui.QPixmap()) #5是unet,7是vnet
            self.label_7.setPixmap( QtGui.QPixmap())
            self.label_6.setText("Unet Accaurate:")
            self.label_8.setText("Vnet Accaurate:")
            self.label_7.repaint()
            self.label_8.repaint()
            
            #run models
            img_path = self.file_path
            img = Image.open(img_path) # read pictures
            
            mask = inference_one(net=self.net,
                             image=img,
                             device=self.device) #前向推导，
            colors = get_colors(n_classes=cfg.n_classes) #获取表示像素C个类别对应的颜色RGB值
            w, h = img.size
            img_mask = np.zeros([h, w, 3], np.uint8)
            for idx in range(0, len(mask)): #遍历C张图，对于为（x,y）初的像素值，值为1则表示输出图像对应像素点为该类别
                image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8)) # 像素值1则变为255
                array_img = np.asarray(image_idx)
                img_mask[np.where(array_img==255)] = colors[idx] 
            
            img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)  
            
            img3 = QtGui.QImage(img_mask[:], img_mask.shape[1], img_mask.shape[0], img_mask.shape[1] * 3, QtGui.QImage.Format_RGB888)
            res_pixmap = QtGui.QPixmap(img3) 

            res_pixmap = res_pixmap.scaled(self.label_5.width(), self.label_5.height())
            self.label_5.setPixmap(res_pixmap )  
            
            if self.masks_path !=None:
                gt_mask = cv2.imread( self.masks_path.replace("labels" , "masks"), 0)  
                pred_maks = img_mask
                acc_val = calAcc_2( pred_maks,  gt_mask ,  os.path.basename(self.masks_path))  
                
                str_tmp = "mask Acc:%.4f"%acc_val
                self.label_6.setText( str_tmp ) 
                self.label_6.repaint()                     
            """
            unet_seg = Unet_seg() #initialize unet segmentation tool
            res_pixmap,np_pred = unet_seg.segImg( self.file_path ) #把分割图显示在label_5
            res_pixmap = res_pixmap.scaled(self.label_5.width(), self.label_5.height())
            self.label_5.setPixmap(res_pixmap ) # unet对应label_5  
            if self.masks_path !=None:
                np_mask = cv2.imread( self.masks_path, 0)
                acc_val = calAcc(np_pred,np_mask)
           
                str_tmp = "Unet segmentation chart Acc:%.4f"%acc_val
                self.label_6.setText( str_tmp ) 
                self.label_6.repaint()      
        
            """
        
            
        else:
           self. warnningBox('PLEASE MAKE SURE OPENING THE IMAGE FILES!')
 
 
 
 
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
 
    w = MyWindow()
    w.resize(1140, 450)
    w.setFixedSize(w.width(), w.height())
    w.move(0, 0)
    w.setWindowTitle('Unet segmentation and prediction')
    w.show()
    
    sys.exit(app.exec_())
 