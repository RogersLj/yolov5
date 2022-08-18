# yolov5



# 用自己的数据集训练yolov5

- 代码准备
- 准备数据
- 将xml标注文件转换成yolov5训练格式txt，并划分训练集
- 修改训练配置并训练
- 进行推测



## github下载yolov5代码

```shell
git clone https://github.com/ultralytics/yolov5
```

下载依赖文件

```shell
pip install -r yolov5/requirements.txt
```



### yolov5的标注格式为txt

![image-20220810091943292](https://cdn.jsdelivr.net/gh/RogersLj/Image@master/uPic/image-20220810091943292.png)

![image-20220810091955487](https://cdn.jsdelivr.net/gh/RogersLj/Image@master/uPic/image-20220810091955487.png)



## txt文件的每一行为一个bounding box

图中有三个目标，两个人和一个领带

- 每一行为一个目标

- 每一行对应的数值表示为：
  - 类别	
  - x中心	
  - y中心	
  - 宽	
  - 高

- 坐标为0-1的归一化数值
- 类别从0开始



### xml标注文件

```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000100.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>321781133</flickrid>
	</source>
	<owner>
		<flickrid>RailroadGuy</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>432</width>
		<height>256</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>train</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>3</xmin>
			<ymin>47</ymin>
			<xmax>432</xmax>
			<ymax>222</ymax>
		</bndbox>
	</object>
</annotation>

```



- filename

  文件名

- size
  - 宽width
  - 高height
  - 深度depth

- object

  - name

  - bndbox

    - xmin

    - ymin
    
    - xmax
    
    - ymax





---

# 数据准备



```python
import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()

    info_dict = {}
    info_dict['bboxes'] = []

    for elem in root:
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict['image_size'] = tuple(image_size)

        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)

    return info_dict


class_name_to_id_mapping = {"fire": 0}

def convert_to_yolov5(info_dict):
    print_buffer = []

    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KetError:
            print("Invalid class: Must be fire")

        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])

        image_w, image_h, image_c = info_dict["image_size"] 
        b_center_x /= image_w 
        b_center_y /= image_h
        b_width    /= image_w 
        b_height   /= image_h 

        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        save_file_name = os.path.join("Annotations", info_dict["filename"].replace("jpg", "txt"))
        print("\n".join(print_buffer), file= open(save_file_name, "w"))
        


Annotations = [os.path.join('Annotations', x) for x in os.listdir('Annotations') if x[-3:] == "xml"]
Annotations.sort()

for ann in tqdm(Annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)


random.seed(108)

# Read JPEGImages and Annotations
JPEGImages = [os.path.join('JPEGImages', x) for x in os.listdir('JPEGImages')]
Annotations = [os.path.join('Annotations', x) for x in os.listdir('Annotations') if x[-3:] == "txt"]

JPEGImages.sort()
Annotations.sort()

# Split the dataset into train-valid-test splits 
train_JPEGImages, val_JPEGImages, train_Annotations, val_Annotations = train_test_split(JPEGImages, Annotations, test_size = 0.2, random_state = 1)
val_JPEGImages, test_JPEGImages, val_Annotations, test_Annotations = train_test_split(val_JPEGImages, val_Annotations, test_size = 0.5, random_state = 1)


#Utility function to move JPEGImages
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_JPEGImages, 'JPEGImages/train')
move_files_to_folder(val_JPEGImages, 'JPEGImages/val/')
move_files_to_folder(test_JPEGImages, 'JPEGImages/test/')
move_files_to_folder(train_Annotations, 'Annotations/train/')
move_files_to_folder(val_Annotations, 'Annotations/val/')
move_files_to_folder(test_Annotations, 'Annotations/test/')

```



## 训练时数据存放的文件夹，图像文件应该命名为images，图像标注文件应该命名为labels



---

# 训练的选项

- `img`

  图像的大小

- `batch`

  batch size

- `epochs`

  训练的epoch

- `data`

  数据的YAML文件，包括image和labels的位置

- `workers`

  cpu核数

- `cfg`

  具体网络结构`yolo5s.yaml`, `yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`

- `weights`

  预训练权重

- `name`

  训练的数据

- `hyp`

  超参数选项



---

## 配置文件

data config file

data/fire_data.yaml



models/yolov5s.yaml



# 模型训练

python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch 32 --epochs 100 --data fire_data.yaml --weights yolov5s.pt --workers 24 --name fire_det



# 模型预测
python detect.py --source ../Fire_DataSet/images/test/ --weights runs/train/fire_det4/weights/best.pt --conf 0.25 --name fire_det
