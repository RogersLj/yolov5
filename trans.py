import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

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

Annotations = [os.path.join('Annotations', x) for x in os.listdir('Annotations') if x[-3:] == "txt"]
print(Annotations)
