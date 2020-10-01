from utils.text_detection import TextDetection
from utils.deskew_method import deskew
from keras.models import load_model
import cv2
import os

## contour prediction
model_name = 'contour_classify_vgg.h5'
print("loading keras model")
classify_model = load_model(model_name)

def model_extract(img_path):
    deskew_res = deskew(img_path)
    res = False
    for type in ["uncropped"]:
        detect_text_res = TextDetection(img=deskew_res[type], path=img_path, cls_model=classify_model, debug=False)
        img = detect_text_res.oriented_orig_img
        [h, w] = img.shape[:2]
        print("height", h, "width", w)
        im_out = img.copy()
        lines = detect_text_res.lines
        color_1 = [0, 0, 255]
        color_2 = [0, 255, 0]
        color_3 = [255, 0, 0]
        curr_color = color_1
        for l in lines:
            if curr_color == color_1:
                curr_color = color_2
            elif curr_color == color_2:
                curr_color = color_3
            else:
                curr_color = color_1
            for block in l:
                [x,y,w,h] = block["pts"]
                cv2.rectangle(im_out, (x, y), (x + w, y + h), curr_color, 2)
        cv2.imwrite(os.path.join("res_out",os.path.basename(img_path)),im_out)

    return res

if __name__=="__main__":
    folder = "error-samples"
    tot = len(os.listdir(folder))
    # for im in ['0897EP02_02_19_2020_EOD_SB.jpeg']:
    for idx,im in enumerate(os.listdir(folder)):
        print("Processing ",idx, "of",tot)
        model_extract(os.path.join(folder,im))