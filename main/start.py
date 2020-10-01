from utils.text_detection import TextDetection
import cv2
import os
from main.OcrModel_10 import OcrModel, DecoderType
from main.DataLoader import DataLoader
from utils.deskew_method import deskew
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import regex

class FilePaths:
    ocr_model_file = "ocr_saved_model/ocr_model_writer_1"
    "filenames and paths to data"
    fnCharList = ocr_model_file+'/charList.txt'
    fnAccuracy = ocr_model_file+'/accuracy.txt'
    checkpoint_ctpn = "checkpoints_ctpn/"

keras_graph_1 = tf.get_default_graph()
keras_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
keras_session = tf.Session(config=config)
K.set_session(keras_session)

ocr_graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
decoderType = DecoderType.BestPath
ocr_session = tf.Session(graph=ocr_graph,config=config)
ocr_model = OcrModel(open(FilePaths.fnCharList).read(), FilePaths.ocr_model_file,ocr_graph, ocr_session, decoderType, mustRestore=True)
print("New session created")

## contour prediction
model_name = 'contour_classify_vgg.h5'
print("loading keras model")
classify_model = load_model(model_name)

def contains_word(key_wrd, text,mismatches_allowed=False,end=False):
    chars_len = len([c for c in list(key_wrd) if c.isalpha()])
    if not mismatches_allowed:
        if chars_len <= 3:
            mismatches_allowed = 0
        elif chars_len < 7:
            mismatches_allowed = 1
        else:
            mismatches_allowed = 2
    if end:
        reg_str = r"(" + key_wrd + r"$){e<=" + str(mismatches_allowed) + "}"
    else:
        reg_str = r"(" + key_wrd + r"){e<=" + str(mismatches_allowed) + "}"
    if not end:
        search_out = regex.search(reg_str, " "+text+" ")
    else:
        search_out = regex.search(reg_str, " " + text)
    if search_out:
        return True, search_out
    return False, 0

def contains_key_words(text, mismatches=[],end=False):
    text = " "+text.replace("\n"," ").lower()+" "
    key_wrds = ["dispensed","machine","counter","cassette"," date "," increase ","decrease"," out ","rejected"," time ",
                "remaining"," total "," left "," disp "," tot "," str "," inc "," dec "," ic "," dc "," oc ","record",
                " pay"," card","amount"," bal "," bank "," type ","cleared"]
    for idx,key in enumerate(key_wrds):
        if len(mismatches) > 0:
            mis_allowed = mismatches[idx]
        else:
            mis_allowed = False
        if isinstance(key,str):
            word_search = contains_word(key, text,mis_allowed,end)
            if word_search[0]:
                print("Found match",word_search[1],"key--",key)
                return word_search[1]
        else:
            full_word_search = contains_word(" ".join(key),text,mis_allowed,end)
            if full_word_search[0]:
                contains_all_words = True
                for word in key:
                    if not contains_word(word,full_word_search[1].group(),mis_allowed,end)[0]:
                        contains_all_words = False
                        break
                if contains_all_words:
                    print("Found match", full_word_search[1], "key--", key)
                    return full_word_search[1]
    return False

def process_pts(coords):
    '''

    :param coords:
    :return: change format from [x,y,w,h] to [x1,y1,x2,y2]
    '''
    [x,y,w,h] = coords
    return [x,y,x+w,y+h]


def model_extract(img_path):
    deskew_res = deskew(img_path)
    res = False
    for type in ["uncropped","uncropped_flip","cropped","cropped_flip"]:
        detect_text_res = TextDetection(img=deskew_res[type],path=img_path,cls_model=classify_model,debug=False)
        img = detect_text_res.oriented_orig_img
        [h,w] = img.shape[:2]
        print("height",h, "width", w)
        im_out = img.copy()
        lines = detect_text_res.lines
        text_blocks = []
        texts_obj = {}
        color_1 = [0, 0, 255]
        color_2 = [0, 255, 0]
        color_3 = [255, 0, 0]
        curr_color = color_1
        for line in lines:
            if curr_color == color_1:
                curr_color = color_2
            elif curr_color == color_2:
                curr_color = color_3
            else:
                curr_color = color_1
            for block in line:
                # id = block["id"]
                [x,y,w,h] = block["pts"]
                cv2.rectangle(im_out,(x,y),(x+w,y+h),curr_color,2)
                text_blocks.append(block)
        cv2.imwrite(os.path.join("res_out",os.path.basename(img_path)),im_out)

        loader = DataLoader(text_blocks, 50, ocr_model.imgSize, img)
        print("Total samples", len(loader.samples))
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            print('Batch:', iterInfo[0], '/', iterInfo[1])
            batch = loader.getNext()
            (recognized, _) = ocr_model.inferBatch(batch=batch)
            # print(recognized)
            for id, text in zip(batch.ids, recognized):
                texts_obj[id] = text

        for line_idx, line in enumerate(lines):
            for block_idx, block in enumerate(line):
                id = block["id"]
                lines[line_idx][block_idx]["text"] = texts_obj[id]
                lines[line_idx][block_idx]["pts"] = process_pts(lines[line_idx][block_idx]["pts"])
        # lines = detect_text_res.merge_close_texts()
        detect_text_res.lines = lines
        
        full_text = ""
        for line in detect_text_res.lines:
            line_txt = ""
            if curr_color == color_1:
                curr_color = color_2
            elif curr_color == color_2:
                curr_color = color_3
            else:
                curr_color = color_1
            for block in line:
                [x1, y1, x2, y2] = block["pts"]
                # cv2.rectangle(im_out, (x1, y1), (x2, y2), curr_color, 2)
                line_txt = line_txt + " " + block["text"]
                print("Text--", line_txt, "pts--",block["pts"] )
            print('---------------------')
            full_text = full_text + line_txt + "\n"
        print("File--", os.path.basename(img_path))
        if contains_key_words(full_text):
            print("Found Res in", type)
            cv2.imwrite(os.path.join('res_out', os.path.basename(img_path)), im_out)
            return detect_text_res
        elif type=="uncropped":
            res = detect_text_res
    return res

if __name__=="__main__":
    folder = "error-samples"
    for im in ['S1AC0141502_02_18_2020_EOD_SA.jpeg']:
    # for im in os.listdir(folder):
        im_p = os.path.join(folder,im)
        model_extract(im_p)


