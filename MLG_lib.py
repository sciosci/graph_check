import cv2
import pytesseract
import pandas as pd
from PIL import Image
import numpy as np
import shlex, subprocess
import shutil, os
from os import listdir
import sys,math
import PIL
import cv2
import numpy as np
from joblib import Parallel, delayed

def prepare_training_data(source_dir_list,target_dir):
    counter = 0
    for source in source_dir_list:
        print(source)
        file_names = listdir(source)
        if counter == 0:
            for file_name in file_names:
                shutil.copy(source+file_name, target_dir+'Mis_'+file_name)
            counter += 1
        elif counter == 1:
            for file_name in file_names:
                shutil.copy(source+file_name, target_dir+'Nor_'+file_name)
            counter += 1
        elif counter == 2:
            for file_name in file_names:
                shutil.copy(source+file_name, target_dir+'Other_'+file_name)

def check_digit(input):
    try:
        float(input)
        return True
    except:
        return False

def feature_extract_training(labeled_data, file):
    y = -1
    if file[:4] == 'Mis_':
        y = 1
    elif file[:4] == 'Nor_':
        y = 0
    else:
        y = 2


    try:
        tb = pd.read_csv(labeled_data + file[:-4] + '-pred1-texts.csv')
    except:
        return

    y_axis_value = tb[tb['type'] == 'y-axis-label']['text'].tolist()

    y_axis_pos_y = tb[tb['type'] == 'y-axis-label']['y'].tolist()
    y_axis_pos_y = [float(yy) for index, yy in enumerate(y_axis_pos_y) if check_digit(y_axis_value[index])]

    y_axis_value = [float(yy) for yy in y_axis_value if check_digit(yy)]

    y_axis_pairs = zip(y_axis_pos_y, y_axis_value)
    res = sorted(y_axis_pairs, key=lambda x: x[0], reverse=True)

    sorted(y_axis_pos_y)
    steps = []
    for index, yy in enumerate(y_axis_pos_y):
        if index == len(y_axis_pos_y) - 1:
            break
        step = y_axis_pos_y[index + 1] - yy
        steps.append(step)

    if len(steps) > 1:
        average_step = np.mean(steps)
    else:
        return
    print(average_step)
    increase_rates = []
    for index, yy in enumerate(y_axis_value):
        if index == len(y_axis_value) - 1:
            break
        increase_rate = (y_axis_value[index + 1] - yy) / average_step
        increase_rates.append(increase_rate)

    average_increase_rate = np.mean(increase_rates)

    x_axis_pos_y = tb[tb['type'] == 'x-axis-label']['y'].tolist()
    x_axis_pos_y = [float(xx) for xx in x_axis_pos_y]

    average_x_axis_pos_y = np.mean(x_axis_pos_y)

    if np.std(x_axis_pos_y) > 2:
        # print("double x axes")
        x_axis_pos_y = [float(xx) for xx in x_axis_pos_y if xx > average_x_axis_pos_y]
    average_x_axis_pos_y = np.mean(x_axis_pos_y)

    inference_flag_1 = 0
    min_y_1 = 0
    # print(res)
    # print(average_step)
    if len(res) == 0:
        return []
    if abs(res[0][0] - average_x_axis_pos_y) > abs(average_step / 2) and check_digit(res[1][1]) and res[-1][
        0] < average_x_axis_pos_y:
        # inference
        # print('inference one')
        inference_flag_1 = 1
        # print(res[0][1])
        # print(average_increase_rate)
        # print(abs(res[0][0] - average_x_axis_pos_y))
        min_y_1 = res[0][1] - average_increase_rate * (res[0][0] - average_x_axis_pos_y)

    inference_flag_2 = 0
    if not check_digit(res[0][1]) and check_digit(res[1][1]) and inference_flag_1 == 0:
        # inference
        # print('inference two')
        inference_flag_2 = 1
        min_y_1 = res[1][1] - average_increase_rate * (res[1][0] - x_axis_pos_y)

    increase_rate_std = np.std(increase_rates)
    inference_flag = 0
    if inference_flag_1 == 0 and inference_flag_2 == 0:
        min_y = res[0][1]
    else:
        inference_flag = 1
        print('assign')
        print(inference_flag_1)
        print(min_y_1)
        min_y = min_y_1

    if math.isnan(min_y):
        print(len(res))
        min_y = 0

    if np.isnan(increase_rate_std):
        increase_rate_std = 0

    if np.isnan(average_x_axis_pos_y):
        return []
    return [y, [inference_flag, (res[0][0] - average_x_axis_pos_y), min_y, increase_rate_std]]

def get_features_trainining(img_list,labeled_data):
    X = []
    y = []
    files = []
    for i in range(len(img_list)):
        res = feature_extract_training(labeled_data,img_list[i])
        if res != None and len(res) == 2 and len(res[1])==4:
            X.append(res[1])
            y.append(res[0])
            files.append(img_list[i])

    return X,y,files


def validation_prediction(labeled_data, file, y, annotated_graphs, target_dir):
    outcome = ''
    if file in listdir(annotated_graphs + 'Misleading\\') and y == 1:
        outcome = 'TP'
        shutil.copy(labeled_data + file, target_dir + 'TP\\' + file)
        shutil.copy(labeled_data + file[:-4]+'-texts-all.csv', target_dir + 'TP\\' + file[:-4]+'-texts-all.csv')
    elif file in listdir(annotated_graphs + 'Normal\\') or file in listdir(annotated_graphs + 'Other\\') and y == 1:
        outcome = 'FP'
        shutil.copy(labeled_data + file, target_dir + 'FP\\' + file)
        shutil.copy(labeled_data + file[:-4]+'-texts-all.csv', target_dir + 'FP\\' + file[:-4]+'-texts-all.csv')
    elif file in listdir(annotated_graphs + 'Normal\\') or file in listdir(annotated_graphs + 'Other\\') and y == 0:
        outcome = 'TN'
        shutil.copy(labeled_data + file, target_dir + 'TN\\' + file)
        shutil.copy(labeled_data + file[:-4]+'-texts-all.csv', target_dir + 'TN\\' + file[:-4]+'-texts-all.csv')
    elif file in listdir(annotated_graphs + 'Misleading\\') and y == 0:
        outcome = 'FN'
        shutil.copy(labeled_data + file, target_dir + 'FN\\' + file)
        shutil.copy(labeled_data + file[:-4]+'-texts-all.csv', target_dir + 'FN\\' + file[:-4]+'-texts-all.csv')


def feature_extract_test(labeled_data, file):

    try:
        tb = pd.read_csv(labeled_data + file[:-4] + '-pred1-texts.csv')
    except:
        return

    y_axis_value = tb[tb['type'] == 'y-axis-label']['text'].tolist()

    y_axis_pos_y = tb[tb['type'] == 'y-axis-label']['y'].tolist()
    y_axis_pos_y = [float(yy) for index, yy in enumerate(y_axis_pos_y) if check_digit(y_axis_value[index])]

    y_axis_value = [float(yy) for yy in y_axis_value if check_digit(yy)]

    y_axis_pairs = zip(y_axis_pos_y, y_axis_value)
    res = sorted(y_axis_pairs, key=lambda x: x[0], reverse=True)

    sorted(y_axis_pos_y)
    steps = []
    for index, yy in enumerate(y_axis_pos_y):
        if index == len(y_axis_pos_y) - 1:
            break
        step = y_axis_pos_y[index + 1] - yy
        steps.append(step)

    if len(steps) > 1:
        average_step = np.mean(steps)
    else:
        return
    print(average_step)
    increase_rates = []
    for index, yy in enumerate(y_axis_value):
        if index == len(y_axis_value) - 1:
            break
        increase_rate = (y_axis_value[index + 1] - yy) / average_step
        increase_rates.append(increase_rate)

    average_increase_rate = np.mean(increase_rates)

    x_axis_pos_y = tb[tb['type'] == 'x-axis-label']['y'].tolist()
    x_axis_pos_y = [float(xx) for xx in x_axis_pos_y]

    average_x_axis_pos_y = np.mean(x_axis_pos_y)

    if np.std(x_axis_pos_y) > 2:
        x_axis_pos_y = [float(xx) for xx in x_axis_pos_y if xx > average_x_axis_pos_y]
    average_x_axis_pos_y = np.mean(x_axis_pos_y)

    inference_flag_1 = 0
    min_y_1 = 0

        return []
    if abs(res[0][0] - average_x_axis_pos_y) > abs(average_step / 2) and check_digit(res[1][1]) and res[-1][
        0] < average_x_axis_pos_y:
        # inference
        inference_flag_1 = 1
        min_y_1 = res[0][1] - average_increase_rate * (res[0][0] - average_x_axis_pos_y)

    inference_flag_2 = 0
    if not check_digit(res[0][1]) and check_digit(res[1][1]) and inference_flag_1 == 0:
        # inference
        inference_flag_2 = 1
        min_y_1 = res[1][1] - average_increase_rate * (res[1][0] - x_axis_pos_y)

    increase_rate_std = np.std(increase_rates)
    inference_flag = 0
    if inference_flag_1 == 0 and inference_flag_2 == 0:
        min_y = res[0][1]
    else:
        inference_flag = 1
        print('assign')
        print(inference_flag_1)
        print(min_y_1)
        min_y = min_y_1

    if math.isnan(min_y):
        print(len(res))
        min_y = 0

    if np.isnan(increase_rate_std):
        increase_rate_std = 0

    if np.isnan(average_x_axis_pos_y):
        return []
    return [inference_flag, (res[0][0] - average_x_axis_pos_y), min_y, increase_rate_std]

def get_features_test(img_list,labeled_data):
    X = []
    files = []
    for i in range(len(img_list)):
        res = feature_extract_training(labeled_data,img_list[i])
        if res != None and len(res) == 2 and len(res[1])==4:
            X.append(res)

            files.append(img_list[i])

    return X,files

def predict(random_forest,X_test,threshold,my_class):

    predicted_proba = random_forest.predict_proba(X_test)
    predicted = (predicted_proba[:, my_class] >= threshold).astype('int')

    return predicted



def darknet_line_extrat(line):
    terms = line.split(' ')
    term_type = terms[0][:-1]
    pos = [0] * 6
    index = 0
    for tt in terms:
        if index == 0:
            if tt.split(':')[0] == 'Enter' or len(tt.split(':')[0]) == 0 :
                return -1

            pos[0] = tt.split(':')[0]
            index+=1
        if tt == '':
            continue
        if '%' in tt:
            pos[index] = int(tt.split('%')[0])
        if 'left_x' in tt:
            index = 2
        try:
            float(tt)
            pos[index] = int(tt)
            index += 1
        except:
            continue
    return pos

#https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def flood_fill_single(im_path):

    # Read image
    im_in = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE);

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    cv2.imwrite(im_path[:-4]+'_flood.png', im_out)

    return

#https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
def check_fonts(im_path):
    src = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    font_list = []

    # The second cell is the label matrix
    labels = output[1]

    # The third cell is the stat matrix
    stats = output[2]

    h_max = max([st[cv2.CC_STAT_HEIGHT] for st in stats])
    w_max = max([st[cv2.CC_STAT_WIDTH] for st in stats])

    for i, st in enumerate(stats):
        h = st[cv2.CC_STAT_HEIGHT]
        w = st[cv2.CC_STAT_WIDTH]
        # remove dots in the figure
        if h < h_max/15 and w < w_max/15:
            continue
        if h >=w:
            font_list.append(h)
        else:
            font_list.append(w)

    return sorted(font_list)[:-1]



def text_recognition(file, box_list):
    lst = box_list

    file_name = file.split('\\')[-1]
    try:
        os.mkdir('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\')
    except:
        return -1

    im = Image.open(open(file, "rb"))

    l_list = []
    font_list = []
    for index, box in enumerate(lst[1]):

        im1 = im.crop((lst[1][index] - 3, lst[2][index] - 3, lst[1][index] + lst[3][index] + 3,
                       lst[2][index] + lst[4][index] + 3))
        im1.save('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp.png', 'PNG')
        img = cv2.imread('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp.png')
        if lst[7][index] == 'Number':
            flood_fill_single('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp.png')
            font_list = check_fonts('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp_flood.png')
            df = pytesseract.image_to_data(img, lang='digitsall_layer', \
                                           config='--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',
                                           output_type='data.frame')
            if len(font_list)>=3:
                lst[8][index] = np.std(font_list)
            else:
                lst[8][index] = 0

        else:
            df = pytesseract.image_to_data(img, lang='eng', \
                                           config='--psm 8 --oem 1 ', output_type='data.frame')

            lst[8][index] = 0

        string = df.sort_values(by=['conf'])['text'].iloc[-1]
        conf = max(df['conf'].tolist())
        lst[5][index] = string
        lst[9][index] = conf


    return lst

#https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
def letter_counter(im_path):
    src = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    font_list = []

    # The second cell is the label matrix
    labels = output[1]

    # The third cell is the stat matrix
    stats = output[2]


    return len(stats)-1

def enlarge_corp(img_path,ratio):
    img = Image.open(img_path)
    width, height = img.size
    img = img.resize((width*ratio, height*ratio), PIL.Image.ANTIALIAS)
    img.save(img_path[:-4]+str(ratio)+'.png')



# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def ocr_preprocessing(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    thresh = thresholding(img)

    cv2.imwrite(img_path[:-4]+'_prepro'+'.png', thresh)



def text_recognition_validate(file, box_list):
    lst = box_list
    file_name = file.split('/')[-1]
    tmp_path = '/home/thuang12/Desktop/Misleading_Graph/data/tmp/'
    try:
        #os.mkdir('tmp/' + file_name + '/')
        os.mkdir(tmp_path + file_name + '/')
    except:
        return -1
    im = Image.open(open(file, "rb"))
    l_list = []
    font_list = []
    for index, box in enumerate(lst[1]):
        im1 = im.crop((lst[1][index] - 3, lst[2][index] - 3, lst[1][index] + lst[3][index] + 3,
                       lst[2][index] + lst[4][index] + 3))
        #im1.save('tmp/' + file_name + '/tmp.png', 'PNG')
        im1.save(tmp_path + file_name + '/tmp.png', 'PNG')
        #img = cv2.imread('tmp/' + file_name + '/tmp.png')
        img = cv2.imread(tmp_path + file_name + '/tmp.png')
        #ocr_preprocessing('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp.png')
        #flood_fill_single('tmp/' + file_name + '/tmp.png')
        flood_fill_single(tmp_path + file_name + '/tmp.png')
        #font_list = check_fonts('tmp/' + file_name + '/tmp_flood.png')
        font_list = check_fonts(tmp_path + file_name + '/tmp_flood.png')
        #letter_count = letter_counter('tmp/' + file_name + '/tmp_flood.png')
        letter_count = letter_counter(tmp_path + file_name + '/tmp_flood.png')
        if len(font_list)>=3:
            #print(font_list)
            lst[8][index] = np.std(font_list)
        else:
            lst[8][index] = 0
        #df1 = pytesseract.image_to_data(img, lang='digitsall_layer', \
        #                               config='--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',
        #                               output_type='data.frame')
        #pil_img = Image.open('tmp/' + file_name + '/tmp.png')
        pil_img = Image.open(tmp_path + file_name + '/tmp.png')
        rotated_pil_img = pil_img.rotate(270,expand=True)
        #rotated_pil_img.save('tmp/' + file_name + '/tmp2.png')
        rotated_pil_img.save(tmp_path + file_name + '/tmp2.png')
        pil_img.close()
        #img2 = cv2.imread('tmp/' + file_name + '/tmp2.png')
        img2 = cv2.imread(tmp_path + file_name + '/tmp2.png')
        df1 = pytesseract.image_to_data(img, lang='engrestrict_best', \
                                           config='--psm 8 --oem 1 ', output_type='data.frame')
        #df2 = pytesseract.image_to_data(img, lang='digits_comma', \
        #                                   config='--psm 8 --oem 1 ', output_type='data.frame')
        df2 = pytesseract.image_to_data(img2, lang='engrestrict_best', \
                                           config='--psm 8 --oem 1 ', output_type='data.frame')
        #img2 = cv2.imread('D:\\Misleading_Graph\\data\\tmp\\' + file_name + '\\tmp_prepro.png')
        #df3 = pytesseract.image_to_data(img2, lang='digitsall_layer', \
        #                                config='--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789',
        #                                output_type='data.frame')
        #df4 = pytesseract.image_to_data(img2, lang='eng1', \
        #                                config='--psm 8 --oem 1 ', output_type='data.frame')
        df = pd.concat([df1,df2])
        df = df.sort_values(by=['conf'],ascending=False)
        my_str = ''
        conf = 0
        #my_str = df['text'].iloc[0]
        #conf = df['conf'].iloc[0]
        distance = 999
        print('here')
        print(letter_count)
        for ind, row in df.iterrows():
            if row['conf'] == -1:
                continue
            t_str = str(row['text'])
            if t_str[-1] == '.' or t_str[-1] == '|':
                t_str = t_str[:-1]
            if t_str.split('.')[-1] == '0':
                t_str = str(int(float(t_str)))
            if t_str == 'o':
                t_str = str(0)
            print(t_str)
            if abs(len(str(t_str))-letter_count) <distance and len(t_str)>len(my_str):
                my_str = t_str
                conf = row['conf']
                distance = abs(len(str(t_str))-letter_count)
        print(my_str)
        #print(letter_count)
        lst[5][index] = my_str
        lst[9][index] = conf
    #print(lst[5])
    return lst


def parse_line_text(text_list, target_dir):
    file = text_list[0]
    file_name = file.split('/')[-1]

    lst = text_list[1]

    lst = text_recognition(file, lst)
    print(lst)
    if lst == -1:
        return

    df = pd.DataFrame(zip(lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6]),
                      columns=['id', 'x', 'y', 'width', 'height', 'text', 'type'])
    df.to_csv(target_dir + file_name[:-4] + '-texts.csv', index=False)
    lst2 = text_list[2]

    df = pd.DataFrame(zip(lst[0], lst2[1],lst[9],lst[8]), columns=['id', 'prob', 'conf', 'font'])
    df.to_csv(target_dir + file_name[:-4] + '-texts-prob.csv', index=False)

    shutil.copy(file, target_dir + file_name)
    print('done')


def parse_line_text_validate(text_list,org_img_dir, target_dir):
    file = text_list[0]
    file_name = file.split('/')[-1]
    file = org_img_dir+file_name
    print(file)
    lst = text_list[1]
	
    lst = text_recognition_validate(file, lst)
    print(lst)
    if lst == -1:
        print('return')
        return

    df = pd.DataFrame(zip(lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6]),
                      columns=['id', 'x', 'y', 'width', 'height', 'text', 'type'])
    df.to_csv(target_dir + file_name[:-4] + '-texts.csv', index=False)
    lst2 = text_list[2]

    df = pd.DataFrame(zip(lst[0], lst2[1],lst[9],lst[8]), columns=['id', 'prob', 'conf', 'font'])
    df.to_csv(target_dir + file_name[:-4] + '-texts-prob.csv', index=False)

    shutil.copy(file, target_dir + file_name)
    print('done')


def parse_lines_text_generated(annotation_dir,content):

    counter = 0

    text_list = []


    for line in content:


            text_id = []
            text_type = []
            text_prob = []
            text_x = []
            text_y = []
            text_width = []
            text_height = []
            text_content = []
            text_class = []
            text_font = []
            text_conf = []

            try:
                table = pd.read_csv(annotation_dir+line[:-4]+'_labels.csv')
                text_list.append([line])
            except:
                continue

            for index,row in table.iterrows():
                text_id.append(index)
                text_prob.append(1)
                text_x.append(row['x'])
                text_y.append(row['y'])
                text_width.append(row['width'])
                text_height.append(row['height'])
                text_content.append(row['text'])
                text_type.append('')
                text_class.append('')
                text_font.append(0)
                text_conf.append(1)

            lst = [text_id, text_x, text_y, text_width, text_height, text_content, text_type, text_class, text_font, text_conf]
            text_list[-1].append(lst)
            lst = [text_id, text_prob]
            text_list[-1].append(lst)


    print(counter)
    return text_list

def parse_lines_text(content):
    flag = 0
    file = ''
    counter = 0

    text_list = []
    text_id = []
    text_type = []
    text_prob = []
    text_x = []
    text_y = []
    text_width = []
    text_height = []
    text_content = []
    text_class = []
    text_font = []
    text_conf = []

    file_name_list = []

    for line in content:

        if 'Predicted in' in line:
            flag = 1

            t_id = 0
            if counter == 0:
                file = line.split(': ')[0]
                file_name = file.split('\\')[-1]

            if counter > 0:
                text_list.append([file])

                lst = [text_id, text_x, text_y, text_width, text_height, text_content, text_type, text_class, text_font, text_conf]
                text_list[-1].append(lst)
                lst = [text_id, text_prob]
                text_list[-1].append(lst)

                text_id = []
                text_type = []
                text_prob = []
                text_x = []
                text_y = []
                text_width = []
                text_height = []
                text_content = []
                text_class = []
                text_font = []
                text_conf = []

            counter += 1

            file = line.split(': ')[0]
            file_name = file.split('\\')[-1]


            continue

        if flag == 1:
            im = Image.open(open(file, "rb"))
            info = darknet_line_extrat(line[:-2])

            if info == -1:
                continue

            text_id.append(t_id)
            text_prob.append(info[1])
            text_x.append(info[2])
            text_y.append(info[3])
            text_width.append(info[4])
            text_height.append(info[5])
            text_content.append('')
            text_type.append('')
            text_class.append(info[0])
            text_font.append(0)
            text_conf.append(0)
            t_id += 1

    print(counter)
    return text_list


def subplot_crop(file, box_list, target_dir):
    lst = box_list

    file_name = file.split('/')[-1]

    #print(file)
    try:
        im = Image.open(open(file, "rb"))
    except:
        return

    l_list = []
    for index, box in enumerate(lst[1]):
        im1 = im.crop((lst[1][index] - 15, lst[2][index] - 15, lst[1][index] + lst[3][index] + 15,
                       lst[2][index] + lst[4][index] + 15))
        #im1.save(target_dir + file_name[:-4] + '_'+str(index)+'.png', 'PNG')
        im1.convert('RGB').save(target_dir + file_name[:-4] + '_'+str(index)+'.png', "PNG", optimize=True)

    return

def parse_line_subplot(plot_list, target_dir):
    file = plot_list[0]
    #print(file)
    file_name = file.split('/')[-1]

    lst = plot_list[1]

    lst = subplot_crop(file, lst,target_dir)




def parse_lines_subplot(content):
    flag = 0
    file = ''
    counter = 0

    subplot_list = []
    subplot_id = []
    subplot_type = []
    subplot_prob = []
    subplot_x = []
    subplot_y = []
    subplot_width = []
    subplot_height = []
    subplot_class = []

    file_name_list = []

    for line in content:

        if 'Predicted in' in line:
            flag = 1

            t_id = 0
            if counter == 0:
                file = line.split(': ')[0]
                file_name = file.split('\\')[-1]

            if counter > 0:
                subplot_list.append([file])

                lst = [subplot_id, subplot_x, subplot_y, subplot_width, subplot_height, subplot_class]
                subplot_list[-1].append(lst)
                lst = [subplot_id, subplot_prob]
                subplot_list[-1].append(lst)

                subplot_id = []
                subplot_prob = []
                subplot_x = []
                subplot_y = []
                subplot_width = []
                subplot_height = []
                subplot_class = []

            counter += 1

            file = line.split(': ')[0]
            file_name = file.split('\\')[-1]


            continue

        if flag == 1:

            try:
                im = Image.open(open(file, "rb"))
            except:
                continue

            info = darknet_line_extrat(line[:-2])

            if info == -1:
                continue

            subplot_id.append(t_id)
            subplot_prob.append(info[1])
            subplot_x.append(info[2])
            subplot_y.append(info[3])
            subplot_width.append(info[4])
            subplot_height.append(info[5])
            subplot_class.append(info[0])
            t_id += 1

    print(counter)
    return subplot_list


def get_role(file_name,data_path,labeled_data):
    if file_name[-4:] != '.png':
        return

    t2 = pd.read_csv(labeled_data + file_name[:-4] + '-texts-prob.csv')
    if t2.shape[0] <2:
        return

    #command_line = 'powershell.exe wsl python /home/hzhuang/rev/run_text_role_classifier.py single ' + data_path + file_name
    command_line = 'python /home/thuang12/Desktop/Misleading_Graph/rev/scripts/run_text_role_classifier.py single '+ data_path + file_name
    args = command_line.split(' ')
    p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(p.communicate())
    t1 = pd.read_csv(labeled_data+file_name[:-4]+'-pred1-texts.csv')
    #t2 = pd.read_csv(labeled_data+file_name[:-4]+'-texts-prob.csv')
    t_all = t1.merge(t2,on='id')
    t_all.to_csv(labeled_data+file_name[:-4]+'-texts-all.csv')

def get_role_v(file_name,data_path,labeled_data):
    if file_name[-4:] != '.png' or file_name[-10:] == '_check.png':
        return

    #t2 = pd.read_csv(labeled_data + file_name[:-4] + '-texts-prob.csv')
    #if t2.shape[0] <2:
    #    return

    command_line = 'powershell.exe wsl python /home/hzhuang/rev/run_text_role_classifier.py single ' + data_path + file_name
    args = command_line.split(' ')
    p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(p.communicate())
    #t1 = pd.read_csv(labeled_data+file_name[:-4]+'-pred1-texts.csv')
    #t2 = pd.read_csv(labeled_data+file_name[:-4]+'-texts-prob.csv')
    #t_all = t1.merge(t2,on='id')
    #t_all.to_csv(labeled_data+file_name[:-4]+'-texts-all.csv')


def get_prob_dic(probability_dir):
    #Compound_Prob
    f1 = open(probability_dir + 'Comp.txt', "r")
    lines_1 = f1.readlines()
    f1.close()
    f2 = open(probability_dir + 'Subplot.txt', "r")
    lines_2 = f2.readlines
    f2.close()
    f3 = open(probability_dir + '3D.txt', "w")
    lines_3 = f3.readlines
    f3.close()

    comp_dic = {}

    for line in lines_1:
        comp_dic[line.split(',')[0]] = line.split(',')[1]

    subp_dic = {}

    for line in lines_2:
        subp_dic[line.split(',')[0]] = line.split(',')[1]

    d3_dic = {}

    for line in lines_3:
        d3_dic[line.split(',')[0]] = line.split(',')[1]

    return comp_dic, subp_dic, d3_dic


def get_prob_dic(file, comp_dic, subp_dic, d3_dic):
    comp_prob = -1
    subp_prob = -1
    d3_prob = -1
    if '_' in file:
        comp_prob = comp_dic[file.split('_')[0] + '.png']
        subp_prob = subp_dic[file]
    else:
        comp_prob = com_dic[file]
        subp_prob = -1

    d3_prob = d3_dic[file]

    return comp_prob, subp_prob, d3_prob

