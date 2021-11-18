import os
import glob
import struct
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv2
print('opencv version:', cv2.__version__)


# 半角カタカナとひらがな辞書
han_to_zen_dict = {
    'ｱ':'あ',
    'ｲ':'い',
    'ｳ':'う',
    'ｴ':'え',
    'ｵ':'お',
    'ｶ':'か',
    'ｷ':'き',
    'ｸ':'く',
    'ｹ':'け',
    'ｺ':'こ',
    'ｻ':'さ',
    'ｼ':'し',
    'ｽ':'す',
    'ｾ':'せ',
    'ｿ':'そ',
    'ﾀ':'た',
    'ﾁ':'ち',
    'ﾂ':'つ',
    'ﾃ':'て',
    'ﾄ':'と',
    'ﾅ':'な',
    'ﾆ':'に',
    'ﾇ':'ぬ',
    'ﾈ':'ね',
    'ﾉ':'の',
    'ﾊ':'は',
    'ﾋ':'ひ',
    'ﾌ':'ふ',
    'ﾍ':'へ',
    'ﾎ':'ほ',
    'ﾏ':'ま',
    'ﾐ':'み',
    'ﾑ':'む',
    'ﾒ':'め',
    'ﾓ':'も',
    'ﾔ':'や',
    'ﾕ':'ゆ',
    'ﾖ':'よ',
    'ﾗ':'ら',
    'ﾘ':'り',
    'ﾙ':'る',
    'ﾚ':'れ',
    'ﾛ':'ろ',
    'ﾜ':'わ',
    'ｦ':'を',
    'ﾝ':'ん',
    }

def jis0201_to_char(JIS0201_PATH):
    """
    jis0201とcharの辞書生成

    Parameters
    ----------
    JIS0201_PATH : string
        下記ファイルのパス
        http://ash.jp/ash/src/codetbl/jis0201.txt

    Returns
    -------
    JIS0201_char_dict : dict
        JIS0201とCharのdict E.g. '0xb1': 'あ'
    """
    JIS0201_char_dict = {}
    with open(JIS0201_PATH, 'r') as f:
        for row in f:
            if row[0] != '#':
                rows = row.split("\t")
                unicode_char = chr(int(rows[1], 16))

                if unicode_char in han_to_zen_dict:
                    hiragana = han_to_zen_dict[chr(int(rows[1], 16))]
                    JIS0201_char_dict[rows[0].lower()] = hiragana
                else:
                    JIS0201_char_dict[rows[0].lower()] = unicode_char
    
    return JIS0201_char_dict

def jis0208_to_char(JIS0208_PATH):
    """
    jis0208とcharの辞書生成

    Parameters
    ----------
    JIS0208_PATH : string
        下記ファイルのパス
        http://ash.jp/ash/src/codetbl/jis0208.txt

    Returns
    -------
    JIS0208_char_dict : dict
        JIS0208とCharのdict E.g. '0xb1': 'あ'
    """
    JIS0208_char_dict = {}
    with open(JIS0208_PATH, 'r') as f:
        for row in f:
            if row[0] != '#':
                rows = row.split("\t")
                unicode_char = chr(int(rows[2], 16))
                JIS0208_char_dict[rows[1].lower()] = unicode_char
    return JIS0208_char_dict


def convert_m_type(target_dir, JIS0201_char_dict):
    """
    ETL1, 6, 7のバイナリデータ(M-Type)をPNGに変換します
    M-Type : http://etlcdb.db.aist.go.jp/etlcdb/etln/form_m.htm 

    Parameters
    ----------
    target_dir : string
        バイナリデータがある親ディレクトリ. (E.g. ~/ETL7/)
    JIS0201_char_dict: dict
        JIS0201とCharのdict E.g. '0xb1': 'あ'

    Returns
    -------
    img_path_list : list
        saveしたimgのフルパスのリスト. E.g. [[img_full_path, unicode char e.g.'あ']...]
    
    """

    allFiles = glob.glob(target_dir + '*')
    targetFiles = [path for path in allFiles if os.path.isfile(path)]

    img_path_list = []

    # 変換. binary -> png.
    for etlfile in targetFiles:
        if 'INFO' in etlfile: # infoにはpngが含まれていないため.
            continue

        # 出力先ディレクトリ作成.
        dir_name = os.path.basename(etlfile)
        img_dir = target_dir + dir_name + '_IMG'
        os.makedirs(img_dir, exist_ok=True)
        print('create output image folder:', img_dir)

        # convert m-type data.
        RECORD_SIZE = 2052
        X_SIZE = 64
        Y_SIZE = 63
        DATA_NUM_POSITION = 0
        SERIAL_SHEETS_NUM_POSTION = 2
        JIS_CODE_POSITION = 3 # JIS Code (JIS X 0201)
        IMAGE_POSITION = 18
        Y_POSITION = 14
        X_POSITION = 15

        f = open(etlfile, 'rb')
        f.seek(0)

        while True:
            s = f.read(RECORD_SIZE)

            # stream完了でbreak.
            if not s:
                break

            # byte unpack
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            # read code jis.
            code_jis = r[JIS_CODE_POSITION]

            # JIS0201からunicode文字を取得.
            unicode_char = JIS0201_char_dict[ hex(code_jis) ]

            # JIS_CODE毎の出力ディレクトリ作成.
            jis_path = img_dir + '/' + str(code_jis)
            os.makedirs(jis_path, exist_ok=True)
            # imgのファイル名作成.
            fn = "{0:02x}_{1:02x}_{2:04x}.png".format(code_jis, r[DATA_NUM_POSITION], r[SERIAL_SHEETS_NUM_POSTION])
            image_full_path = jis_path + "/" + fn

            # when already exist image, skip save. 
            if os.path.exists(image_full_path) == False:
                # read image.
                iF = Image.frombytes('F', (X_SIZE, Y_SIZE), r[IMAGE_POSITION], 'bit', 4)
                iP = iF.convert('L')

                # save image.
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)
                iE.save(image_full_path)

            img_path_list.append([image_full_path, unicode_char])

    return img_path_list

def convert_b_type(target_dir, JIS0208_char_dict):
    """
    ETL8B, 9のバイナリデータ(B-Type)をPNGに変換します
    B-Type : http://etlcdb.db.aist.go.jp/specification-of-etl-8

    Parameters
    ----------
    target_dir : string
        バイナリデータがある親ディレクトリ. (E.g. ~/ETL8B/)
    JIS0208_char_dict: dict
        JIS0208とCharのdict E.g. '0xb1': 'あ'

    Returns
    -------
    img_path_list : list
        saveしたimgのフルパスのリスト. E.g. [[img_full_path, unicode char e.g.'あ']...]
    
    """
    allFiles = glob.glob(target_dir + '*')
    targetFiles = [path for path in allFiles if os.path.isfile(path)]
    #print(targetFiles)

    img_path_list = []

    # 変換. binary -> png.
    for etlfile in targetFiles:
        if 'INFO' in etlfile: # infoにはpngが含まれていないため.
            continue

        # 出力先ディレクトリ作成.
        dir_name = os.path.basename(etlfile)
        img_dir = target_dir + dir_name + '_IMG'
        os.makedirs(img_dir, exist_ok=True)
        print('create output image folder:', img_dir)

        # convert m-type data.
        RECORD_SIZE = 512
        X_SIZE = 64
        Y_SIZE = 63
        SERIAL_SHEETS_NUM_POSTION = 0
        JIS_CODE_POSITION = 1 # JIS Code (JIS X 0208)
        IMAGE_POSITION = 3

        f = open(etlfile, 'rb')
        f.seek(0)

        while True:
            s = f.read(RECORD_SIZE)

            # stream完了でbreak.
            if not s:
                break

            # byte unpack
            r = struct.unpack('>2H4s504s', s)

            # read code jis.
            code_jis = r[JIS_CODE_POSITION]
            if code_jis == 8224: # this is dummy file.
                continue

            # JIS0208からunicode文字を取得.
            unicode_char = JIS0208_char_dict[ hex(code_jis) ]

            # JIS_CODE毎の出力ディレクトリ作成.
            jis_path = img_dir + '/' + str(code_jis)
            os.makedirs(jis_path, exist_ok=True)
            # imgのファイル名作成.
            fn = "{0:02x}_{1:04x}.png".format(code_jis, r[SERIAL_SHEETS_NUM_POSTION])
            image_full_path = jis_path + "/" + fn

            # when already exist image, skip save. 
            if os.path.exists(image_full_path) == False:
                # read image.
                iF = Image.frombytes('1', (X_SIZE, Y_SIZE), r[IMAGE_POSITION], 'raw')
                iP = iF.convert('L')

                # save image.
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)
                iE.save(image_full_path)

            img_path_list.append([image_full_path, unicode_char])

    return img_path_list

def convertCVtoPIL(cv_image):
    ''' 
    OpenCV型 -> PIL型 
    '''
    new_image = cv_image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

# convert PIL to OpenCV
def convertPILtoCV(pil_image):
    ''' 
    PIL型 -> OpenCV型
    
    Parameters
    ----------
    pil_image : obj

    Returns
    -------
    cv image : np.array
    '''
    new_image = np.array(pil_image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    #return np.asarray(pil_image)
    return new_image

def create_contour(img_path):
    """
    opencvを利用し、外接矩形を抽出します.
    この関数は、ETL文字データベースの黒画像に白文字で1文字のみ文字が存在することを前提としています.

    Parameters
    ----------
    img_path : string
        画像データのフルパス E.g.[[img_full_path, unicode char e.g.'あ']...]

    Returns
    -------
    output : list
        img_pathに外接矩形の座標を付与したリスト E.g. [[img_path, [leftTop_x, leftTop_y, width, height], unicode char e.g. 'あ']...]
    
    """
    # load image.
    img_cv = cv2.imread(img_path[0], cv2.IMREAD_COLOR)
    # get image size.
    height, width, channels = img_cv.shape
    image_size = height * width

    # 入力画像の中には、画像の端まで文字がかかれているものがある
    # 余白がないとうまく文字の外接矩形を切り出せないため黒画素の画像を追加し余白を生成する.
    BLANK_SIZE = 1
    height_blank_array = np.zeros((height, BLANK_SIZE, 3), np.uint8)
    width_blank_array = np.zeros((1, width+BLANK_SIZE*2, 3), np.uint8) # width+BLANK_SIZE*2は先に幅1の画像を両端に連結するため.
    width_blank_pil = Image.fromarray(np.uint8(width_blank_array))
    height_blank_pil = Image.fromarray(np.uint8(height_blank_array))
    # 左右に連結.
    concat_img_cv = cv2.hconcat([convertPILtoCV(height_blank_pil), img_cv])
    concat_img_cv = cv2.hconcat([concat_img_cv, convertPILtoCV(height_blank_pil)])
    # 上下に連結.
    concat_img_cv = cv2.vconcat([convertPILtoCV(width_blank_pil), concat_img_cv])
    concat_img_cv = cv2.vconcat([concat_img_cv, convertPILtoCV(width_blank_pil)])

    # to gray scale.
    img_gray = cv2.cvtColor(concat_img_cv, cv2.COLOR_RGB2GRAY)

    # 2値化.
    # ref http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    ret,img_thresh = cv2.threshold(img_gray,155,255,cv2.THRESH_BINARY_INV)

    # 輪郭の取得.
    contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # 画像全体を占めるareaを除去.
        if image_size * 0.99 < area:
            contours.pop(index)
            continue
        # 画像サイズの0.3%以下はノイズとして除去.
        # 64*63の画像中に12以下areaでノイズとして判定
        if area < image_size * 0.003:
            contours.pop(index)

    # 文字の外接矩形を取得.
    # inputの画像に文字は1文字のみという前提.
    # そのため、最小のx,y座標, 最大のx,y座標を抽出し、矩形を作り文字の外接矩形とする.
    x_list = []
    y_list = []

    for index, contour in enumerate(contours):
        for coordinate in contour:
            x_list.append(coordinate[0][0])
            y_list.append(coordinate[0][1])

    # 追加したBLANK_SIZE分座標を調整.
    if len(x_list) != 0 and len(y_list): 
        min_x = min(x_list) - BLANK_SIZE
        if min_x < 0:
            min_x = 0
        min_y = min(y_list) - BLANK_SIZE
        if min_y < 0:
            min_y = 0
        max_x = max(x_list) - BLANK_SIZE
        max_y = max(y_list) - BLANK_SIZE
        rec_width = max_x - min_x
        if width < rec_width:
            rec_width = width
        rec_height = max_y - min_y
        if height < rec_height:
            rec_height = height
    else:
        print('this file was not get contours:', img_path[0])
        min_x, min_y, rec_width, rec_height = 0, 0, 0, 0

    return [img_path[0], [min_x,min_y,rec_width,rec_height], img_path[1]]

def showCVimage(cv_image, description=''):
    """
    show openCV image as PIL image.
    """
    if len(description) != 0:
        print(description)
    array = np.asarray(convertCVtoPIL(cv_image))

    _, axes = plt.subplots(1, 1,figsize=(10,10))
    #_, axes = plt.subplots(dpi=300)

    axes.imshow(array)
    axes.set_facecolor('m')
    plt.show()
    
    del array  

def showRectangleImage(img_path, xywh, name):
    """
    画像に矩形をplotして出力.
    """
    cv_image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    rec_cv_image = cv2.rectangle(
        cv_image,
        (xywh[0][0], xywh[0][1]),
        (xywh[0][0]+xywh[0][2], xywh[0][1]+xywh[0][3]), (255, 0, 0), 1
    )
    showCVimage(rec_cv_image, name)