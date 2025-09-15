import cv2
import os
import glob

def extract_and_save_contours(img_file, output_dir):
    img_name = os.path.basename(img_file)
    img = cv2.imread(img_file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # 提取轮廓
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    draw_img = img.copy()
    # 轮廓近似
    contours_approx = []
    for cnt in contours:
        epsilon = 0.012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contours_approx.append(approx)
    # 绘制轮廓
    res = cv2.drawContours(draw_img, contours_approx, -1, (0, 255, 0), 3)
    # 保存文件
    img_clean_name = os.path.splitext(img_name)[0]
    thresh_path = os.path.join(output_dir, img_clean_name + '_thresh.png')
    res_path = os.path.join(output_dir, img_clean_name + '_res.png')
    cv2.imwrite(thresh_path, thresh_img)
    cv2.imwrite(res_path, res)

    return contours_approx, res_path

def process_all_images(target_name:str, image_dir:str = None, output_dir:str = None, ):

    if image_dir is None:
        image_dir = os.path.join(os.getcwd(), 'output','mask_predictions')
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(),'output','final_output')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    file_name_list = glob.glob(os.path.join(image_dir,'*'))
    img_file_list = [f for f in file_name_list if os.path.isfile(f) and f.lower().endswith(image_extensions)]
    if not img_file_list:
        print('No image files found')
        return

    matched_files = [ f for f in img_file_list if os.path.basename(f) == target_name ]
    if matched_files:
        img_file = matched_files[0]  # ✅ 只保留第一个匹配的文件
    else:
        print(f'No file named "{target_name}" found.')
        return

    contours, res_path = extract_and_save_contours(img_file, output_dir)
    return contours, res_path

if __name__ == '__main__':
    process_all_images('u2net')