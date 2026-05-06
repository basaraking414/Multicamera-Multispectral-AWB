import os
import shutil
import sys
from natsort import natsorted

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def find_and_copy_matching_files(folder_a, folder_b, folder_c,folder_d,folder_e):
    # 遍历文件夹A中的所有JPG文件
    jpg_files = [f for f in os.listdir(folder_a) if f.endswith('.jpg')]
    jpg_files = natsorted(jpg_files)
    dng_files = [f for f in os.listdir(folder_a) if f.endswith('.dng')]
    dng_files = natsorted(dng_files)
    bin_files = [f for f in os.listdir(folder_b) if f.endswith('.bin')]
    bin_files = natsorted(bin_files)

    for i,filename_a in enumerate(jpg_files):
        if filename_a.startswith('.trashed'):
            continue
        if filename_a.endswith('.jpg'):
            # 提取文件名的最�?个字�?
            last_eight_a = filename_a[-12:-4]  # 提取倒数8个字�?
            # 在文件夹B中查找匹配的BIN文件
            for filename_b in bin_files:
                # 提取文件名的最�?个字�?
                last_eight_b = filename_b[-12:-4]  # 提取倒数8个字�?.bin后缀
                # 比较字符串是否匹�?
                if abs(int(last_eight_a) - int(last_eight_b)) == 0:
                    # 拷贝文件并重命名
                    src_bin_path = os.path.join(folder_b, filename_b)
                    new_bin_name = f"{os.path.splitext(filename_a)[0]}.bin"  # 使用jpg文件名重命名
                    dst_bin_path = os.path.join(folder_c, new_bin_name)  # 目标路径

                    src_img_path = os.path.join(folder_a, filename_a)
                    dst_img_path = os.path.join(folder_d, filename_a)

                    src_dng_path = os.path.join(folder_a, dng_files[i])
                    new_dng_name = f"{os.path.splitext(filename_a)[0]}.dng"
                    dst_dng_path = os.path.join(folder_e, new_dng_name)

                    shutil.copy(src_bin_path, dst_bin_path)  # 拷贝并重命名
                    shutil.copy(src_img_path, dst_img_path)
                    shutil.copy(src_dng_path, dst_dng_path)

                    print(f'已将 {filename_b} 拷贝并重命名�?{new_bin_name} �?{dst_bin_path}')


if __name__ == '__main__':

    camera_folder_path = r"E:\ricky\research\oppo-project-2025\Dump data\Camera\\"
    mcs_folder_path = r"E:\ricky\research\oppo-project-2025\Dump data\MultispectralSensor\RawData\\"
    match_bin_folder_path = r'McsBin\\'
    match_camera_folder_path = r'image\\'
    match_dng_folder_path = r'image_dng\\'
    if not os.path.exists(match_bin_folder_path):
        os.makedirs(match_bin_folder_path)
    if not os.path.exists(match_camera_folder_path):
        os.makedirs(match_camera_folder_path)
    if not os.path.exists(match_dng_folder_path):
        os.makedirs(match_dng_folder_path)
    find_and_copy_matching_files(camera_folder_path, mcs_folder_path, match_bin_folder_path,match_camera_folder_path,match_dng_folder_path)
