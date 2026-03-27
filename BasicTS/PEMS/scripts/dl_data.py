import gzip
import shutil
import glob
import os

folder = "../d03_station_5min"
output_folder = "/data/yuzhang_fei/PEMS/PEMSD3_2025"  # 指定输出文件夹

os.makedirs(output_folder, exist_ok=True)

for gz_file in glob.glob(os.path.join(folder, "*.txt.gz")):
    filename = os.path.basename(gz_file)[:-3]  # 获取文件名并去掉 .gz
    txt_file = os.path.join(output_folder, filename)  # 输出到指定文件夹
    
    with gzip.open(gz_file, 'rb') as f_in:
        with open(txt_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # os.remove(gz_file)  # 删除原压缩文件（可选）
    print(f"解压: {gz_file} -> {txt_file}")