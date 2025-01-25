import netCDF4 as nc
import numpy as np
import os
from datetime import datetime

def process_NC_file(file_path):
    """
    处理单个 .NC 文件，提取指定变量值，返回一个字典。
    """
    dataset = nc.Dataset(file_path)

    # 提取变量值
    variables_to_extract = ['u', 'v', 't', 'rh', 'msl', 'ps', 'pcp']
    data = {}

    # 提取每个变量的数据
    for var in variables_to_extract:
        values = dataset.variables[var][0, 0, :, :]  # 取第一个record和第一个层
        fill_value = dataset.variables[var]._FillValue  # 获取填充值
        values = np.where(values == fill_value, np.nan, values)  # 替换填充值为NaN
        data[var] = values
    
    dataset.close()
    return data

def process_all_NC_files(root_dir):
    """
    遍历 root 目录下所有 .NC 文件，提取数据并保存到一个合并的 N*(228,296,7) 的 tensor 中。
    """
    all_data = []  # 存储所有文件的处理结果

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".NC"):
                file_path = os.path.join(root, file)
                print(f"正在处理文件: {file_path}")
                try:
                    file_data = process_NC_file(file_path)
                    all_data.append(file_data)
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")

    # 将所有数据组合到一个张量中
    N = len(all_data)
    x_dim, y_dim = 296, 228
    tensor = np.empty((N, y_dim, x_dim, 7))

    # 按顺序存储变量数据到张量中
    for i, data in enumerate(all_data):
        tensor[i, :, :, 0] = data['u'][:y_dim, :x_dim]  # 选择前228（y_dim）行和前296（x_dim）列
        tensor[i, :, :, 1] = data['v'][:y_dim, :x_dim]
        tensor[i, :, :, 2] = data['t'][:y_dim, :x_dim]
        tensor[i, :, :, 3] = data['rh'][:y_dim, :x_dim]
        tensor[i, :, :, 4] = data['msl'][:y_dim, :x_dim]
        tensor[i, :, :, 5] = data['ps'][:y_dim, :x_dim]
        tensor[i, :, :, 6] = data['pcp'][:y_dim, :x_dim]

    return tensor

def process_nc_file(file_path):
    """
    处理单个 .nc 文件，提取指定变量值并返回字典。
    """
    dataset = nc.Dataset(file_path)

    # 提取变量值
    variables_to_extract = ['u', 'v', 't', 'rh', 'slp', 'ps', 'tp']
    data = {}
    # 提取每个变量的数据
    for var in variables_to_extract:
        if var in dataset.variables:
            values = dataset.variables[var][:]
            # 处理变量的维度
            if values.ndim == 2:  
                data[var] = values
            elif values.ndim == 3:
                data[var] = values[0, :, :]  # 取第一层或第一个时间点的数据
            else:
                data[var] = np.nan  # 如果维度异常，填充NaN

            # 替换填充值为 NaN
            fill_value = getattr(dataset.variables[var], '_FillValue', None)
            if fill_value is not None:
                data[var] = np.where(data[var] == fill_value, np.nan, data[var])
        else:
            # 如果变量不存在，填充 NaN
            data[var] = np.nan * np.ones((231, 297))  # 使用合理默认值
    dataset.close()
    return data

def process_all_nc_files(root_dir):
    """
    遍历 root 目录下所有 .nc 文件，提取数据并保存到一个合并的 N*(228,296,7) 的张量中。
    """
    all_data = []  # 存储所有文件的处理结果

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # 确保是文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹下的所有 .nc 文件
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.nc'):  # 只处理 .nc 文件
                    file_path = os.path.join(folder_path, file_name)
                    print(f"正在处理文件: {file_path}")
                    try:
                        file_data = process_nc_file(file_path)
                        all_data.append(file_data)
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

    # 将所有数据组合到一个张量中
    N = len(all_data)
    y_dim, x_dim = 228, 296
    tensor = np.empty((N, y_dim, x_dim, 7))

    # 按顺序存储变量数据到张量中
    for i, data in enumerate(all_data):
        # 对每个变量进行裁剪或插值到目标形状
        for j, var in enumerate(['u', 'v', 't', 'rh', 'slp', 'ps', 'tp']):
            if isinstance(data[var], np.ndarray) and data[var].ndim == 2:
                # 对二维数据进行裁剪，保证形状为 (228, 296)
                tensor[i, :, :, j] = data[var][:y_dim, :x_dim]
            else:
                tensor[i, :, :, j] = np.nan * np.ones((y_dim, x_dim))  # 填充NaN

    return tensor

def main():
    root_dir_NC = "day0226_0318"  
    root_dir_nc = "day0412_1220"  

    # 处理 .NC 文件
    tensor_NC = process_all_NC_files(root_dir_NC)

    # 处理 .nc 文件
    tensor_nc = process_all_nc_files(root_dir_nc)

    # 合并两个张量
    combined_tensor = np.concatenate((tensor_NC, tensor_nc), axis=0)

    # 保存合并后的张量
    np.save('combined_tensor.npy', combined_tensor)
    print(f"合并后的张量已保存为 'combined_tensor.npy'")
    print(f"生成的合并张量形状：{combined_tensor.shape}")

if __name__ == "__main__":
    main()