import json
import os
from utils.data_processor import data_process

if __name__ == "__main__":
    config_path = "configs/data_process_config.json"
    
    # 读取 JSON 配置文件
    if not os.path.exists(config_path):
        print(f"Error: 找不到配置文件 {config_path}")
    else:
        with open(config_path, 'r') as f:
            process_config = json.load(f)
            
        # 调用封装好的主处理函数
        data_process(process_config)