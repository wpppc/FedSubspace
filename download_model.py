# download_model.py
from modelscope import snapshot_download
import os

def download_model_ms(model_name, local_dir):
    """
    使用ModelScope下载模型到指定目录
    
    Args:
        model_name: ModelScope模型ID
        local_dir: 本地保存目录
    """
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"开始下载模型: {model_name}")
    print(f"保存到: {local_dir}")
    
    try:
        model_dir = snapshot_download(
            model_id=model_name,
            cache_dir=local_dir,
            revision='master'  # 默认使用master分支
        )
        print(f"✅ 模型下载完成: {model_name}")
        print(f"📁 实际保存路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

if __name__ == "__main__":
    # 下载Llama2-7B基础模型 (ModelScope上的版本)
    download_model_ms(
        "LLM-Research/llama-2-7b",  # ModelScope上的Llama模型
        "/home/wuqicen/base_models/llama2-7b"
    )