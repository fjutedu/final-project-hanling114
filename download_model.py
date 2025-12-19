import os
import hashlib
import requests
MODELS_CONFIG = {
    "model": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        "path": "checkpoints/zh_model.onnx",
        "sha256": "9929917bf8cabb26fd528ea44d3a6699c11e87317a14765312420be230be0f3d"
    },
    "config": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
        "path": "checkpoints/zh_model.onnx.json",
        "sha256": "d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b"
    }
}
def get_file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取，防止大文件占用过多内存
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
def download():
    os.makedirs("checkpoints", exist_ok=True)
    
    for name, info in MODELS_CONFIG.items():
        url = info["url"]
        path = info["path"]
        expected_hash = info["sha256"]
        if os.path.exists(path):
            current_hash = get_file_sha256(path)
            if current_hash == expected_hash:
                print(f" {name} 模型已存在且校验通过。")
                continue
            else:
                print(f" {name} 校验不一致，准备重新下载...")
                os.remove(path)

        print(f"正在下载 {name}: {url}...")
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 3. 下载后立即再次校验
            final_hash = get_file_sha256(path)
            if final_hash == expected_hash:
                print(f"{name} 下载成功并完成校验。")
            else:
                print(f"{name} 下载后的校验失败！")
        except Exception as e:
            print(f"下载过程中出错: {e}")

if __name__ == "__main__":
    download()