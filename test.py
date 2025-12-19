import librosa
import noisereduce as nr
import soundfile as sf
import torch
from pydub import AudioSegment
input_path = 'original.wav'
output_path = 'tests/sample/enroll.wav'
def denoise_audio():
    # 1. 加载音频 (强制转为单声道，并重采样为作业推荐的 24000Hz)
    y, sr = librosa.load(input_path, sr=24000, mono=True)

    # 2. 执行降噪
    # prop_decrease=1.0 表示完全消除检测到的噪声，0.8 左右声音会更自然
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    # 3. 移除首尾静音（让模型更专注于有效人声）
    yt, _ = librosa.effects.trim(reduced_noise, top_db=20)

    # 4. 归一化音量（防止声音太小）
    yt = librosa.util.normalize(yt)

    # 5. 保存到作业要求的目录
    sf.write(output_path, yt, sr)
    print(f"✨ 降噪完成！文件已保存至: {output_path}")

def cut_start():
    file_path = 'tests/sample/enroll.wav'
    y, sr = librosa.load(file_path, sr=None)
    cut_samples = int(0.4 * sr)
    y_new = y[cut_samples:]
    sf.write(file_path, y_new, sr)
    print(f"成功切除前0.5秒！当前采样率: {sr}Hz, 时长: {len(y_new)/sr:.2f}s")
def cut_end():
    file_path = 'tests/sample/enroll.wav'
    y, sr = librosa.load(file_path, sr=None)
    cut_samples = int(1 * sr)
    y_new = y[:-cut_samples]
    sf.write(file_path, y_new, sr)
    print(f"成功切除！当前采样率: {sr}Hz, 时长: {len(y_new)/sr:.2f}s")
hash_pretrained_dict = {
    "dc3c97e17592963677a4a1681f30c653": ["v2", "v2", False],  # s2G488k.pth#sovits_v1_pretrained
    "43797be674a37c1c83ee81081941ed0f": ["v2", "v3", False],  # s2Gv3.pth#sovits_v3_pretrained
    "6642b37f3dbb1f76882b69937c95a5f3": ["v2", "v2", False],  # s2G2333K.pth#sovits_v2_pretrained
    "4f26b9476d0c5033e04162c486074374": ["v2", "v4", False],  # s2Gv4.pth#sovits_v4_pretrained
    "c7e9fce2223f3db685cdfa1e6368728a": ["v2", "v2Pro", False],  # s2Gv2Pro.pth#sovits_v2Pro_pretrained
    "66b313e39455b57ab1b0bc0b239c9d0a": ["v2", "v2ProPlus", False],  # s2Gv2ProPlus.pth#sovits_v2ProPlus_pretrained
}
import hashlib
def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()
head2version = {
    b"00": ["v1", "v1", False],
    b"01": ["v2", "v2", False],
    b"02": ["v2", "v3", False],
    b"03": ["v2", "v3", True],
    b"04": ["v2", "v4", True],
    b"05": ["v2", "v2Pro", False],
    b"06": ["v2", "v2ProPlus", False],
}
import os
def get_sovits_version_from_path_fast(sovits_path):
    ###1-if it is pretrained sovits models, by hash
    hash = get_hash_from_file(sovits_path)
    if hash in hash_pretrained_dict:
        return hash_pretrained_dict[hash]
    ###2-new weights, by head
    with open(sovits_path, "rb") as f:
        version = f.read(2)
    if version != b"PK":
        return head2version[version]
    ###3-old weights, by file size
    if_lora_v3 = False
    size = os.path.getsize(sovits_path)
    """
            v1weights:about 82942KB
                half thr:82978KB
            v2weights:about 83014KB
            v3weights:about 750MB
    """
    if size < 82978 * 1024:
        model_version = version = "v1"
    elif size < 700 * 1024 * 1024:
        model_version = version = "v2"
    else:
        version = "v2"
        model_version = "v3"
    return version, model_version, if_lora_v3
print(get_sovits_version_from_path_fast("./model/SoVITS_G.pth"))
denoise_audio()
