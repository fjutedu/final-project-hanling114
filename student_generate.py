import argparse
import os
import sys

import soundfile as sf
rel_path = "./FFmpeg/bin"
abs_path = os.path.abspath(rel_path)
os.environ["PATH"] += os.pathsep + abs_path
try:
    import jieba_fast
except ImportError:
    import jieba
    sys.modules['jieba_fast'] = jieba

now_dir = os.getcwd()
gpt_path = os.path.join(now_dir, "model", "GPT.ckpt")
sovits_path = os.path.join(now_dir, "model", "SoVITS_G.pth")
source_code_path = os.path.join(now_dir, "GPT_SoVITS")

sys.path.insert(0, source_code_path)
sys.path.insert(0, os.path.join(source_code_path, "GPT_SoVITS"))
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir, "GPT-SoVITS"))

# 导入核心推理函数
# 示例：确保在推理前执行了类似的操作（具体函数名随版本可能不同）
#from GPT_SoVITS.inference_webui import get_tts_wav, change_gpt_weights, change_sovits_weights
from get_tts_wav import get_tts_wav
#change_gpt_weights(gpt_path)
#change_sovits_weights(sovits_path)
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Inference")
    parser.add_argument("--enroll", type=str, required=True, help="参考音频路径")
    parser.add_argument("--text", type=str, required=True, help="包含目标文本的txt文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出音频路径")
    args = parser.parse_args()
    
    
    # 3. 初始化加载权重
    print("正在加载模型权重...")

    # 4. 读取目标文本内容
    with open(args.text, 'r', encoding='utf-8') as f:
        target_text = f.read().strip()

    # 5. 参考音频的文本（Prompt Text）
    # 改进方向：如果助教没给参考音频的文字，这里可以写死一段文字
    # 或者如果你安装了 funasr，可以调用 ASR 识别。这里先设为默认值。
    # 注意：这个文字必须尽量匹配 enroll.wav 里的内容，否则效果会差。
    prompt_text = "在人工智能技术飞速发展的今天，语音合成技术已经能够让我们以假乱真地还原任何人的音色。" 
    
    print(f"开始合成语音，目标文本：{target_text[:20]}...")

    # 6. 调用合成引擎
    # 参数说明：输入文本, 文本语言, 参考音频, 参考文本, 参考语言, 切分策略等
    # 返回值：一个生成器，产出 (采样率, 采样点数组)
    try:
        # get_tts_wav 是一个生成器，我们取最后一次产生的结果
        results = get_tts_wav(
            ref_wav_path=args.enroll,
            prompt_text=prompt_text,
            prompt_language="中文",
            text=target_text,
            text_language="中文",
            how_to_cut="按标点符号切分", # 改进：增加长文本稳定性
            top_k=20, # 参数微调：控制随机性
            top_p=0.6,
            temperature=0.6
        )
        
        # 迭代生成器获取最终音频数据
        for sampling_rate, audio_data in results:
            # 7. 保存结果
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            sf.write(args.out, audio_data, sampling_rate)
            
        print(f"合成成功！音频已保存至: {args.out}")
        
    except Exception as e:
        print(f"合成过程中出错: {str(e)}")

if __name__ == "__main__":
    main()