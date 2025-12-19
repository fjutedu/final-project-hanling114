import argparse, os, subprocess, soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 调用学生的生成脚本
    if not os.path.exists("student_generate.py"):
        raise FileNotFoundError("缺少 student_generate.py，请实现命令行入口。")

    cmd = ["python", "student_generate.py",
           "--enroll", args.enroll,
           "--text", args.text,
           "--out", args.out]
    subprocess.check_call(cmd)

    # 检查是否生成输出
    if not os.path.exists(args.out):
        raise FileNotFoundError("未生成输出音频：%s" % args.out)

    data, sr = sf.read(args.out)
    dur = len(data) / sr

    # 时长与采样率检查
    if dur < 8.0:
        raise AssertionError(f"输出语音过短 ({dur:.2f}s)，应≥8s")
    if sr not in (16000, 22050, 24000, 32000):
        raise AssertionError(f"采样率应为 16000/22050/24000 之一，当前 {sr}")

    print("✅ 基础检查通过！输出时长 %.2fs, 采样率 %d" % (dur, sr))

if __name__ == "__main__":
    main()
