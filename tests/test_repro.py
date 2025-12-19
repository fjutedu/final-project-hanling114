import os
import subprocess

def test_can_run():
    # 1. 检查必要的文件是否存在
    assert os.path.exists("student_generate.py"), "错误：找不到 student_generate.py"
    assert os.path.exists("evaluate.py"), "错误：找不到 evaluate.py"
    # 2. 检查 student_generate.py 是否能正常解析参数（即使不实际合成）
    # 运行 python student_generate.py --help，如果返回 0 说明脚本没语法错误
    result = subprocess.run(
        ["python", "student_generate.py", "--help"], 
        capture_output=True, 
        text=True
    )
    assert result.returncode == 0, f"脚本运行失败，错误信息: {result.stderr}"

    print("基础文件结构与命令行解析检查通过！")