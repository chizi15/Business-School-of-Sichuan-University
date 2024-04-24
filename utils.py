import subprocess
import os


os.environ["PYTHONIOENCODING"] = "UTF-8"
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "requirements.txt")
dependency_path = os.path.join(current_dir, "dependencies")
mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 导出当前环境的依赖库
# with open(file="requirements.txt", mode="w") as f:
#     subprocess.check_call(["pip", "list", "--format=freeze"], stdout=f)
# print("Requirements file created successfully!")

subprocess.check_call(["python", "-m", "pip", "install", "--upgrade", "pip", "-i", mirror], shell=True)

# 按requirements.txt下载依赖库
# subprocess.check_call(["pip", "download", "-r", file_path, "-d", dependency_path, "-i", mirror], shell=True)

# 安装依赖库
try:
    for filename in os.listdir(dependency_path):
        if filename.endswith(".whl") or filename.endswith(".tar.gz"):
            subprocess.check_call(
                args=[
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links", dependency_path,
                    os.path.join(dependency_path, filename),
                ],
                shell=True,
            )
        else:
            print(f"File {filename} is not a valid wheel file.")
except subprocess.CalledProcessError:
    print("Failed to install the local package.")

# try:
#     # 安装剩余依赖库
#     subprocess.check_call(["pip", "install", "fitter==1.4.1", "--index-url", mirror], shell=True)
# except subprocess.CalledProcessError:
#     print("Failed to install the package.")
    

# 清空当前python环境下所有的依赖库，使python环境恢复到初始状态
# subprocess.check_call(["pip", "uninstall", "-y"], shell=True)
