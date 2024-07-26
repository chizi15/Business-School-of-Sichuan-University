#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import subprocess
import os

"""
1. 如果pycaret最新版装不上，或者依赖库版本冲突，则先装低版本比如3.2.0，再升级到高版本。
2. 依赖库版本文件和离线依赖库中不能有pip，否则当依次检查、卸载、安装每一个依赖库直到pip时，卸载pip时会报错，而无法卸载。因为如果能够卸载pip，后续的依赖库将不能被安装。
"""


# def detect_encoding(file_path: str):
#     import chardet

#     with open(file_path, "rb") as f:
#         return chardet.detect(f.read())["encoding"]


def install_missing_libraries(
    libs: list,
    file_path: str,
    mirrors: list[str],
    dependencies_path: str,
    setup_type: int | str = "local",
):
    """
    检查指定的库是否已经安装，如果没有安装，则使用pip安装它们。注意：使用subprocess.check_call安装python库时不能用梯子，否则会一直连接超时。
    Args:
        libs (list): 需要检查的库列表。
        setup_type (int): 安装类型，
        0: 是每一个库都从每一个镜像中依次尝试下载，遍历检查的速度慢，但不易出现掉落到最后一个官方镜像，从而下载失败的情况。
        1: 表示所有依赖库都从每一个镜像依次尝试下载，如果在某一个镜像中有一个库下载不成功，则会跳到下一个镜像，所以该方式容易落到最后一个镜像去下载，速度会变很慢。
        2: 表示直接安装指定的库。在新环境中，使用这种方式安装最纯净，不会有多余的库。
        'local': 表示安装下载到本地的依赖库。
        mirrors (list): 镜像地址列表。
        file_path (str): 依赖库文件路径。
        dependencies_path (str): 依赖库下载路径。
    Returns:
        bool: True表示所有库都已经安装，False表示有库未安装。
    """
    all_installed = True

    if setup_type == 0:
        for lib in libs:
            try:
                importlib.import_module(lib)
            except ImportError:
                # logs_setup.warning(f"{lib} is not installed, installing...")
                print(f"{lib} is not installed, installing...")
                for mirror in mirrors:
                    try:
                        subprocess.check_call(
                            ["pip", "install", lib, "--index-url", mirror]
                        )

                        # subprocess.check_call(['pip', 'install', lib, '--index-url', mirror, '--user']) # --user表示安装到用户目录下，而不是系统目录，此时需要将'C:\Users\ZC\AppData\Roaming\Python\版本号（如Python310）\Scripts'添加到环境变量中

                        # subprocess.check_call(['pip', 'install', lib, '--index-url', mirror, '--user', '--no-cache-dir']) # --no-cache-dir表示不使用缓存，直接从网络下载

                        # subprocess.check_call(['pip', 'install', lib, '--index-url', mirror, '--user', '--no-cache-dir', '--no-binary', ':all:']) # --no-binary表示不使用二进制文件，直接从源码安装

                        # subprocess.check_call(['pip', 'install', lib, '--index-url', mirror, '--user', '--no-cache-dir', '--no-binary', ':all:', '--compile']) # --compile表示安装时编译源码

                        # subprocess.check_call(['pip', 'install', lib, '--index-url', mirror, '--user', '--no-cache-dir', '--no-binary', ':all:', '--compile', '--ignore-installed']) # --ignore-installed表示忽略已安装的库

                        # logs_setup.info(f"{lib} has been installed successfully.")
                        print(f"{lib} has been installed successfully.")
                        importlib.invalidate_caches()
                        break
                    except subprocess.CalledProcessError:
                        pass
                else:
                    print(f"Failed to install {lib}.")
                    all_installed = False

    elif setup_type == 1:
        print(f"'\n'{libs}'\n'")
        for mirror in mirrors:
            try:
                subprocess.check_call(["pip", "install", "-i", mirror, "-r", file_path])
                break
            except subprocess.CalledProcessError:
                pass
        else:
            print(f"Failed to install all libraries needed.")
            all_installed = False

    elif setup_type == 2:
        print(f"'\n'{libs}'\n'")
        for mirror in mirrors:
            try:
                subprocess.check_call(["pip", "install", "-i", mirror, *libs])
                break
            except subprocess.CalledProcessError:
                pass
        else:
            print(f"Failed to install all libraries needed.")
            all_installed = False

    elif setup_type == "local":
        filename = None
        for filename in os.listdir(dependencies_path):
            if filename.endswith(".whl") or filename.endswith(".tar.gz"):
                try:
                    subprocess.check_call(
                        args=[
                            "pip",
                            "install",
                            "--no-index",
                            "--find-links",
                            dependencies_path,
                            os.path.join(dependencies_path, filename),
                        ],
                        shell=True,
                    )
                except subprocess.CalledProcessError:
                    print(f"Failed to install the local package {filename}.")
                    all_installed = False
            else:
                print(f"File {filename} is not a valid wheel file.")

    else:
        raise ValueError(f"Invalid setup_type: {setup_type}")

    return all_installed


def download_pack(
    libs: list,
    mirrors: list[str],
    file_path: str,
    dependencies_path: str,
    setup_type: int | str,
) -> bool:
    """
    下载依赖库并打包。
    Args:
        mirrors (list): 镜像地址列表。
        file_path (str): 依赖库文件路径。
    Returns:
        bool: True表示所有库都已经下载并打包，False表示有库未下载或打包。
    """
    for mirror in mirrors:
        try:
            if setup_type == ("local" or 1):
                subprocess.check_call(
                    args=[
                        "pip",
                        "download",
                        "-r",
                        file_path,
                        "-i",
                        mirror,
                        "-d",
                        dependencies_path,
                    ],
                    shell=True,
                )
                all_downloaded = True
                break
            elif setup_type == (0 or 2):
                for lib in libs:
                    subprocess.check_call(
                        args=[
                            "pip",
                            "download",
                            lib,
                            "-i",
                            mirror,
                            "-d",
                            dependencies_path,
                        ],
                        shell=True,
                    )
                all_downloaded = True
                break
            else:
                raise ValueError(f"Invalid setup_type: {setup_type}")

        except subprocess.CalledProcessError:
            pass

    else:
        all_downloaded = False

    return all_downloaded


if __name__ == "__main__":
    """
    特别重要：搭建新环境或升级python库时setup.py脚本的使用方法

    方法一：
    1. 在conda中创建一个指定Python版本的纯净虚拟环境
    2. 在该环境中手动pip依次安装所有依赖库，即不使用subprocess安装；并测试新环境是否能成功运行所有脚本；如果成功，进行下一步；不成功，则根据之前的依赖库版本txt文件，将当前环境中重要的依赖库依次替换为之前txt文件中的依赖库版本，例如scipy、numpy、pandas、scikit-learn等，直到所有脚本能成功运行；进入下一步。
    3. 此时需要将当前环境的依赖库版本保存下来，即运行__init__.py脚本，生成requirements.txt文件，在下一步中使用。
    4. 此时需要按requirements.txt中的版本号将依赖库下载到本地dependencies_download文件夹中，即运行setup.py脚本，同时将download_packages = True，install_packages = False，setup_type_download = 'local'，dependencies_version = ["requirements.txt"]。
    5. 将依赖库下载到本地后，再到其他conda虚拟环境或者python环境中运行setup.py脚本，进行最后验证；同时将download_packages = False，install_packages = True，setup_type_install = 'local'，dependencies_version = ["requirements.txt"]。

    方法二：
    将原本能跑通的conda环境复制出一个新环境，在新环境中手动升级python和pycaret[full]，如果依赖库版本有冲突，则按照原来环境中的依赖库版本号，重新安装那些有冲突的库，直到所有脚本能成功运行。

    方法三：
    按照pycaret320_py3106_basic.txt新建一个conda环境，在新环境中手动升级python和pycaret[full]，如果依赖库版本有冲突，则按照原来环境中的依赖库版本号，重新安装那些有冲突的库，直到所有脚本能成功运行。
    """

    download_packages = True  # True, False
    install_packages = True  # True, False
    setup_type_download = 2  # 0, 1, 2, 'local'; 其中，2表示下载libs_basic列表中的库, 'local'表示按本地txt文件下载依赖库
    setup_type_install = "local"  # 0, 1, 2, 'local'
    dependencies_version = [
        "requirements.txt",
        # "pycaret332_py3119.txt",
        # "pycaret320_py3106_basic.txt",
        # "pycaret320_py3106_full.txt",
    ]
    current_dir = os.path.dirname(__file__)
    file_path = current_dir + "\\" + dependencies_version[0]
    dependencies_path = current_dir + "\\dependencies_download"
    script_name = os.path.basename(__file__)
    mirrors = [
        "https://mirrors.huaweicloud.com/repository/pypi/simple/",
        "https://mirrors.cloud.tencent.com/pypi/simple/",
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://mirror.baidu.com/pypi/simple/",
        "https://mirrors.163.com/pypi/simple/",
        "https://pypi.douban.com/simple",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.ustc.edu.cn/pypi/web/simple",
        "https://pypi.org/simple/",
    ]
    libs_basic = [
        "chinese_calendar",
        "openpyxl",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "fitter",
        "prophet",
    ]  # 第一，一定要遵循“奥卡姆剃刀原则”，能用尽量少的依赖库就尽量少用，以防各个依赖库相互冲突。第二，对核心功能影响越大的库，最好放在越后面安装，以免将其放在前面安装时，当安装后面有版本冲突的库时，会导致前面的库被卸载，从而导致前面的库无法使用。第三，interpret-community的依赖库可能和pycaret[full]的依赖库存在版本冲突，当冲突出现时，则不装interpret-community，只装pycaret[full]，同样可以实现绝大多数的分析功能；一般情况下，interpret-community应在pycaret[full]之后安装，可减少版本冲突的影响。第四，pycaret[full]可能不包含xgboost，如不包含需添加到libs_basic列表中；但安装xgboost后出现scipy等计算库的报错，则应注释掉。

    if download_packages:
        # 内网用户无法从外网下载，可以使用后面本地打包的方式'local'安装依赖库
        for mirror in mirrors[:2]:
            try:
                subprocess.check_call(
                    ["python", "-m", "pip", "install", "--upgrade", "pip", "-i", mirror]
                )
                print("'pip' has been upgrade successfully.")
                # subprocess.check_call(["pip", "install", "pipreqs", "-i", mirror])
                # print("'pipreqs' has been installed successfully.")
                subprocess.check_call(
                    ["pip", "install", "chardet", "-i", mirror]
                )  # 安装chardet库，用于检测文件编码，一定要放到detect_encoding函数前面执行，以防新环境中没有安装chardet库，导致脚本无法运行
                print("'chardet' has been installed successfully.")
                break
            except subprocess.CalledProcessError:
                pass
        else:
            print("Failed to upgrade 'pip'.")

    try:
        # encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding="UTF-16") as f:
            libs = f.readlines()
            libs = [lib.strip() for lib in libs if lib.strip() != ""]
    except Exception as e:
        print(f"Error detecting encoding: {e}\n, using default encoding 'UTF-16'.\n")
        try:
            encoding = "UTF-16"
            with open(file_path, "r", encoding=encoding) as f:
                libs = f.readlines()
                libs = [
                    lib.strip() for lib in libs if lib.strip() != ""
                ]  # 去掉每行末尾的换行符，并去掉空行
                # print(f"'\n'{libs}'\n'")
        except Exception as e:
            print(f"Error reading file: {e}\n, using encoding 'ascii'.\n")
            try:
                encoding = "ascii"
                with open(file_path, "r", encoding=encoding) as f:
                    libs = f.readlines()
                    libs = [lib.strip() for lib in libs if lib.strip() != ""]
            except Exception as e:
                libs = libs_basic

    if setup_type_download == 2:
        libs = libs_basic

    if download_packages:
        print("\nStart to download the libraries needed.\n")
        all_downloaded = download_pack(
            libs=libs,
            mirrors=mirrors,
            file_path=file_path,
            dependencies_path=dependencies_path,
            setup_type=setup_type_download,
        )
        if all_downloaded:
            print("\nAll libraries needed have been downloaded successfully.\n")
        else:
            print("\nFailed to download all libraries needed.\n")

    if install_packages:
        print("\nStart to check the libraries needed.\n")
        all_installed: bool = install_missing_libraries(
            libs=libs,
            mirrors=mirrors,
            file_path=file_path,
            dependencies_path=dependencies_path,
            setup_type=setup_type_install,
        )
        if all_installed:
            print("\nAll libraries needed have been installed successfully.\n")
        else:
            raise Exception("Failed to install all libraries needed.")

    print(f"{script_name} is finished.\n")
