if __name__ == "__main__":
    import subprocess

    # from utils import check_models
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # from Docs.configs import individual

    # obfuscated = False
    # check_model = True
    output_dependency_libraries = True

    current_dir = os.path.dirname(__file__)
    file_path_txt = os.path.join(current_dir, "requirements.txt")
    file_path_yml = os.path.join(current_dir, "environment.yml")

    root_dir = os.path.dirname(current_dir)
    # if individual:
    #     files = {
    #         "Docs": os.path.join(root_dir, "Docs", "utils.py"),
    #         "Economics": os.path.join(root_dir, "Economics", "main.py"),
    #         "Geomechanics": os.path.join(root_dir, "Geomechanics", "models.py"),
    #         "Engn_horizontal": os.path.join(root_dir, "Engn_horizontal", "train_pred_cpr_pop.py"),
    #         "Engn_inclination": os.path.join(root_dir, "Engn_inclination", "train_pred_cpr_pop.py"),
    #         "Engn_vertical": os.path.join(root_dir, "Engn_vertical", "train_pred_cpr_pop.py"),
    #     }
    # else:
    #     files = {
    #         "Docs": os.path.join(root_dir, "Docs", "utils.py"),
    #         "Economics": os.path.join(root_dir, "Economics", "main.py"),
    #         "Geomechanics": os.path.join(root_dir, "Geomechanics", "models.py"),
    #         "Engineering": os.path.join(root_dir, "Engineering", "train_pred_cpr_pop.py"),
    #     }

    # 将当前python环境的所有依赖库导出为txt文件
    if output_dependency_libraries:
        with open(file_path_txt, "w") as f:
            subprocess.check_call(["pip", "list", "--format=freeze"], stdout=f)
        print("\nrequirements.txt has been generated!\n")
        # 将当前conda环境的所有依赖库导出为yml文件
        with open(file_path_yml, "w") as f:
            subprocess.check_call(["conda", "env", "export"], stdout=f)
        print("environment.yml has been generated!\n")

    # # 将各个文件夹下一个最重要的脚本进行模糊加密
    # if obfuscated:
    #     for name, file_path in files.items():
    #         try:
    #             subprocess.check_call(args=["pyarmor", "gen", "--output", os.path.join(root_dir, name), file_path], shell=True)
    #         except Exception as e:
    #             print(f"\n{e}\n")

    # # 检查当前环境可使用的模型
    # if check_model:
    #     check_models()

    print("\n__init__.py is finished.\n")
