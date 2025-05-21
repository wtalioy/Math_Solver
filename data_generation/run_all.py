import subprocess
import sys

if __name__ == "__main__":
    print("开始训练...")
    train_result = subprocess.run(
        [sys.executable, "train.py"], cwd="./data_generation")
    if train_result.returncode != 0:
        print("训练失败，已退出。")
        sys.exit(1)
    print("训练完成，开始测试...")
    test_result = subprocess.run(
        [sys.executable, "test.py"], cwd="./data_generation")
    if test_result.returncode != 0:
        print("测试失败。")
        sys.exit(1)
    print("全部流程完成！")
