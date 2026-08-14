"""pytest 公共配置：把项目根目录加入 sys.path（测试从任意目录运行均可导入项目模块）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
