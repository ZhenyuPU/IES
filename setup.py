from setuptools import setup, find_packages

setup(
    name='IES',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'IES': ['data/**/*'],  # 包含 data 目录及其所有子目录中的文件
        'IES': ['Utils/schema.json'],
    },
    install_requires=[
        # 如果有依赖的库，可以列在这里
    ],
    description='A package for integrated energy system',
    author='Zhenyu PU',
    author_email='zhenyupu@foxmail.com',
    url='https://github.com/zhenyupu',  # 如果有GitHub链接的话
)
