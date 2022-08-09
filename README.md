# paddleV2.0.2 gpu Linux版本

## 1．打包为wheel文件

安装wheel
```bash
pip3.7 install wheel
```
打包wheel
```bash
pip3.7 wheel --wheel-dir=./wheel_dir ./
```
> wheel-dir 为wheel 输出文件夹，后面接项目文件夹（即包含setup.py的文件夹）
