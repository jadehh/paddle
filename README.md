# paddleV2.3 gpu Linux版本

## 1．打包为wheel文件

安装wheel
```bash
pip install wheel
```
打包wheel
```bash
pip wheel --wheel-dir=./wheel_dir ./
```
> wheel-dir 为wheel 输出文件夹，后面接项目文件夹（即包含setup.py的文件夹）
