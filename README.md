# paddle v2.1.0-gpu-win-py3.6 cuda10


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

## 需要修改的地方
paddle/dataset/image.py
```bash
if six.PY3:
    import subprocess
    import sys
    import_cv2_proc = subprocess.Popen(
        [sys.executable, "-c", "import cv2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = import_cv2_proc.communicate()
    retcode = import_cv2_proc.poll()
    if retcode != 0:
        cv2 = None
    else:
        import cv2
else:
    try:
        import cv2
    except ImportError:
        cv2 = None
```
替换成
```python
import cv2
```
```bash
rm paddle/fluid/core_avx.lib
rm paddle/fluid/core_avx.pyd
rm paddle/libs
```

paddle/fluid/core.py

```text
.core_avx
```
替换成
```text
core_avx
```