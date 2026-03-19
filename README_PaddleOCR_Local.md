# PaddleOCR 本地图片识别说明

这个项目提供了一套本地运行的图片 OCR 工具，使用 `PaddleOCR` 从图片中提取文字。

适合的场景：

- 扫描件文字提取
- 手机拍照图片提取文字
- 中文、英文混合图片识别
- 批量识别整个文件夹内的图片

## 已新增文件

- `paddle_ocr_local.py`：主识别脚本
- `requirements-paddleocr.txt`：依赖清单
- `run_paddle_ocr.bat`：Windows 双击启动脚本

## 目录建议

建议把待识别图片放到：

- `D:\OCR Test\ocr_images`

识别结果默认输出到：

- `D:\OCR Test\paddleocr_output`

输出内容包括：

- 每张图片一个 `.txt` 纯文本文件
- 每张图片一个 `.json` 明细文件
- 一个 `summary.csv` 汇总文件

## 安装步骤

### 1. 安装 Python

建议安装 Python 3.10 或以上版本。

安装后先确认下面任意一个命令可用：

```powershell
py -3 --version
```

或：

```powershell
python --version
```

### 2. 安装依赖

进入项目目录后执行：

```powershell
py -3 -m pip install -r requirements-paddleocr.txt
```

如果你的电脑没有 `py`，就改用：

```powershell
python -m pip install -r requirements-paddleocr.txt
```

## 首次运行

首次运行时，PaddleOCR 通常会自动下载所需模型到本机缓存目录。下载完成后，后续就可以本地重复使用。

如果你想完全离线运行，可以先准备好本地模型目录，然后在命令里指定：

- `--det-model-dir`
- `--rec-model-dir`
- `--cls-model-dir`

## 运行方式

### 方式 1：直接双击

双击：

- `D:\OCR Test\run_paddle_ocr.bat`

### 方式 2：命令行运行整个图片文件夹

```powershell
py -3 paddle_ocr_local.py "D:\OCR Test\ocr_images"
```

### 方式 3：识别单张图片

```powershell
py -3 paddle_ocr_local.py "D:\OCR Test\ocr_images\sample.jpg"
```

### 方式 4：指定英文模型

```powershell
py -3 paddle_ocr_local.py "D:\OCR Test\ocr_images" --lang en
```

### 方式 5：指定本地模型目录

```powershell
py -3 paddle_ocr_local.py "D:\OCR Test\ocr_images" --det-model-dir "D:\models\det" --rec-model-dir "D:\models\rec" --cls-model-dir "D:\models\cls"
```

## 参数说明

- `--output-dir`：指定输出目录
- `--lang`：识别语言，默认 `ch`
- `--no-recursive`：文件夹模式下不递归扫描子目录
- `--disable-angle-cls`：关闭文字方向分类
- `--det-model-dir`：指定本地检测模型目录
- `--rec-model-dir`：指定本地识别模型目录
- `--cls-model-dir`：指定本地方向分类模型目录
- `--show-log`：显示 PaddleOCR 内部日志

## 常见问题

### 1. 提示找不到 Python

说明当前系统没有安装 Python，或者没有加入环境变量。

优先检查：

```powershell
py -3 --version
```

### 2. 首次运行很慢

这是正常现象。首次运行通常会下载模型，之后会明显快很多。

### 3. 想完全离线使用

你可以先把模型下载到本地目录，再用下面参数指定：

- `--det-model-dir`
- `--rec-model-dir`
- `--cls-model-dir`

### 4. 输出结果在哪里

默认在：

- `D:\OCR Test\paddleocr_output`

## 当前环境说明

我已经把 PaddleOCR 工程文件搭好了，但这台机器当前命令行里还没有可直接调用的 `python` 或 `py`，所以还没法在这里替你完成安装和实际跑通。

等你本机装好 Python 后，按上面的命令执行即可。如果你愿意，我下一步可以继续帮你把它升级成：

- 图片 OCR + PDF OCR
- OCR 结果自动汇总到 Excel
- 带简单界面的本地桌面版
