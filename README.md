# Football Player Tracker

目标：在足球视频中手动选定几个特定球员，随后持续画脚下圆环，并保持球员之间的连线跟随运动。

## 1. 安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果希望最终视频保留原音频，请额外安装 `ffmpeg`，并确保终端中能运行：

```bash
ffmpeg -version
```

## 2. 最简单的运行方式

把视频放到当前目录，例如 `match.mp4`：

```bash
python football_tracker.py match.mp4 --targets 3 --show --repair
```

程序会停在第一帧，依次让你框选 A、B、C 三个球员。

选择 ROI 时：
- 鼠标拖框
- Enter / Space 确认
- 每个球员选一次

输出默认是：

```text
result.mp4
```

## 3. 球员第一帧看不清

例如从第 12.5 秒开始：

```bash
python football_tracker.py match.mp4 --start 12.5 --targets 3 --show --repair
```

输出视频也会从 12.5 秒开始。

## 4. 三种连线方式

三人闭合三角形（默认）：

```bash
python football_tracker.py match.mp4 --targets 3 --connect polygon
```

只连 A-B、B-C：

```bash
python football_tracker.py match.mp4 --targets 3 --connect chain
```

所有目标两两相连：

```bash
python football_tracker.py match.mp4 --targets 4 --connect all
```

## 5. 跟踪太抖 / 太迟钝

默认：

```text
--smooth 0.28
```

更稳：

```bash
--smooth 0.18
```

响应更快：

```bash
--smooth 0.40
```

## 6. 远景球员漏检

可以尝试：

```bash
--conf 0.08 --imgsz 1600
```

但推理会更慢、更吃 GPU。

## 7. 模型大小

默认：

```text
yolo26s.pt
```

速度优先：

```bash
--model yolo26n.pt
```

如果设备性能足够、希望检测更稳，可以使用更大的 YOLO 检测模型。

## 8. 人工修复模式

加上：

```text
--repair
```

如果某个目标连续丢失较久，程序会暂停，让你重新框选该球员，并把逻辑身份 A/B/C 绑定到新的 Track ID。

这对同队球员互相遮挡、镜头切换后的成片处理很实用。
