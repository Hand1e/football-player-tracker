# Football Player Tracker

在足球视频中手动选定 2 个或多个特定球员，随后持续在脚下绘制固定尺寸圆环，并保持球员之间的连线跟随运动。

核心方案：**YOLO + BoT-SORT/ReID + 应用层 anti-ID-switch + EMA 平滑**。

## 新版重点

- 支持 `--pick-start` / `--select-frame` 可视化选择起始帧
  - 拖动 `Frame` 进度条
  - `Space` 播放/暂停
  - 左/右方向键单帧前后（也支持 A/D）
  - `[` / `]` 后退/前进 1 秒
  - `Enter` 确认；输出从确认帧开始
- 不传 `--connect` 时自动决定
  - 2 人：`chain`
  - 3 人及以上：`polygon`
- 输出视频不再显示 A/B/C 标签，只画脚下圆环和连线
- 圆环尺寸只在初次绑定时确定，后续不随检测框远近而放大/缩小
- `--style-ui` 样式选择界面
  - 5 色：黄 / 红 / 蓝 / 紫 / 橙
  - 圆环和连线可分别选颜色、粗细
  - 粗细范围 1~10
- 保留命令行样式参数：`--ring-color`、`--ring-thickness`、`--line-color`、`--line-thickness`
- 颜色支持英文和中文名称
- 增加交叉防换人：
  - 两人靠近/遮挡时冻结逻辑身份
  - 遮挡期间优先使用运动预测，不立即相信可能已交换的 Track ID
  - 分开后用 **HSV 三维球衣特征 + 位置 + 方向 + 尺寸** 做全局匹配
  - HSV 特征会排除草地绿色像素
  - 白/亮球衣与深色球衣增加明暗约束，降低错误重连
  - 低置信度时宁可继续预测，也不立即切换到另一名球员

## 安装

建议 Python 3.10+：

```bash
pip install -r requirements.txt
```

如果要保留源视频音频，请安装 `ffmpeg` 并确保：

```bash
ffmpeg -version
```

可以正常运行。

## 推荐用法：两名球员

```bash
python football_tracker.py match.mp4 --targets 2 --pick-start --style-ui --show --repair
```

两人时不需要写 `--connect chain`，程序会自动选择 `chain`。

## 起始帧选择

加：

```bash
--pick-start
```

或：

```bash
--select-frame
```

操作：

```text
拖动 Frame      选择画面
Space           播放 / 暂停
Left / Right    单帧前后
A / D           单帧前后备用键
[ / ]           后退 / 前进 1 秒
Enter           确认当前帧
Q / ESC         取消
```

输出视频从确认帧开始。

## 选择球员

确认起始帧后，程序依次让你框选目标球员。框尽量完整覆盖球员身体。

内部仍用 A/B/C 区分逻辑目标，但最终输出不会显示这些字母。

## 样式

打开可视化样式选择器：

```bash
--style-ui
```

也可以直接命令行指定：

```bash
python football_tracker.py match.mp4 --targets 2 --ring-color yellow --ring-thickness 5 --line-color red --line-thickness 4
```

支持中文：

```bash
--ring-color 黄 --line-color 红
```

可用颜色：

```text
yellow / red / blue / purple / orange
黄 / 红 / 蓝 / 紫 / 橙
```

## 连线方式

不传 `--connect` 时：

```text
2 人        -> chain
3 人及以上  -> polygon
```

也可以手动覆盖：

```bash
--connect chain
--connect polygon
--connect all
```

## 圆环大小

圆环尺寸在第一次绑定球员时根据该球员检测框确定，之后保持固定，不再因为镜头拉近/拉远或 bbox 抖动而变化。

## 交叉防换人

程序在 BoT-SORT/ReID 之外增加一层逻辑身份保护：

```text
正常跟踪
  ↓
两名目标靠近 / 遮挡
  ↓
冻结身份绑定
  ↓
短时运动预测
  ↓
重新分开
  ↓
HSV 球衣 + 位置 + 方向 + 尺寸
  ↓
Hungarian 全局分配
  ↓
高置信度才重绑 Track ID
```

常用调试参数：

```text
--occlusion-iou 0.10
--occlusion-distance 1.15
--separation-frames 4
--identity-threshold 0.58
--identity-margin 0.06
--reid-gate 2.6
```

完全关闭额外防换人逻辑：

```bash
--no-anti-switch
```

注意：同队球员球衣完全相同时，仅靠球衣颜色无法 100% 区分身份。程序还会同时使用运动轨迹、位置、尺寸以及 BoT-SORT 自带 ReID。极端长遮挡或镜头切换仍建议使用 `--repair`。

## 人工修复

```bash
--repair
```

目标长时间无法确认时，程序会暂停，让你重新框选对应球员。

## 远景漏检

可以尝试：

```bash
--conf 0.08 --imgsz 1600
```

## 平滑程度

默认：

```text
--smooth 0.28
```

更稳但响应稍慢：

```bash
--smooth 0.18
```

更灵敏：

```bash
--smooth 0.40
```

## 项目结构

```text
football_tracker.py      主程序
tracker_core.py          跟踪、HSV 特征、身份保护、全局重关联
ui_tools.py              起始帧选择、目标选择、样式选择
football_botsort.yaml    BoT-SORT/ReID 配置
requirements.txt         Python 依赖
```

## 输出

默认：

```text
result.mp4
```

如果系统存在 ffmpeg，会尝试把原视频音频重新合并到输出视频。
