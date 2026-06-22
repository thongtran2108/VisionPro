"""
core/translations.py — Bảng dịch cho VisionPro (vi / en / zh).

Mỗi record = (source, en, vi, zh):
  • source = ĐÚNG chuỗi literal trong code (đang trộn Anh/Việt) — dùng làm key.
  • en/vi/zh = bản dịch tương ứng. Nếu một ngôn ngữ trùng source thì để y
    nguyên source (vòng build bên dưới tự bỏ qua → fallback về source).

Nguyên tắc:
  • Chuỗi nguồn tiếng Anh (menu, tên tool, category…) → en == source.
  • Chuỗi nguồn tiếng Việt (mô tả, thông báo…)        → vi == source.
  • Thiếu bản dịch cho ngôn ngữ nào → tr() trả nguyên source (an toàn).

Thêm chuỗi mới: chỉ cần thêm 1 dòng vào _ENTRIES (không phải sửa core/i18n.py).
Tên thương hiệu giữ nguyên: Vision Ultimate, VisionPro, NEO Vision Pro,
PatMax, PatFind, YOLO…
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

# (source, en, vi, zh)
_ENTRIES: List[Tuple[str, str, str, str]] = [

    # ── Splash / progress khởi động (nguồn: Việt) ──────────────────
    ("Đang khởi tạo giao diện…", "Initializing interface…", "Đang khởi tạo giao diện…", "正在初始化界面…"),
    ("Đang nạp dự án…", "Loading project…", "Đang nạp dự án…", "正在加载项目…"),
    ("Hoàn tất", "Done", "Hoàn tất", "完成"),
    ("Đang dựng giao diện…", "Building interface…", "Đang dựng giao diện…", "正在构建界面…"),
    ("Đang tạo menu…", "Creating menus…", "Đang tạo menu…", "正在创建菜单…"),
    ("Đang kết nối tín hiệu…", "Connecting signals…", "Đang kết nối tín hiệu…", "正在连接信号…"),
    ("Giao diện sẵn sàng", "Interface ready", "Giao diện sẵn sàng", "界面就绪"),
    ("Đang đọc file…", "Reading file…", "Đang đọc file…", "正在读取文件…"),
    ("Đang phân tích JSON…", "Parsing JSON…", "Đang phân tích JSON…", "正在解析 JSON…"),
    ("Đang dựng pipeline…", "Building pipeline…", "Đang dựng pipeline…", "正在构建流程…"),
    ("Đang dựng node…", "Building nodes…", "Đang dựng node…", "正在构建节点…"),

    # ── Menu bar (nguồn: Anh) ──────────────────────────────────────
    ("File", "File", "Tệp", "文件"),
    ("New Pipeline", "New Pipeline", "Pipeline mới", "新建流程"),
    ("Open Pipeline…", "Open Pipeline…", "Mở pipeline…", "打开流程…"),
    ("Save Pipeline", "Save Pipeline", "Lưu pipeline", "保存流程"),
    ("Save As…", "Save As…", "Lưu thành…", "另存为…"),
    ("Switch Project…", "Switch Project…", "Đổi dự án…", "切换项目…"),
    ("Mở dialog Recent files → pick project khác.",
     "Open the Recent files dialog → pick another project.",
     "Mở dialog Recent files → pick project khác.",
     "打开最近文件对话框 → 选择其他项目。"),
    ("Quit", "Quit", "Thoát", "退出"),
    ("Run", "Run", "Chạy", "运行"),
    ("Run Pipeline", "Run Pipeline", "Chạy pipeline", "运行流程"),
    ("Stop", "Stop", "Dừng", "停止"),
    ("View", "View", "Hiển thị", "视图"),
    ("Fit Canvas", "Fit Canvas", "Vừa khung hình", "适应画布"),
    ("Reset Zoom", "Reset Zoom", "Đặt lại zoom", "重置缩放"),
    ("Switch to Image Viewer", "Switch to Image Viewer", "Sang Image Viewer", "切换到图像查看器"),
    ("Full Image View", "Full Image View", "Xem ảnh toàn màn hình", "全屏图像视图"),
    ("  ● Full Image View — nhấn F11 hoặc Esc để thoát",
     "  ● Full Image View — press F11 or Esc to exit",
     "  ● Full Image View — nhấn F11 hoặc Esc để thoát",
     "  ● 全屏图像视图 — 按 F11 或 Esc 退出"),
    ("🤖 Open YOLO Studio", "🤖 Open YOLO Studio", "🤖 Mở YOLO Studio", "🤖 打开 YOLO Studio"),
    ("✏ Label Images...", "✏ Label Images...", "✏ Gắn nhãn ảnh...", "✏ 标注图像..."),
    ("Tools", "Tools", "Công cụ", "工具"),
    ("🔌  PLC Connection…", "🔌  PLC Connection…", "🔌  Kết nối PLC…", "🔌  PLC 连接…"),
    ("🌐  SFC / MES Integration…", "🌐  SFC / MES Integration…", "🌐  Tích hợp SFC / MES…", "🌐  SFC / MES 集成…"),
    ("📷  Camera Setup…", "📷  Camera Setup…", "📷  Cài đặt camera…", "📷  相机设置…"),
    ("Account", "Account", "Tài khoản", "账户"),
    ("🔓  Đăng nhập…", "🔓  Log in…", "🔓  Đăng nhập…", "🔓  登录…"),
    ("🔒  Đăng xuất", "🔒  Log out", "🔒  Đăng xuất", "🔒  注销"),
    ("🔑  Đổi mật khẩu…", "🔑  Change Password…", "🔑  Đổi mật khẩu…", "🔑  修改密码…"),
    ("Help", "Help", "Trợ giúp", "帮助"),
    ("About", "About", "Giới thiệu", "关于"),
    ("Shortcuts", "Shortcuts", "Phím tắt", "快捷键"),

    # ── Toolbar (nguồn: Anh + Việt) ────────────────────────────────
    ("New", "New", "Mới", "新建"),
    ("Open", "Open", "Mở", "打开"),
    ("Save", "Save", "Lưu", "保存"),
    ("Fit", "Fit", "Vừa", "适应"),
    ("Zoom", "Zoom", "Zoom", "缩放"),
    ("Clear", "Clear", "Xoá", "清空"),
    ("Fit canvas", "Fit canvas", "Vừa khung hình", "适应画布"),
    ("Reset zoom", "Reset zoom", "Đặt lại zoom", "重置缩放"),
    ("Clear all", "Clear all", "Xoá tất cả", "清除全部"),
    ("▶   RUN", "▶   RUN", "▶   CHẠY", "▶   运行"),
    ("nodes", "nodes", "node", "节点"),
    ("🔒  Đăng nhập", "🔒  Log in", "🔒  Đăng nhập", "🔒  登录"),
    ("Đăng xuất", "Log out", "Đăng xuất", "注销"),
    ("Đang đăng nhập — bấm để đăng xuất.",
     "Logged in — click to log out.",
     "Đang đăng nhập — bấm để đăng xuất.",
     "已登录 — 点击注销。"),
    ("Đăng nhập để Sửa tool / YOLO / pipeline. RUN không cần đăng nhập.",
     "Log in to edit tools / YOLO / pipeline. RUN needs no login.",
     "Đăng nhập để Sửa tool / YOLO / pipeline. RUN không cần đăng nhập.",
     "登录以编辑工具 / YOLO / 流程。运行无需登录。"),

    # ── Center tabs ────────────────────────────────────────────────
    ("🔧  Pipeline Canvas", "🔧  Pipeline Canvas", "🔧  Pipeline Canvas", "🔧  流程画布"),
    ("👁  Image Viewer", "👁  Image Viewer", "👁  Trình xem ảnh", "👁  图像查看器"),

    # ── Tool Library panel ─────────────────────────────────────────
    ("⬡  T TOOL LIBRARY", "⬡  TOOL LIBRARY", "⬡  THƯ VIỆN TOOL", "⬡  工具库"),
    ("tools", "tools", "tool", "个工具"),
    ("🔍  Search tools or T name...", "🔍  Search tools or T name...", "🔍  Tìm tool…", "🔍  搜索工具…"),

    # ── Categories ─────────────────────────────────────────────────
    ("Acquire Image", "Acquire Image", "Lấy ảnh", "采集图像"),
    ("Pattern Find", "Pattern Find", "Tìm mẫu", "模板查找"),
    ("Fixture", "Fixture", "Định vị", "定位基准"),
    ("Caliper", "Caliper", "Thước cặp", "卡尺"),
    ("Blob Analysis", "Blob Analysis", "Phân tích Blob", "斑点分析"),
    ("Edge & Geometry", "Edge & Geometry", "Biên & Hình học", "边缘与几何"),
    ("Color Analysis", "Color Analysis", "Phân tích màu", "颜色分析"),
    ("ID & Read", "ID & Read", "Mã & Đọc", "识别与读取"),
    ("Measurement", "Measurement", "Đo lường", "测量"),
    ("Surface Inspection", "Surface Inspection", "Kiểm tra bề mặt", "表面检测"),
    ("Image Processing", "Image Processing", "Xử lý ảnh", "图像处理"),
    ("Calibration", "Calibration", "Hiệu chuẩn", "标定"),
    ("Logic & Flow", "Logic & Flow", "Logic & Luồng", "逻辑与流程"),
    ("Output & Display", "Output & Display", "Xuất & Hiển thị", "输出与显示"),
    ("Communication", "Communication", "Truyền thông", "通信"),
    # "YOLO" giữ nguyên cho cả 3 ngôn ngữ.

    # ── Tool names ─────────────────────────────────────────────────
    ("Camera Acquire", "Camera Acquire", "Lấy ảnh Camera", "相机采集"),
    ("Search PatMax", "Search PatMax", "Tìm PatMax", "PatMax 搜索"),
    ("PatMax Align Tool", "PatMax Align Tool", "Căn chỉnh PatMax", "PatMax 对位工具"),
    ("Caliper Multi-Edge", "Caliper Multi-Edge", "Thước cặp đa cạnh", "多边缘卡尺"),
    ("Find Line", "Find Line", "Tìm đường thẳng", "找直线"),
    ("Find Circle", "Find Circle", "Tìm đường tròn", "找圆"),
    ("Line (2 Points)", "Line (2 Points)", "Đường thẳng (2 điểm)", "直线(两点)"),
    ("Create Rectangle", "Create Rectangle", "Tạo hình chữ nhật", "创建矩形"),
    ("Create Circle", "Create Circle", "Tạo hình tròn", "创建圆"),
    ("Create Ellipse", "Create Ellipse", "Tạo elip", "创建椭圆"),
    ("Create Trapezoid", "Create Trapezoid", "Tạo hình thang", "创建梯形"),
    ("Create Polygon", "Create Polygon", "Tạo đa giác", "创建多边形"),
    ("Intensity Stats", "Intensity Stats", "Thống kê cường độ", "强度统计"),
    ("Color Picker", "Color Picker", "Chọn màu", "取色器"),
    ("Color Segmentation", "Color Segmentation", "Phân vùng màu", "颜色分割"),
    ("Color Match", "Color Match", "So màu", "颜色匹配"),
    ("ID Reader", "ID Reader", "Đọc mã (ID)", "条码识别"),
    ("OCR Max", "OCR Max", "OCR", "OCR 识别"),
    ("Line Intersection", "Line Intersection", "Giao điểm 2 đường", "直线交点"),
    ("Distance Line-Line", "Distance Line-Line", "Khoảng cách Đường-Đường", "线-线距离"),
    ("Line Relation (GD&T)", "Line Relation (GD&T)", "Quan hệ đường (GD&T)", "直线关系 (GD&T)"),
    ("Distance Point-Point", "Distance Point-Point", "Khoảng cách Điểm-Điểm", "点-点距离"),
    ("Distance Point-Line", "Distance Point-Line", "Khoảng cách Điểm-Đường", "点-线距离"),
    ("Angle Line-Line", "Angle Line-Line", "Góc Đường-Đường", "线-线夹角"),
    ("Area Measure", "Area Measure", "Đo diện tích", "面积测量"),
    ("Golden Compare", "Golden Compare", "So mẫu chuẩn", "黄金模板比对"),
    ("Presence / Absence", "Presence / Absence", "Có / Không", "有无检测"),
    ("Surface Defect", "Surface Defect", "Lỗi bề mặt", "表面缺陷"),
    ("Scratch Detection", "Scratch Detection", "Phát hiện trầy xước", "划痕检测"),
    ("Image Math (2 ảnh)", "Image Math (2 imgs)", "Toán ảnh (2 ảnh)", "图像运算(两图)"),
    ("Rotate / Flip", "Rotate / Flip", "Xoay / Lật", "旋转 / 翻转"),
    ("Perspective Warp", "Perspective Warp", "Biến đổi phối cảnh", "透视变换"),
    ("Enhance Contrast", "Enhance Contrast", "Tăng tương phản", "增强对比度"),
    ("Smooth (Median/Bilateral)", "Smooth (Median/Bilateral)", "Làm mượt (Median/Bilateral)", "平滑(中值/双边)"),
    ("Edge Filter", "Edge Filter", "Lọc biên", "边缘滤波"),
    ("Image Convert", "Image Convert", "Chuyển đổi ảnh", "图像转换"),
    ("Crop ROI", "Crop ROI", "Cắt ROI", "裁剪 ROI"),
    ("Threshold", "Threshold", "Ngưỡng", "阈值"),
    ("Morphology", "Morphology", "Hình thái học", "形态学"),
    ("Gaussian Blur", "Gaussian Blur", "Làm mờ Gauss", "高斯模糊"),
    ("Sharpen", "Sharpen", "Làm nét", "锐化"),
    ("Find Contours", "Find Contours", "Tìm đường viền", "查找轮廓"),
    ("Calibrate (Checkerboard)", "Calibrate (Checkerboard)", "Hiệu chuẩn (Bàn cờ)", "标定(棋盘格)"),
    ("Math / Formula", "Math / Formula", "Toán / Công thức", "数学 / 公式"),
    ("AND Gate", "AND Gate", "Cổng AND", "与门"),
    ("OR Gate", "OR Gate", "Cổng OR", "或门"),
    ("NOT Gate", "NOT Gate", "Cổng NOT", "非门"),
    ("Compare", "Compare", "So sánh", "比较"),
    ("Pass/Fail Judge", "Pass/Fail Judge", "Phán định Pass/Fail", "合格/不合格判定"),
    ("Script Tool", "Script Tool", "Công cụ Script", "脚本工具"),
    ("Display", "Display", "Hiển thị", "显示"),
    ("Message", "Message", "Thông báo", "消息"),
    ("Save Image", "Save Image", "Lưu ảnh", "保存图像"),
    ("CSV Logger", "CSV Logger", "Ghi CSV", "CSV 记录"),
    ("Yield Statistics", "Yield Statistics", "Thống kê năng suất", "良率统计"),
    ("PLC Write", "PLC Write", "Ghi PLC", "PLC 写入"),
    ("TCP Send", "TCP Send", "Gửi TCP", "TCP 发送"),
    ("Modbus Write", "Modbus Write", "Ghi Modbus", "Modbus 写入"),
    ("YOLO Detect", "YOLO Detect", "Phát hiện YOLO", "YOLO 检测"),

    # ── Properties panel ───────────────────────────────────────────
    ("⚙  PROPERTIES", "⚙  PROPERTIES", "⚙  THUỘC TÍNH", "⚙  属性"),
    ("Info", "Info", "Thông tin", "信息"),
    ("Params", "Params", "Tham số", "参数"),
    ("Output", "Output", "Đầu ra", "输出"),
    ("Preview", "Preview", "Xem trước", "预览"),
    ("PORTS", "PORTS", "CỔNG", "端口"),
    ("Category", "Category", "Danh mục", "类别"),
    ("Click a node to inspect it", "Click a node to inspect it", "Bấm vào node để xem chi tiết", "点击节点以查看"),
    ("Changes apply on next run  ▶", "Changes apply on next run  ▶", "Thay đổi áp dụng ở lần chạy kế  ▶", "更改将在下次运行时生效  ▶"),
    ("Select a node\nto view properties.", "Select a node\nto view properties.", "Chọn một node\nđể xem thuộc tính.", "选择一个节点\n以查看属性。"),
    ("Select a node\nto edit parameters.", "Select a node\nto edit parameters.", "Chọn một node\nđể sửa tham số.", "选择一个节点\n以编辑参数。"),
    ("Run pipeline to\nsee outputs.", "Run pipeline to\nsee outputs.", "Chạy pipeline để\nxem đầu ra.", "运行流程以\n查看输出。"),
    ("No parameters\nfor this tool.", "No parameters\nfor this tool.", "Tool này\nkhông có tham số.", "此工具\n没有参数。"),
    ("No output yet.\nRun the pipeline first.", "No output yet.\nRun the pipeline first.", "Chưa có đầu ra.\nHãy chạy pipeline trước.", "暂无输出。\n请先运行流程。"),
    ("🖱  Draw ROI (drag rectangle)", "🖱  Draw ROI (drag rectangle)", "🖱  Vẽ ROI (kéo hình chữ nhật)", "🖱  绘制 ROI（拖动矩形）"),
    ("🖱  Draw ROI band (drag rectangle)", "🖱  Draw ROI band (drag rectangle)", "🖱  Vẽ dải ROI (kéo hình chữ nhật)", "🖱  绘制 ROI 带（拖动矩形）"),
    ("🖱  Pick point on image (click)", "🖱  Pick point on image (click)", "🖱  Chọn điểm trên ảnh (click)", "🖱  在图像上选点（点击）"),
    ("Draw ROI", "Draw ROI", "Vẽ ROI", "绘制 ROI"),
    ("Chưa có ảnh upstream. Hãy Run pipeline trước để load ảnh.",
     "No upstream image. Run the pipeline first to load an image.",
     "Chưa có ảnh upstream. Hãy Run pipeline trước để load ảnh.",
     "没有上游图像。请先运行流程以加载图像。"),

    # ── Startup picker ─────────────────────────────────────────────
    ("VisionPro — Load AOI", "VisionPro — Load AOI", "VisionPro — Tải AOI", "VisionPro — 加载 AOI"),
    ("Chọn file AOI gần đây để tiếp tục, hoặc bắt đầu mới.",
     "Pick a recent AOI file to continue, or start fresh.",
     "Chọn file AOI gần đây để tiếp tục, hoặc bắt đầu mới.",
     "选择最近的 AOI 文件以继续，或新建。"),
    ("📂  RECENT FILES", "📂  RECENT FILES", "📂  FILE GẦN ĐÂY", "📂  最近文件"),
    ("📁  Browse…", "📁  Browse…", "📁  Duyệt…", "📁  浏览…"),
    ("Mở file .aoi/.json bất kỳ qua file dialog.",
     "Open any .aoi/.json file via the file dialog.",
     "Mở file .aoi/.json bất kỳ qua file dialog.",
     "通过文件对话框打开任意 .aoi/.json 文件。"),
    ("✨  New blank project", "✨  New blank project", "✨  Dự án trống mới", "✨  新建空白项目"),
    ("Tạo pipeline trống, không load file.",
     "Create an empty pipeline without loading a file.",
     "Tạo pipeline trống, không load file.",
     "创建空流程，不加载文件。"),
    ("Open Selected →", "Open Selected →", "Mở mục đã chọn →", "打开所选 →"),
    ("Mở file đang chọn trong recent list. Double-click vào file cũng được.",
     "Open the selected file in the recent list. Double-clicking a file also works.",
     "Mở file đang chọn trong recent list. Double-click vào file cũng được.",
     "打开最近列表中所选的文件。双击文件也可以。"),
    ("Cancel", "Cancel", "Huỷ", "取消"),
    ("Chưa có file AOI nào gần đây", "No recent AOI files yet", "Chưa có file AOI nào gần đây", "暂无最近的 AOI 文件"),
    ("⭐  DEFAULT", "⭐  DEFAULT", "⭐  MẶC ĐỊNH", "⭐  默认"),

    # ── Login / Change-password / Activation dialogs (nguồn: Việt) ─
    ("Đăng nhập", "Log in", "Đăng nhập", "登录"),
    ("🔒  Đăng nhập Admin", "🔒  Admin Login", "🔒  Đăng nhập Admin", "🔒  管理员登录"),
    ("Đăng nhập để được Sửa tham số và Chạy tool.",
     "Log in to edit parameters and run tools.",
     "Đăng nhập để được Sửa tham số và Chạy tool.",
     "登录以编辑参数并运行工具。"),
    ("Tài khoản", "Username", "Tài khoản", "账户"),
    ("Mật khẩu", "Password", "Mật khẩu", "密码"),
    ("Nhập mật khẩu…", "Enter password…", "Nhập mật khẩu…", "输入密码…"),
    ("Huỷ", "Cancel", "Huỷ", "取消"),
    ("✖  Sai tài khoản hoặc mật khẩu.", "✖  Wrong username or password.", "✖  Sai tài khoản hoặc mật khẩu.", "✖  账户或密码错误。"),
    ("Đổi mật khẩu", "Change Password", "Đổi mật khẩu", "修改密码"),
    ("🔑  Đổi mật khẩu Admin", "🔑  Change Admin Password", "🔑  Đổi mật khẩu Admin", "🔑  修改管理员密码"),
    ("Mật khẩu hiện tại", "Current password", "Mật khẩu hiện tại", "当前密码"),
    ("Mật khẩu đang dùng…", "Current password…", "Mật khẩu đang dùng…", "当前密码…"),
    ("Mật khẩu mới", "New password", "Mật khẩu mới", "新密码"),
    ("Tối thiểu 4 ký tự…", "At least 4 characters…", "Tối thiểu 4 ký tự…", "至少 4 个字符…"),
    ("Nhập lại mật khẩu mới", "Confirm new password", "Nhập lại mật khẩu mới", "确认新密码"),
    ("Gõ lại mật khẩu mới…", "Re-enter the new password…", "Gõ lại mật khẩu mới…", "再次输入新密码…"),
    ("Lưu", "Save", "Lưu", "保存"),
    ("✖  Mật khẩu nhập lại không khớp.", "✖  Passwords do not match.", "✖  Mật khẩu nhập lại không khớp.", "✖  两次输入的密码不一致。"),
    ("Kích hoạt VisionPro", "Activate VisionPro", "Kích hoạt VisionPro", "激活 VisionPro"),
    ("🔑  Kích hoạt chương trình", "🔑  Activate the application", "🔑  Kích hoạt chương trình", "🔑  激活程序"),
    ("Chương trình cần file <b>license.key</b> hợp lệ để chạy. Gửi <b>Machine ID</b> dưới đây cho nhà cung cấp để nhận file license dành riêng cho máy này, sau đó bấm <b>“Chọn file license…”</b>.",
     "The application needs a valid <b>license.key</b> file to run. Send the <b>Machine ID</b> below to your vendor to receive a license file for this machine, then click <b>“Choose license file…”</b>.",
     "Chương trình cần file <b>license.key</b> hợp lệ để chạy. Gửi <b>Machine ID</b> dưới đây cho nhà cung cấp để nhận file license dành riêng cho máy này, sau đó bấm <b>“Chọn file license…”</b>.",
     "本程序需要有效的 <b>license.key</b> 文件才能运行。请将下方的 <b>Machine ID</b> 发送给供应商，以获取本机专用的许可证文件，然后点击 <b>“选择许可证文件…”</b>。"),
    ("MACHINE ID (máy này)", "MACHINE ID (this machine)", "MACHINE ID (máy này)", "MACHINE ID（本机）"),
    ("Copy", "Copy", "Sao chép", "复制"),
    ("Thoát", "Quit", "Thoát", "退出"),
    ("📂  Chọn file license…", "📂  Choose license file…", "📂  Chọn file license…", "📂  选择许可证文件…"),
    ("Tiếp tục", "Continue", "Tiếp tục", "继续"),
    ("✓  Đã copy Machine ID vào clipboard.", "✓  Machine ID copied to clipboard.", "✓  Đã copy Machine ID vào clipboard.", "✓  已将 Machine ID 复制到剪贴板。"),
    ("Vĩnh viễn", "Lifetime", "Vĩnh viễn", "永久"),
    ("Chọn file license", "Choose license file", "Chọn file license", "选择许可证文件"),
    ("License (*.key *.lic *.txt);;Tất cả (*)", "License (*.key *.lic *.txt);;All (*)", "License (*.key *.lic *.txt);;Tất cả (*)", "许可证 (*.key *.lic *.txt);;全部 (*)"),

    # ── Language switcher (mới) ────────────────────────────────────
    ("🌐 Language", "🌐 Language", "🌐 Ngôn ngữ", "🌐 语言"),
    ("Đổi ngôn ngữ", "Change language", "Đổi ngôn ngữ", "切换语言"),
    ("Ngôn ngữ sẽ áp dụng sau khi khởi động lại ứng dụng.",
     "The language will take effect after you restart the application.",
     "Ngôn ngữ sẽ áp dụng sau khi khởi động lại ứng dụng.",
     "语言将在重新启动应用程序后生效。"),

    # ── About / Shortcuts (nguồn: Anh) ─────────────────────────────
    ("<h2 style='color:#00d4ff;'>Vision Ultimate v1.0</h2><p>Automated Optical Inspection<br>PySide6 + OpenCV</p><ul><li>Drag-drop pipeline</li><li>30+ inspection tools</li><li>Real-time image viewer</li><li>Pass/Fail judgment</li></ul>",
     "<h2 style='color:#00d4ff;'>Vision Ultimate v1.0</h2><p>Automated Optical Inspection<br>PySide6 + OpenCV</p><ul><li>Drag-drop pipeline</li><li>30+ inspection tools</li><li>Real-time image viewer</li><li>Pass/Fail judgment</li></ul>",
     "<h2 style='color:#00d4ff;'>Vision Ultimate v1.0</h2><p>Kiểm tra quang học tự động<br>PySide6 + OpenCV</p><ul><li>Pipeline kéo-thả</li><li>30+ công cụ kiểm tra</li><li>Xem ảnh thời gian thực</li><li>Phán định Pass/Fail</li></ul>",
     "<h2 style='color:#00d4ff;'>Vision Ultimate v1.0</h2><p>自动光学检测<br>PySide6 + OpenCV</p><ul><li>拖放式流程</li><li>30+ 检测工具</li><li>实时图像查看</li><li>合格/不合格判定</li></ul>"),
    ("F5 — Run pipeline\nF6 — Stop\nDelete — Delete selected\nTab — Toggle Canvas/Viewer\nF — Fit canvas\nR — Reset zoom\nScroll — Zoom\nMiddle-drag — Pan canvas\nDouble-click node — Open detail window\nDrag port→port — Connect nodes\nCtrl+S/O/N — Save/Open/New",
     "F5 — Run pipeline\nF6 — Stop\nDelete — Delete selected\nTab — Toggle Canvas/Viewer\nF — Fit canvas\nR — Reset zoom\nScroll — Zoom\nMiddle-drag — Pan canvas\nDouble-click node — Open detail window\nDrag port→port — Connect nodes\nCtrl+S/O/N — Save/Open/New",
     "F5 — Chạy pipeline\nF6 — Dừng\nDelete — Xoá mục đã chọn\nTab — Chuyển Canvas/Viewer\nF — Vừa khung hình\nR — Đặt lại zoom\nScroll — Zoom\nKéo chuột giữa — Di chuyển canvas\nDouble-click node — Mở cửa sổ chi tiết\nKéo port→port — Nối node\nCtrl+S/O/N — Lưu/Mở/Mới",
     "F5 — 运行流程\nF6 — 停止\nDelete — 删除所选\nTab — 切换画布/查看器\nF — 适应画布\nR — 重置缩放\n滚轮 — 缩放\n中键拖动 — 平移画布\n双击节点 — 打开详情窗口\n拖动 端口→端口 — 连接节点\nCtrl+S/O/N — 保存/打开/新建"),
]


def _build() -> Dict[str, Dict[str, str]]:
    tables: Dict[str, Dict[str, str]] = {"en": {}, "vi": {}, "zh": {}}

    # 1) Dữ liệu dịch hàng loạt (auto-extract toàn bộ chuỗi GUI →
    #    core/i18n_data.json). Bao phủ tên/mô tả/nhãn/tooltip tool, choices,
    #    và mọi chuỗi trong các dialog.
    try:
        path = os.path.join(os.path.dirname(__file__), "i18n_data.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for src, v in data.items():
            for lang in ("en", "vi", "zh"):
                t = v.get(lang)
                if t and t != src:
                    tables[lang][src] = t
    except Exception as e:  # pragma: no cover
        print(f"[i18n] load i18n_data.json failed: {e}")

    # 2) Bản dịch tuyển chọn cho chrome (menu/toolbar/tab/category/splash…)
    #    — ưu tiên đè lên dữ liệu auto để đảm bảo chất lượng & nhất quán.
    for src, en, vi, zh in _ENTRIES:
        if en != src:
            tables["en"][src] = en
        if vi != src:
            tables["vi"][src] = vi
        if zh != src:
            tables["zh"][src] = zh
    return tables


TRANSLATIONS: Dict[str, Dict[str, str]] = _build()
