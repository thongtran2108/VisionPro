# AOI Vision Pro — PySide6

## Cài đặt
```bash
pip install PySide6 opencv-python numpy Pillow scipy
python main.py
```

## Cấu trúc Project
```
aoi_project/
├── main.py                    # Entry point
├── config/
│   └── settings.py            # App settings & constants
├── core/
│   ├── inspection_engine.py   # Engine xử lý AOI
│   ├── tool_registry.py       # Đăng ký các tool kiểm tra
│   └── flow_graph.py          # Quản lý pipeline node
├── ui/
│   ├── main_window.py         # Cửa sổ chính
│   ├── canvas_view.py         # Vùng kéo thả node (GraphicsView)
│   ├── node_item.py           # Node widget trên canvas
│   ├── tool_library.py        # Panel thư viện tool (trái)
│   ├── properties



 git clone -b master https://github.com/thongtran2108/VisionPro.git


phần pathmax
kéo phần kéo x,y theo tham chiếu
thêm phần vẽ region: polygon, tròn, elip