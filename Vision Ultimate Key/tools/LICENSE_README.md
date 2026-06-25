# Kích hoạt (License) & Đăng nhập Admin — Hướng dẫn

Hệ thống gồm 2 lớp độc lập:

| Lớp | Mục đích | Cơ chế |
|-----|----------|--------|
| **License** | Cho phép **bản/máy này** chạy (chống lậu) | File `license.key` ký số, **khoá theo máy** |
| **Login** | Ai được **Sửa & Chạy** tool | Tài khoản admin `VisionUltimate` (đổi được mật khẩu) |

Khởi động: app kiểm tra license → chưa hợp lệ thì hiện cổng **Kích hoạt** (khoá hẳn).
Sau khi vào app: mặc định **khoá** Sửa/Chạy — bấm **🔒 Đăng nhập** mới mở khoá.

---

## A. Đăng nhập Admin

- Tài khoản mặc định: **`VisionUltimate`** / **`8888`**.
- Đổi mật khẩu: menu **Account → Đổi mật khẩu**.
- Mật khẩu lưu dạng **PBKDF2-SHA256 có salt** trong QSettings (không lưu thô).
- Quên mật khẩu? Gọi `AuthManager().reset_to_default()` (về lại `VisionUltimate`/`8888`).
- Máy đã cài bản cũ (tên `VUTM`) sẽ **tự đổi** sang `VisionUltimate`, giữ nguyên mật khẩu.

Chưa đăng nhập sẽ bị khoá: kéo thêm tool, di chuyển/nối/xoá node, sửa params
trong Properties, mở cửa sổ chi tiết node, và **Run**. Vẫn xem được pipeline/ảnh.

---

## B. License — kích hoạt (KHÔNG cần keypair)

> Mô hình mới dùng **HMAC bí mật dùng chung** (`_LICENSE_SECRET` trong
> `core/licensing.py`). App và Activator build từ **cùng source** nên **tự
> khớp** — KHÔNG còn `.pem`, KHÔNG `--init`, KHÔNG "build rồi nhúng key".
> Chỉ cần **build app + activator một lần từ source này**, sau đó kích hoạt
> máy nào cũng được.

### Cách 1 (DỄ NHẤT) — Phần mềm Activator GUI (`tools/activator.py`)
Mở app → **chọn thời hạn** (30/90/180/365 ngày · vĩnh viễn · tuỳ chỉnh số
ngày · theo ngày hết hạn) → bấm **Kích hoạt**. App Vision tự nhận.
```bash
python tools/activator.py          # chạy từ source
```
Đóng gói thành 1 exe độc lập (double-click):
```cmd
tools\build_activator.bat          # -> dist\VisionProActivator.exe
```
- **Active máy này:** để mặc định “Máy này” → chọn ngày → Kích hoạt → mở
  VisionPro là chạy (cả `.exe` lẫn chạy từ code đều nhận, vì license lưu ở
  `%APPDATA%\VisionPro\`).
- **Cấp cho máy khác:** chọn “Máy khác”, dán **Machine ID** → app tạo
  `license.key` để gửi đi. Có nút **Gỡ kích hoạt** / **Làm mới** sẵn.
- ⚠ Đây là CÔNG CỤ NHÀ PHÁT HÀNH (chứa bí mật ký license) — đừng phát tán bừa.

### Cách 2 — CLI một-bước (`tools/activate.py`)
Chạy trực tiếp trên máy cần kích hoạt → tự đọc Machine ID → ký + cài → xong.
```bash
python tools/activate.py                       # active máy hiện tại (vĩnh viễn)
python tools/activate.py --customer "Cong ty ABC" --edition Pro
python tools/activate.py --expires 2027-12-31  # hết hạn theo ngày
python tools/activate.py --duration-days 365   # số ngày kể từ active
python tools/activate.py --any                 # license không khoá máy
```
Windows: double-click **`tools/active.bat`**. Không cần file `.pem` nào.

### Cách 3 — gửi file license cho khách tự import (`tools/license_keygen.py`)
Khi không tiện chạy trên máy khách: khách gửi **Machine ID** (đọc trong cổng
Kích hoạt) → bạn tạo file rồi gửi lại:
```bash
# Hết hạn theo ngày tuyệt đối:
python tools/license_keygen.py --issue \
    --machine 1A2B-3C4D-5E6F-7890 --customer "Cong ty ABC" \
    --expires 2027-12-31 --out ABC_license.key

# HOẶC hết hạn theo số ngày kể từ lúc khách active (chống chỉnh giờ):
python tools/license_keygen.py --issue \
    --machine 1A2B-3C4D-5E6F-7890 --customer "Cong ty ABC" \
    --duration-days 365 --out ABC_license.key

# HOẶC vĩnh viễn: bỏ cả --expires lẫn --duration-days.
```
Khách bấm **“Chọn file license…”** trong cổng Kích hoạt (hoặc chỉ cần **bỏ
`license.key` cạnh file chạy app** → mở app tự active, không cần import).

> ⚠ **Đổi bí mật cho riêng bạn trước khi phát hành thật:** sửa `_LICENSE_SECRET`
> trong `core/licensing.py` thành chuỗi riêng rồi **build lại app + activator**
> (giữ source kín). Đổi xong, mọi license cũ hết hiệu lực.

**Tự test nhanh trên máy của bạn:**
```bash
python tools/activate.py            # xong — mở app là đã active
python tools/activate.py --status   # xem trạng thái / nơi lưu
```

---

## C. Hạn dùng & chống chỉnh đồng hồ

Có **2 kiểu hạn** (đặt khi tạo license, dùng riêng hoặc kết hợp — lấy mốc tới
trước):

| Kiểu | Tham số | Ý nghĩa |
|------|---------|---------|
| Ngày tuyệt đối | `--expires 2027-12-31` | Hết hạn đúng ngày đó |
| Theo số ngày | `--duration-days 30` | 30 ngày **kể từ ngày active** trên máy |
| Vĩnh viễn | (không đặt gì) | Không hết hạn |

**Chống chỉnh đồng hồ (đã sửa bug):** app lưu **mốc ngày cao nhất từng thấy**
(high-water mark) + **ngày active** ở `%APPDATA%\VisionPro\.license_state` (có
HMAC chống sửa tay) và registry `HKCU\Software\VisionPro\State`. Hạn tính theo
`max(ngày_hiện_tại, mốc_cao_nhất)` nên:
- **Chỉnh giờ LÙI → vô tác dụng** (đã hết hạn vẫn hết hạn).
- **Gỡ rồi cài lại cùng license → KHÔNG reset** số ngày (ngày active giữ nguyên).
- Sửa tay file state → HMAC sai → bị bỏ qua.

## Cơ chế bảo vệ (tổng hợp)

- **HMAC-SHA256 bí mật dùng chung**: license ký bằng `_LICENSE_SECRET` (nằm
  trong source/app). Mức "chống sao chép thông thường" — không bảo vệ trước
  người reverse-engineer được exe để moi bí mật. Cần mạnh hơn thì giữ source
  kín + đổi bí mật định kỳ.
- **Khoá máy**: `machine_id` phải khớp vân tay phần cứng (Windows `MachineGuid`).
- **Chống sửa file**: đổi 1 ký tự ⇒ HMAC sai ⇒ từ chối.

## D. Nơi lưu license & cách gỡ (deactivate)

Khi kích hoạt, license được **cài vào hệ thống** (không chỉ là file trong thư
mục). App tìm theo thứ tự:
1. cạnh file chạy (app dir) — `license.key`
2. **`%APPDATA%\VisionPro\license.key`** (Windows) · `~/.config/VisionPro/license.key` (Linux)

> ⚠ Vì vậy **xoá file `license.key` trong thư mục KHÔNG huỷ kích hoạt** — bản
> chính nằm ở `%APPDATA%\VisionPro\`. Đó là lý do app vẫn chạy sau khi xoá.

**Xem đang dùng license ở đâu / còn hạn không:**
```bash
python tools/activate.py --status
```
**Gỡ kích hoạt (xoá license khỏi mọi nơi app đọc) — để test lại cổng Kích hoạt:**
```bash
python tools/activate.py --deactivate
```
(Nhớ **tắt app** trước khi gỡ; app chỉ kiểm tra license lúc khởi động.)
Hoặc xoá tay file `%APPDATA%\VisionPro\license.key` (gõ `%APPDATA%` vào thanh
địa chỉ Explorer → mở thư mục `VisionPro`).

> **Lưu ý khi TEST license có hạn:** `--deactivate` **giữ** mốc ngày + ngày
> active (để chống reset duration). Muốn test lại từ đầu (giả lập máy mới),
> xoá luôn state:
> ```bash
> python tools/activate.py --reset-state
> ```
