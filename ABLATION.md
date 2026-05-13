# Ablation Study — LVC Module

## Tổng quan các variant

| Variant | File model | Mô tả |
|---|---|---|
| `baseline` | `src/models/Net_train_module.py` | LiNo-UniPS gốc, không có LVC |
| `lvc_loss` | `src/models/Net_lvc_loss.py` | Version A — chỉ LVC loss weighting |
| `lvc_feat` | `src/models/Net_lvc_feat.py` | Version B — chỉ LVC feature scaling (PMA input) |
| `lvc_full` | `src/models/Net_lvc_full.py` | Version C — kết hợp cả A và B (proposed) |

---

## Chạy training

Tất cả các variant đều chạy qua **một file duy nhất**: `train.py`.  
Chọn variant bằng flag `--variant`.

```bash
# Baseline
python train.py \
    --variant baseline \
    --data_root /path/to/data \
    --save_dir checkpoints/baseline

# Version A — LVC loss only
python train.py \
    --variant lvc_loss \
    --data_root /path/to/data \
    --save_dir checkpoints/lvc_loss

# Version B — LVC feature scaling only
python train.py \
    --variant lvc_feat \
    --data_root /path/to/data \
    --save_dir checkpoints/lvc_feat

# Version C — Full LVC (proposed method)
python train.py \
    --variant lvc_full \
    --data_root /path/to/data \
    --save_dir checkpoints/lvc_full
```

---

## Tham số LVC (chỉ áp dụng cho lvc_loss / lvc_feat / lvc_full)

Chỉnh tại **command line** khi gọi `train.py`, hoặc sửa default trong `lvc.py`:

| Tham số | Flag | Default | Ý nghĩa |
|---|---|---|---|
| `alpha` | `--lvc_alpha` | `10.0` | Độ nhạy của hàm exponential. Tăng → C(p) bão hoà nhanh hơn với CV² nhỏ |
| `w_min` | `--lvc_w_min` | `0.1` | Weight tối thiểu cho pixel ít lighting variation. `0` = ignore hoàn toàn, `1` = bằng baseline |
| `per_channel` | `--lvc_per_channel` / `--no-lvc_per_channel` | `True` (max channel) | Cách tính CV²: `True` = lấy max CV² qua 3 kênh RGB; `False` = trung bình grayscale |

**Chi tiết hai mode tính CV²:**

- **Max channel** (`--lvc_per_channel`, default): $\text{CV}^2(p) = \max_{c \in \{R,G,B\}} \frac{\text{Var}_k[I_c(p,k)]}{\mathbb{E}_k[I_c(p,k)]^2 + \varepsilon}$ — nhạy hơn với specular highlight trên một kênh đơn lẻ.
- **Grayscale** (`--no-lvc_per_channel`): $\text{CV}^2(p) = \frac{\text{Var}_k[\bar{I}(p,k)]}{\mathbb{E}_k[\bar{I}(p,k)]^2 + \varepsilon}$ với $\bar{I} = \frac{1}{3}(R+G+B)$ — ổn định hơn với nhiễu kênh màu.

Ví dụ thay đổi tham số:

```bash
python train.py \
    --variant lvc_full \
    --lvc_alpha 5.0 \
    --lvc_w_min 0.2 \
    --data_root /path/to/data \
    --save_dir checkpoints/lvc_full_a5_wmin02

# Dùng grayscale mean thay vì max channel
python train.py \
    --variant lvc_full \
    --no-lvc_per_channel \
    --data_root /path/to/data \
    --save_dir checkpoints/lvc_full_grayscale
```

---

## Vị trí file quan trọng

```
src/models/
├── module/
│   └── lvc.py              # LVCModule — công thức CV², C(p), C_k(p), w(p)
├── Net_train_module.py     # Baseline model
├── Net_lvc_loss.py         # Version A
├── Net_lvc_feat.py         # Version B
└── Net_lvc_full.py         # Version C
train.py                    # Entry point duy nhất cho tất cả variant
```

---

## Output

- **Checkpoints**: lưu tại `--save_dir`, tên file có prefix variant (ví dụ `lvc_full_epoch=05_val_loss=0.0312.ckpt`)
- **TensorBoard logs**: lưu tại `<save_dir>/lightning_logs/<variant>/`

Xem log:
```bash
tensorboard --logdir checkpoints/
```

Các variant sẽ hiện thành các run riêng biệt trong TensorBoard để so sánh trực tiếp.
