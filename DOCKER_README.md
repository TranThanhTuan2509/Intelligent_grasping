# 🦾 FreeGrasp Demo — Hướng dẫn chạy bằng Docker

## Yêu cầu

| GPU | VRAM | Chế độ |
|-----|------|--------|
| RTX 4080 / 3090 / A4000 | ≥ 16GB | Mặc định (float16) |
| **RTX 3060 / 4060 / 3070** | **12GB** | **4-bit mode** (xem bên dưới) |

| Yêu cầu | Phiên bản |
|---------|-----------|
| GPU | NVIDIA RTX 4080 (≥ 16GB VRAM) hoặc tương đương |
| Driver NVIDIA | ≥ 545 |
| Docker | ≥ 24.0 |
| NVIDIA Container Toolkit | Xem bước 1 |

---

## Bước 1 — Cài NVIDIA Container Toolkit (nếu chưa có)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Kiểm tra:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Bước 2 — Tải checkpoint GraspNet

Tải file `checkpoint_fgc.tar` từ Google Drive:  
👉 https://drive.google.com/drive/folders/1w5cZAfY9h0O9908y9YvL88KIPL_7J7EF

Đặt vào thư mục `logs/`:
```
FreeGrasp_code/
└── logs/
    └── checkpoint_fgc.tar   ← đặt vào đây
```

---

## Bước 3 — Cấu hình API Key

File `.env` đã được cấu hình sẵn. Nếu cần đổi key, chỉnh sửa file `.env`:

```env
OPENAI_API_KEY=<your_gemini_api_key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-2.5-pro
```

---

## Bước 4 — Build và chạy

```bash
cd FreeGrasp_code

# Build Docker image (lần đầu ~15–30 phút do tải model)
docker compose build

# Chạy demo
docker compose up
```

Mở trình duyệt: **http://localhost:7860**

> **Lưu ý**: Lần đầu chạy sẽ tải model Molmo-7B (~14GB) tự động từ HuggingFace. Model được cache trong Docker volume để các lần sau không cần tải lại.

---

## Bước 5 — Dùng demo

1. Nhập **Textual Prompt** (ví dụ: "pick the red mug")
2. Upload **RGB image** (`.png`, `.jpg`)
3. Upload **Depth file** (`.npz` — có trong `data/real_examples/hard/1/`)
4. Nhấn **Submit**

Kết quả: keypoints từ Molmo, mask từ LangSAM, grasp pose 3D từ GraspNet.

---

## Dừng và xóa container

```bash
# Dừng
docker compose down

# Dừng và xóa cả dữ liệu cache (cần tải lại model)
docker compose down -v
```
