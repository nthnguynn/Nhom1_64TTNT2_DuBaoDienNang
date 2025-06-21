# preprocess_all.py
import os
from glob import glob
import pandas as pd


# ---------- A. Hàm phụ lấy timestamp đầu của 1 file ----------
def get_first_timestamp(file_path: str) -> pd.Timestamp | None:
    """
    Đọc đúng 1 dòng đầu (nrows=1), parse timestamp để biết mốc
    sớm nhất của file. Trả về pandas.Timestamp hoặc None nếu lỗi.
    """
    try:
        df_head = pd.read_csv(file_path, nrows=1, encoding="utf-8")
        if "timestamp" not in df_head.columns:
            return None
        ts = pd.to_datetime(
            df_head["timestamp"].iloc[0],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        return ts if pd.notna(ts) else None
    except Exception:
        return None


# ---------- B. Xử lý 1 file ----------
def process_one_csv(file_path: str, resample_rule: str = "1min"):
    df = pd.read_csv(file_path, encoding="utf-8")

    if "timestamp" not in df.columns:
        raise KeyError(f"'timestamp' column không tồn tại trong {file_path}")

    # Parse + sort
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Optional: cảnh báo dòng lỗi timestamp
    bad = df["timestamp"].isna().sum()
    if bad:
        print(f"[!] {bad} dòng timestamp lỗi trong {os.path.basename(file_path)}")

    # Resample
    df = df.set_index("timestamp")
    df_res = df.resample(resample_rule).mean().reset_index()

    return df.reset_index(), df_res


# ---------- C. Xử lý toàn thư mục ----------
def process_folder(src_folder: str,
                   dst_folder: str | None = None,
                   resample_rule: str = "1min"):
    if dst_folder is None:
        dst_folder = os.path.join(src_folder, "processed")
    os.makedirs(dst_folder, exist_ok=True)

    # 1. Lấy tất cả file .csv
    paths = glob(os.path.join(src_folder, "*.csv"))
    if not paths:
        print("Không tìm thấy file .csv nào!")
        return

    # 2. Sort danh sách file theo timestamp đầu tiên
    file_meta = []
    for p in paths:
        ts0 = get_first_timestamp(p)
        file_meta.append((p, ts0))

    # Sắp xếp: timestamp None sẽ đẩy xuống cuối
    paths_sorted = [
        p for p, _ in sorted(
            file_meta, key=lambda x: (x[1] is None, x[1])
        )
    ]

    combined = []

    for fp in paths_sorted:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"--- Đang xử lý {name}")

        df_clean, df_res = process_one_csv(fp, resample_rule)

        # Lưu 2 phiên bản
        df_clean.to_csv(os.path.join(dst_folder, f"{name}_clean.csv"),
                        index=False)
        df_res.to_csv(os.path.join(dst_folder, f"{name}_{resample_rule}.csv"),
                      index=False)

        combined.append(df_res.assign(source=name))

    # 3. Gộp tất cả, sort lần cuối
    if combined:
        all_res = pd.concat(combined, ignore_index=True)
        all_res = all_res.sort_values("timestamp").reset_index(drop=True)

        all_out = os.path.join(dst_folder, f"all_files_{resample_rule}.csv")
        all_res.to_csv(all_out, index=False)
        print(f"==> Đã lưu file gộp: {all_out}")


# ---------- D. Chạy trực tiếp ----------
if __name__ == "__main__":
    SRC = r"G:\My Drive\Data_Time"   # thư mục gốc
    process_folder(SRC)
