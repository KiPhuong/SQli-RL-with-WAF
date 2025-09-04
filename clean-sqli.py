import re

def clean_sql_line(line: str) -> str:
    """
    - Xóa mọi ký tự trước và bao gồm cả 'id = 1'.
    - Chuẩn hóa khoảng trắng thừa.
    - Loại bỏ space dư thừa quanh dấu câu (, ) = + - * ...).
    """
    # B1: Tìm 'id = 1' (cho phép nhiều khoảng trắng, không phân biệt hoa thường)
    match = re.search(r"id\s*=\s*1", line, flags=re.IGNORECASE)
    if match:
        line = line[match.end():]  # cắt bỏ cả 'id = 1' và phần trước

    # B2: Chuẩn hóa khoảng trắng (nhiều space thành 1 space)
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def process_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = [clean_sql_line(line) for line in lines]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))

    print(f"✅ Đã xử lý xong! Kết quả được lưu vào {output_file}")


if __name__ == "__main__":
    # Ví dụ: file sqli-misc.txt chứa danh sách SQL injection
    process_file("sqli-misc.txt", "output.txt")
