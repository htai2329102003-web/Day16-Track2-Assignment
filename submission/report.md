# LAB 16 Report — Cloud AI Environment Setup

1. Tôi triển khai bài lab trên AWS tại region `ap-southeast-1` (Singapore), sử dụng Bastion Host và Compute Node CPU `t3.medium`. Luồng chính của bài yêu cầu chạy LightGBM trên CPU instance.

2. Dataset sử dụng là Kaggle Credit Card Fraud Detection (`creditcard.csv`), gồm 284,807 dòng và 31 cột. Target `Class` có 284,315 giao dịch bình thường và 492 giao dịch gian lận.

3. Dữ liệu được chia theo tỷ lệ Train/Validation/Test = `60/20/20`, với `seed = 42`. Số mẫu tương ứng là 170,883 train, 56,962 validation và 56,962 test. Validation được dùng cho early stopping, còn test chỉ dùng để đánh giá cuối.

4. Thời gian load dữ liệu là `2.417 giây`, thời gian training LightGBM là `3.047 giây`, và `best_iteration = 18`.

5. Kết quả trên tập test: `AUC-ROC = 0.933715`, `Accuracy = 0.986956`, `F1-Score = 0.195016`, `Precision = 0.109091`, `Recall = 0.918367`. Accuracy cao nhưng Precision/F1 thấp hơn do dataset mất cân bằng mạnh, chỉ có 492 giao dịch fraud.

6. Inference latency trung bình cho 1 dòng là khoảng `1.2851 ms`. Với batch 1,000 dòng, throughput đo được khoảng `551,178 rows/second`. Benchmark có warm-up trước khi đo inference.

7. Khi benchmark chạy, tiến trình `python3` sử dụng khoảng `100% CPU` trên một core. Máy có khoảng `3.7 GiB RAM`; tại thời điểm đo bằng `free -h`, khoảng `240 MiB` đang được sử dụng và khoảng `3.2 GiB` còn available. README yêu cầu quan sát CPU, RAM và Network bằng `top`, `free -h`, `ip -s link`.

8. Network được quan sát trên interface `ens5` bằng `ip -s link`. Tại thời điểm chụp, RX khoảng `277,232,467 bytes / 194,407 packets`, TX khoảng `3,349,616 bytes / 18,243 packets`. Đây là số byte/gói tích lũy, không phải tốc độ mạng tức thời.

9. AWS Billing tại khoảng `02:35 ngày 03/10/2026` vẫn hiển thị `No data` và `Estimated grand total: USD 0.00`. Danh tính tài khoản đã được xác nhận bằng `aws sts get-caller-identity`. Billing chưa cập nhật tại thời điểm chụp nên số `USD 0.00` không được xem là xác nhận chi phí cuối cùng.

10. Sau khi tải `benchmark.py` và `benchmark_result.json` về Windows, tôi chạy `terraform destroy`. Terraform báo `Destroy complete! Resources: 27 destroyed.` và `terraform state list` không còn trả về resource nào. `benchmark_output_previous_run.txt` là log của một lần chạy trước; ảnh terminal và JSON trong bộ nộp thể hiện lần chạy được báo cáo ở trên. README yêu cầu dọn hạ tầng sau khi hoàn thành để tránh tiếp tục phát sinh chi phí, đặc biệt là NAT Gateway.
