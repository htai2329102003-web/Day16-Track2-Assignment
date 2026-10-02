#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU LightGBM benchmark node"

apt-get update -y
apt-get install -y python3 python3-pip

pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

mkdir -p /home/ubuntu/ml-benchmark
chown ubuntu:ubuntu /home/ubuntu/ml-benchmark

cat > /home/ubuntu/ml-benchmark/benchmark.py <<'BENCHMARK_PY'
${benchmark_script}
BENCHMARK_PY
chown ubuntu:ubuntu /home/ubuntu/ml-benchmark/benchmark.py
chmod 755 /home/ubuntu/ml-benchmark/benchmark.py

echo "CPU environment ready: LightGBM packages installed; benchmark.py is in /home/ubuntu/ml-benchmark."
