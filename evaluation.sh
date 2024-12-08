categories=("공학" "자연" "인문" "사회" "기타")

for category in "${categories[@]}"; do
    echo "Running evaluation for category: $category"
    CUDA_VISIBLE_DEVICES=0 python3.8 evaluation.py --category "$category"
done