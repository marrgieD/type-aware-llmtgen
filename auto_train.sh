#!/bin/bash

# 只要没跑完（或者跑够50次），就一直死循环重试
for ((i=1; i<=50; i++)); do
    echo "=================================================="
    echo "【第 $i 次复活】正在申请 GPU 资源..."
    echo "=================================================="

    # 这里是你原本的命令
    # srun -p nvidia --gres=gpu:3 --mem=150G  --time=00:10:00 bash ./scripts/2Stage_train.sh scripts/experiments/GPT2/2stage/adult/stage2_standard.sh
    # 2. 加上 --overwrite_output_dir False (千万别删旧档)
    # 3. 加上 --output_dir 固定路径 (确保每次都能找到旧档)
    
    srun -p nvidia -t 00:10:00 --gres=gpu:3 --mem=150G \
    bash scripts/2Stage_train.sh scripts/experiments/GPT2/2stage/adult/stage2_standard.sh \
        --output_dir ./runs/stage2_final \
        --overwrite_output_dir False

    # 检查退出码：如果 Python 正常跑完退出（退出码0），那就停止循环
    if [ $? -eq 0 ]; then
        echo "恭喜！训练全部完成！"
        break
    fi

    echo "任务因超时被杀，5秒后自动重新申请..."
    sleep 5
done


srun -p nvidia -t 00:10:00 --gres=gpu:3 --mem=150G