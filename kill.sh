PIDS=$(pgrep -f $1.py)

if [ -z "$PIDS" ]; then
    echo "没有找到 $1.py 进程"
else
    echo "找到 $1.py 进程: $PIDS"
    kill $PIDS
    echo "已发送终止信号"
fi
