#!/usr/bin/env bash
# srv.sh start <bindir> <model> | srv.sh stop   — one p36-server on 127.0.0.1:8123
case "$1" in
start) pgrep -x p36-server >/dev/null && "$0" stop
       setsid nohup "$2/p36-server" -m "$3" --host 127.0.0.1 --port 8123 -c 4096 \
         > /home/server/q36-ab/logs/srv-$(basename "$2").log 2>&1 < /dev/null & ;;
stop)  pkill -x p36-server; for i in $(seq 60); do pgrep -x p36-server >/dev/null || exit 0; sleep 1; done
       echo "p36-server did not exit in 60s" >&2; exit 1 ;;
esac
