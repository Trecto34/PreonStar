#!/bin/bash
# usage: run.sh TAG MODEL [server args...]
tag=$1; model=$2; shift 2
cd /home/server/q36-wt/ink-smallbatch
./p36-server -m $model --port 18436 --ctx 8192 "$@" > /home/server/q36-wt/lookup-test/server-$tag.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:18436/v1/models && break; sleep 1; done
python3 -I /home/server/q36-wt/lookup-test/client.py 18436 $tag /home/server/q36-wt/lookup-test/out-$tag
kill $pid; wait $pid 2>/dev/null
