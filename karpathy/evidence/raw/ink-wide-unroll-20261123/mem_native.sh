#!/usr/bin/env bash
# Bytes for the running p36-server: drm-resident-* (dedup by drm-client-id) + anon RSS. For bench.py's watchdog.
pid=$(pgrep -x p36-server | head -1); [ -z "$pid" ] && { echo 0; exit 0; }
gpu=$(cat /proc/$pid/fdinfo/* 2>/dev/null | awk '/^drm-client-id/{c=$2} /^drm-resident-/{k=c":"$1; if(!(k in s)){s[k]=1; v=$2; u=$3; m=(u=="KiB")?1024:(u=="MiB")?1048576:1; t+=v*m}} END{printf "%d", t+0}')
anon=$(awk '/^RssAnon/{print $2*1024}' /proc/$pid/status)
echo $(( gpu + anon ))
