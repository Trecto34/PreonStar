#!/bin/sh
cp -f /tmp/r2/new.spv /home/server/p36/vulkan/moe_down_q2k_sum_decode_iq2s.spv
exec /home/server/p36/p36-bench "$@"
