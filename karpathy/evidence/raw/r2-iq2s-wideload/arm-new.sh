#!/bin/sh
cp -f /tmp/r2/new.spv /home/server/q36/vulkan/moe_down_q2k_sum_decode_iq2s.spv
exec /home/server/q36/q36-bench "$@"
