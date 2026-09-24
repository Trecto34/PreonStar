#!/bin/sh
# build + run the vulkan-kernels gate at the currently checked-out commit
cd /home/server/q36 || exit 125
make -j$(nproc) q36_test > /tmp/r1/build.log 2>&1 || exit 125
timeout 900 ./q36_test --vulkan-kernels > /tmp/r1/test.log 2>&1
rc=$?
if grep -q "q36_test.c:4958" /tmp/r1/test.log; then exit 1; fi
[ "$rc" = 0 ] && exit 0
exit 125
