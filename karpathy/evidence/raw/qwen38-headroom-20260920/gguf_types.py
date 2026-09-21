import struct, sys, collections

f = open('/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf', 'rb')
magic, ver = struct.unpack('<4sI', f.read(8))
n_tensors, n_kv = struct.unpack('<QQ', f.read(16))

def rd_str():
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')

def skip_val(t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        rd_str()
    elif t == 9:
        et, = struct.unpack('<I', f.read(4))
        n, = struct.unpack('<Q', f.read(8))
        for _ in range(n):
            skip_val(et)
    else:
        f.read(sizes[t])

for _ in range(n_kv):
    rd_str()
    t, = struct.unpack('<I', f.read(4))
    skip_val(t)

# ggml block sizes: (block_elems, bytes)
BLK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 34), 20: (256, 18), 21: (256, 26), 22: (256, 82),
    23: (256, 136),
}
NAME = {18: 'IQ3_XXS', 23: 'IQ4_XS', 13: 'Q5_K', 12: 'Q4_K', 14: 'Q6_K', 0: 'F32',
        1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K'}

by = collections.Counter()
cnt = collections.Counter()
tot = 0
for _ in range(n_tensors):
    name = rd_str()
    nd, = struct.unpack('<I', f.read(4))
    dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
    t, = struct.unpack('<I', f.read(4))
    off, = struct.unpack('<Q', f.read(8))
    elems = 1
    for d in dims:
        elems *= d
    be, bb = BLK.get(t, (1, 4))
    nb = (elems // be) * bb
    by[t] += nb
    cnt[t] += 1
    tot += nb

for t, nb in by.most_common():
    print(f"{NAME.get(t, 'type%d' % t):8s} {nb/1e9:7.3f} GB  {100*nb/tot:5.1f}%  tensors={cnt[t]}")
print(f"total {tot/1e9:.3f} GB")
