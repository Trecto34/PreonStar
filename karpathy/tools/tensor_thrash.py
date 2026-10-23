
import json, struct, collections, sys

MODEL = "/home/server/models/llm/MiniCPM-V-4.7-35B-A3B.APEX-I-NanoPlus-V2.1.gguf"
f = open(MODEL, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
assert magic == 0x46554747, hex(magic)

def rd_str():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_val(t):
    if t == 8: rd_str()
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n): skip_val(et)
    else:
        size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
        f.read(size)

kv = {}
for _ in range(n_kv):
    k = rd_str(); t = struct.unpack("<I", f.read(4))[0]
    if t == 8: kv[k] = rd_str()
    else: skip_val(t)

align = kv.get("general.alignment", 32)
infos = []
for _ in range(n_tensors):
    name = rd_str()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
    tt = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    infos.append((name, dims, tt, off))

data_start = f.tell()
data_start = (data_start + align - 1) // align * align
f.seek(0, 2); fsize = f.tell()

# tensor byte extents
blk = {0:1,1:1,2:2,3:2,6:4,7:4,8:4,9:4,10:4,11:4,12:4,13:4,14:4,
       15:2,16:2,17:2,18:2,19:2,20:2,21:2,22:2,23:2,24:1,25:1,26:1,
       27:1,28:1,29:1,30:1,31:1}
# gguf type -> (block elems, bytes per block)
tt_info = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),
           9:(32,36),10:(256,84),11:(256,110),12:(256,144),13:(256,176),
           14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),
           19:(256,50),20:(256,84),21:(256,110),22:(256,82),23:(256,136),
           24:(1,1),25:(1,2),26:(32,18),27:(32,20),28:(32,24),29:(256,66),30:(256,74)}
tens = []
for name, dims, tt, off in infos:
    nel = 1
    for d in dims: nel *= d
    be, bb = tt_info.get(tt, (1,1))
    nbytes = (nel // be) * bb if be > 1 else nel * bb
    tens.append((data_start + off, nbytes, name, tt, nel))
tens.sort()

def find(off):
    lo, hi = 0, len(tens)-1
    while lo <= hi:
        mid = (lo+hi)//2
        s, n, name, tt, nel = tens[mid]
        if off < s: hi = mid-1
        elif off >= s+n: lo = mid+1
        else: return name, tt, nel, n
    return None, None, None, None

d = json.load(open("/tmp/thrash.json"))
back = d["back"]; evicted = d["evicted"]
agg_back = collections.Counter(); agg_evi = collections.Counter()
for i in back:
    nm, *_ = find(i*4096)
    agg_back[nm or "?"] += 1
for i in evicted:
    nm, *_ = find(i*4096)
    agg_evi[nm or "?"] += 1

print(f"tensors={len(tens)} data_start={data_start} file={fsize}")
print(f"refaulted pages total={len(back)}  ({len(back)*4096/2**20:.1f} MiB over 9s)")
print("\nTop refaulted tensors (pages -> MiB):")
for nm, c in agg_back.most_common(12):
    print(f"  {c*4096/2**20:8.1f} MiB  {c:7d}p  {nm}")
print("\nTop evicted tensors:")
for nm, c in agg_evi.most_common(8):
    print(f"  {c*4096/2**20:8.1f} MiB  {c:7d}p  {nm}")
