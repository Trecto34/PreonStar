"""Print absolute file offset, ggml type and dims of tensors matching a pattern.
usage: python3 -I gguf_off.py model.gguf pattern"""
import struct, sys

def tensors(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    f.read(4)
    nt, nkv = struct.unpack('<QQ', f.read(16))
    rs = lambda: f.read(struct.unpack('<Q', f.read(8))[0]).decode('utf-8', 'replace')
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    align = 32
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            at = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
            if at in sz: f.read(sz[at] * n)
            else:
                for _ in range(n): rv(at)
            return None
        b = f.read(sz[t])
        return int.from_bytes(b, 'little') if t in (4, 5, 10, 11) else None
    for _ in range(nkv):
        k = rs(); v = rv(struct.unpack('<I', f.read(4))[0])
        if k == 'general.alignment': align = v
    out = []
    for _ in range(nt):
        n = rs(); nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
        t, off = struct.unpack('<IQ', f.read(12))
        out.append((n, t, dims, off))
    base = (f.tell() + align - 1) // align * align
    return [(n, t, d, base + o) for n, t, d, o in out]

if __name__ == '__main__':
    for n, t, d, o in tensors(sys.argv[1]):
        if sys.argv[2] in n: print(n, t, 'x'.join(map(str, d)), o)
