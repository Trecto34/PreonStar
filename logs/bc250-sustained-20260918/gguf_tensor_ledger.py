#!/usr/bin/env python3
"""Read only the GGUF header and summarize logical tensor bytes."""
import collections
import re
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
         10: 8, 11: 8, 12: 8}
TENSOR = {0: (1, 4), 30: (1, 2), 42: (64, 18)}

def read_u(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return struct.unpack('<' + fmt, b)[0]

def read_str(f):
    return f.read(read_u(f, 'Q')).decode('utf-8')

def skip(f, t):
    if t == 8:
        f.seek(read_u(f, 'Q'), 1)
    elif t == 9:
        elem, count = read_u(f, 'I'), read_u(f, 'Q')
        if elem in SIZES:
            f.seek(SIZES[elem] * count, 1)
        else:
            for _ in range(count):
                skip(f, elem)
    else:
        f.seek(SIZES[t], 1)

with open(sys.argv[1], 'rb') as f:
    assert f.read(4) == b'GGUF'
    version = read_u(f, 'I')
    n_tensor, n_meta = read_u(f, 'Q'), read_u(f, 'Q')
    alignment = 32
    for _ in range(n_meta):
        key, typ = read_str(f), read_u(f, 'I')
        if key == 'general.alignment' and typ == 4:
            alignment = read_u(f, 'I')
        else:
            skip(f, typ)
    by_type = collections.Counter()
    by_name = collections.Counter()
    tensors = []
    for _ in range(n_tensor):
        name, dims = read_str(f), read_u(f, 'I')
        shape = tuple(read_u(f, 'Q') for _ in range(dims))
        typ, off = read_u(f, 'I'), read_u(f, 'Q')
        elements = 1
        for x in shape:
            elements *= x
        block, size = TENSOR[typ]
        nbytes = ((elements + block - 1) // block) * size
        by_type[typ] += nbytes
        category = re.sub(r'^blk\.\d+\.', 'blk.*.', name)
        by_name[category] += nbytes
        tensors.append((name, typ, shape, nbytes, off))
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    print(f'version={version} tensors={n_tensor} metadata={n_meta} data_start={data_start} file_bytes={f.seek(0,2)}')
    print('type_bytes', sorted(by_type.items()))
    q2 = [x for x in tensors if x[1] == 42]
    print('q2_count', len(q2), 'q2_bytes', sum(x[3] for x in q2))
    print('q2_payload_code_bytes', sum(x[3] * 16 // 18 for x in q2))
    print('q2_scale_bytes', sum(x[3] * 2 // 18 for x in q2))
    print('q2_except_embedding', sum(x[3] for x in q2 if 'token_embd' not in x[0]))
    print('top_categories')
    for name, size in by_name.most_common(35):
        print(f'{name},{size}')
