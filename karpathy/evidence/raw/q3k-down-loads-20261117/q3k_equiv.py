# Old per-element Q3_K reads vs the new 11-word preload: same (sc, qv) for every lane/step.
import random
def u8(blk, off): return blk[off]
def u16(blk, off2): return blk[2*off2] | (blk[2*off2+1] << 8)   # wb[] index, little endian
def old(blk, v_im, l0):
    out=[]; y_off=128*v_im+l0
    for p in range(8):
        n=y_off+16*p; group=n>>4
        slo=u8(blk,96+(group&7)); shi=u8(blk,104+(group&3))
        sc=(((slo if group<8 else slo>>4)&15)|(((shi>>(2*(group>>2)))&3)<<4))-32
        qv=[]
        for z in range(2):
            nn=n+z; half=nn>>7; g16=(nn&127)>>4
            q=u8(blk,32+half*32+(nn&31)); mask=1<<(half*4+(g16>>1)); lo=(q>>(2*(g16>>1)))&3
            qv.append(lo-(0 if (u8(blk,nn&31)&mask) else 4))
        out.append((sc,tuple(qv)))
    return out
def new(blk, v_im, l0):
    qo=(32+32*v_im+l0)>>1; ho=l0>>1; so=96>>1
    q4=u16(blk,qo)|(u16(blk,qo+8)<<16); h4=u16(blk,ho)|(u16(blk,ho+8)<<16)
    slo0=u16(blk,so)|(u16(blk,so+1)<<16); slo1=u16(blk,so+2)|(u16(blk,so+3)<<16); shi4=u16(blk,so+4)|(u16(blk,so+5)<<16)
    out=[]
    for p in range(8):
        group=8*v_im+p
        slo=((slo0 if p<4 else slo1)>>(8*(p&3)))&255; shi=(shi4>>(8*(p&3)))&255
        sc=(((slo if group<8 else slo>>4)&15)|(((shi>>(2*(group>>2)))&3)<<4))-32
        qv=[]
        for z in range(2):
            bs=16*(p&1)+8*z; q=(q4>>bs)&255; mask=1<<(v_im*4+(p>>1)); lo=(q>>(2*(p>>1)))&3
            qv.append(lo-(0 if (((h4>>bs)&255)&mask) else 4))
        out.append((sc,tuple(qv)))
    return out
random.seed(1); cases=0
for t in range(3000):
    blk=[random.randrange(256) for _ in range(110)]
    for itid in range(16):
        v_im=itid>>3; l0=2*(itid&7)
        assert old(blk,v_im,l0)==new(blk,v_im,l0),(t,itid); cases+=1
# every element of the block is covered exactly once by the 16 lanes
cov=sorted(128*(i>>3)+2*(i&7)+16*p+z for i in range(16) for p in range(8) for z in range(2))
assert cov==list(range(256))
print("equivalent on",cases,"lane-blocks; coverage ok")
