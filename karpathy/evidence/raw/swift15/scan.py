import struct,sys,collections,os
N={16:'IQ2_XXS',17:'IQ2_XS',19:'IQ1_S',20:'IQ4_NL',29:'IQ1_M'}
for p in sys.argv[1:]:
    f=open(p,'rb')
    rd=lambda fmt:struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
    def s(): n,=rd('Q'); return f.read(n).decode(errors='replace')
    def val(t):
        sz={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if t in sz: return rd(sz[t])[0]
        if t==8: return s()
        et,n=rd('IQ'); return [val(et) for _ in range(n)]
    try:
        _,ver,nt,nkv=rd('IIQQ'); arch=''
        for _ in range(nkv):
            k=s(); t,=rd('I'); v=val(t)
            if k=='general.architecture': arch=v
        c=collections.Counter()
        for _ in range(nt):
            n=s(); nd,=rd('I'); dims=rd('Q'*nd); ty,=rd('I'); rd('Q')
            if ty in N and nd==2 and 'exps' not in n and 'token_embd' not in n: c[N[ty]]+=1
        print(f"{os.path.basename(p)[:60]:60s} {arch:10s} {dict(c)}")
    except Exception as e: print(p,'ERR',e)
