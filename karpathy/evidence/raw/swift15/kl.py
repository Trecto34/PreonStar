import json,math,sys,os
def lp(p):
    lg=json.load(open(p))['logits']; mx=max(lg); l=mx+math.log(sum(math.exp(x-mx) for x in lg)); return [x-l for x in lg]
ref=sys.argv[1]
for f in sorted(os.listdir(ref)):
    r=lp(os.path.join(ref,f)); ra=max(range(len(r)),key=r.__getitem__)
    out=[f]
    for d in sys.argv[2:]:
        q=lp(os.path.join(d,f)); kl=sum(math.exp(a)*(a-b) for a,b in zip(r,q)); qa=max(range(len(q)),key=q.__getitem__)
        out.append(f"{os.path.basename(d)}: KL={kl:.4f} top1={'same' if qa==ra else 'DIFF'}")
    print('  '.join(out))
