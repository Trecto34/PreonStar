import json,sys
a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]))
sa=[s['selected']['id'] for s in a['steps']]; sb=[s['selected']['id'] for s in b['steps']]
same=sum(x==y for x,y in zip(sa,sb)); first=next((i for i,(x,y) in enumerate(zip(sa,sb)) if x!=y),None)
d0=[]
for s,t in zip(a['steps'],b['steps']):
    if s['selected']['id']!=t['selected']['id']: break
    ma={e['token']['id']:e['logit'] for e in s['top_logprobs']}; mb={e['token']['id']:e['logit'] for e in t['top_logprobs']}
    d0.append(max(abs(ma[k]-mb[k]) for k in ma if k in mb))
print(f"tokens match {same}/{len(sa)} first_div={first} max|dlogit| over matched steps={max(d0):.4g} step0={d0[0]:.4g}")
