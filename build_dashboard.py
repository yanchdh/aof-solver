#!/usr/bin/env python3
"""Build interactive aof_dashboard.html"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "equity_169.json")) as f: equity = json.load(f)
with open(os.path.join(HERE, "jackpot_ev.json")) as f: jp_ev = json.load(f)

nash_data = {}
for p in [2, 3, 4]:
    with open(os.path.join(HERE, f"nash_{p}p.json")) as f:
        nash_data[str(p)] = json.load(f)

RANKS = '23456789TJQKA'
CW = [c/1326 for c in ([6]*13 + [4]*78 + [12]*78)]
ST = [sum(equity[i][j] * CW[j] for j in range(169)) for i in range(169)]

def gl(r, c):
    hi, lo = 12-r, 12-c
    if hi == lo: return f"{RANKS[hi]}{RANKS[lo]}"
    if hi > lo: return f"{RANKS[hi]}{RANKS[lo]}s"
    return f"{RANKS[lo]}{RANKS[hi]}o"

# Embed data
html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AOF Solver</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial;background:#1a1a2e;color:#ddd;padding:8px}}
h1{{color:#e94560;text-align:center;font-size:clamp(14px,3vw,18px);margin:4px 0}}
.bar{{background:#16213e;border-radius:8px;padding:7px 10px;margin:4px 0;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.bar label{{font-size:clamp(10px,2vw,12px);color:#888;font-weight:600}}
.bar button,.bar button.tog{{background:#0f3460;color:#ccc;border:1px solid #333;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:clamp(10px,2vw,12px);font-weight:600}}
.bar button.sel{{background:#e94560;color:#fff}}
.bar button.tog{{border:none}}
.bar input[type=range]{{width:clamp(50px,15vw,80px);accent-color:#e94560}}
.bar .val{{font-size:clamp(10px,2vw,11px);color:#4a90d9;min-width:26px}}
.bar .vpi{{width:clamp(36px,10vw,46px);background:#0f3460;color:#4a90d9;border:1px solid #333;border-radius:3px;padding:2px 4px;font-size:clamp(9px,1.8vw,11px);text-align:center}}
.heat{{background:#16213e;border-radius:8px;padding:8px;margin:6px 0;overflow-x:auto}}
.heat h3{{font-size:clamp(11px,2.2vw,14px);color:#ccc;margin-bottom:6px}}
.grid{{border-collapse:collapse}}
.grid td{{width:clamp(28px,6.5vw,46px);height:clamp(22px,5vw,34px);text-align:center;border:1.5px solid #333;font-size:clamp(7px,1.6vw,10px);font-weight:700;line-height:1.2;padding:0;color:#fff;text-shadow:0 0 2px #000}}
.grid th{{width:clamp(12px,3vw,18px);height:clamp(12px,3vw,18px);font-size:clamp(7px,1.6vw,10px);color:#bbb}}
.gain{{background:#16213e;border-radius:8px;padding:8px;margin:6px 0;font-size:clamp(10px,2vw,11px);line-height:1.6}}
.gain .pos{{color:#4a90d9}}.gain .neg{{color:#e94560}}
.gain table{{font-size:clamp(9px,1.8vw,11px)}}
@media(min-width:800px){{
  .heat{{display:inline-block;vertical-align:top}}
}}
@media(max-width:600px){{
  .bar{{padding:5px 8px;gap:4px}}
  .bar button,.bar button.tog{{padding:3px 7px}}
}}
</style></head><body>
<h1>AOF Solver | 8BB | Jackpot ON</h1>

<div class="bar">
  <button id="b2" onclick="setN(2)">2P</button>
  <button id="b3" onclick="setN(3)">3P</button>
  <button id="b4" class="sel" onclick="setN(4)">4P</button>
  <label style="margin-left:8px">抽水</label>
  <input type="range" id="rs" min="0" max="0.5" step="0.01" value="0.20" oninput="uR(this.value)">
  <span class="val" id="rv">0.20</span>
</div>

<div class="bar">
  <label>VPIP</label>
  <span id="sx"></span>
  <button onclick="rst()" style="margin-left:4px">重置</button>
</div>

<div class="bar">
  <label>位置</label>
  <span id="pb"></span>
</div>

<div class="bar">
  <label>前位</label>
  <span id="ox"></span>
  <button onclick="tog()">反选</button>
</div>

<div id="ch"></div>
<div class="gain" id="gx">Computing...</div>
<div style="text-align:center;font-size:10px;color:#555;margin-top:4px">F/J=切位置 | QWE/UIO=前位 | 空格=反选 | 2/3/4=人数</div>

<script>
var EQ={json.dumps(equity)};
var JP={json.dumps(jp_ev)};
var ST={json.dumps(ST)};
var CW={json.dumps(CW)};
var NH={json.dumps(nash_data)};
var RK='{RANKS}';

// Compressed best response engine
function br(nP,po,hi,sA){{
 var N=169,vF=-bl[po],vP=new Array(N).fill(0),pr=[],cp=ec[po];
 for(var i=0;i<hi.length;i++)if(hi[i]==='P')pr.push(i);
 if(po===nP-1){{
  var pt=bl.reduce(function(a,b){{return a+b}},0);
  for(var q of pr.concat(po))pt+=bh(q);pt-=rb;
  if(pr.length===1){{
   var o=pr[0],os=gs(sA,o,hi.slice(0,o));
   for(var i=0;i<N;i++)vP[i]=ev(EQ,i,os,CW)*pt-cp+JP[i];
  }}else{{
   var oss=pr.map(function(o){{return as(gs(sA,o,hi.slice(0,o)),ST,CW)}});
   for(var i=0;i<N;i++){{var s=ST[i]+oss.reduce(function(a,b){{return a+b}},0);vP[i]=(s?s>0?ST[i]/s:1/(pr.length+1):0)*pt-cp+JP[i];}}
  }}
 }}else{{
  var po2=[];for(var q=po+1;q<nP;q++)po2.push(q);
  for(var b=0;b<(1<<po2.length);b++){{
   var su='',w=1,al=pr.concat(po),sk=false;
   for(var k=0;k<po2.length;k++)su+=((b>>k)&1)?'P':'F';
   for(var k=0;k<po2.length;k++){{
    var q=po2[k],qH=hi+'P'+su.substring(0,k);
    var p2=su[k]==='P'?ap(gs(sA,q,qH),CW):1-ap(gs(sA,q,qH),CW);
    if(p2<1e-6){{sk=true;break}}w*=p2;if(su[k]==='P')al.push(q);
   }}
   if(sk)continue;
   var pt2=bl.reduce(function(a,b){{return a+b}},0);
   for(var q of al)pt2+=bh(q);pt2-=rb;
   if(al.length===1){{var pf=pt2-cp;for(var i=0;i<N;i++)vP[i]+=w*pf}}
   else if(al.length===2){{
    var o=al.find(function(q){{return q!==po}}),oH;
    if(o<po)oH=hi.slice(0,o);else{{var k=o-po-1;oH=hi+'P'+su.substring(0,k)}}
    var os=gs(sA,o,oH);
    for(var i=0;i<N;i++)vP[i]+=w*(ev(EQ,i,os,CW)*pt2-cp+JP[i]);
   }}else{{
    var oss2=[];for(var o of al){{if(o===po)continue;var oH2;
    if(o<po)oH2=hi.slice(0,o);else{{var k=o-po-1;oH2=hi+'P'+su.substring(0,k)}}
    oss2.push(as(gs(sA,o,oH2),ST,CW))}}
    for(var i=0;i<N;i++){{var s=ST[i]+oss2.reduce(function(a,b){{return a+b}},0);vP[i]+=w*(((s?ST[i]/s:1/al.length)*pt2-cp)+JP[i])}}
   }}
  }}
 }}
 var br2=new Array(N);for(var i=0;i<N;i++)br2[i]=vP[i]>vF?1:0;
 return {{vP:vP,vF:vF,br:br2}};
}}
function gs(a,p,h){{var i=a[p].ix[h];return i!==undefined?a[p].ss[i]:new Array(169).fill(0.5)}}
function ap(s,cw){{var x=0;for(var i=0;i<169;i++)x+=s[i]*cw[i];return x}}
function as(s,st,cw){{var a=0,b=0;for(var i=0;i<169;i++){{var w=s[i]*cw[i];a+=st[i]*w;b+=w}}return b>1e-10?a/b:0.5}}
function ev(eq,hi,os,cw){{var a=0,b=0;for(var j=0;j<169;j++){{var w=os[j]*cw[j];a+=eq[hi][j]*w;b+=w}}return b>1e-10?a/b:0.5}}
function bh(p){{return ec[p]-bl[p]}}

// State
var N=4,up=0,rb=0.2,dev={{}},pp=[],sv={{l:33,r:38,u:64}};
var bl=[0,0,0.5,1],ec=[8,8,8,8],nm=['UTG','CO','SB','BB'];
var hs=[[''],['F','P'],['FF','FP','PF','PP'],['FFP','FPF','FPP','PFF','PFP','PPF','PPP']];
var hi=[];

function setN(n){{
 N=n;bl=new Array(n).fill(0);bl[n-1]=1;if(n>=2)bl[n-2]=0.5;
 ec=new Array(n).fill(8);ec[n-1]=8;
 if(n===2)nm=['SB','BB'];else if(n===3)nm=['BTN','SB','BB'];else nm=['UTG','BTN','SB','BB'];
 hs=[];hi=[];
 for(var p=0;p<n;p++){{
  var hh=[],ix={{}};
  if(p===0){{hh.push('');ix['']=0}}
  else if(p===n-1){{for(var b=0;b<(1<<(n-1));b++){{var pr='';for(var k=0;k<n-1;k++)pr+=((b>>k)&1)?'P':'F';if(pr.indexOf('P')>=0){{ix[pr]=hh.length;hh.push(pr)}}}}}}
  else{{for(var b=0;b<(1<<p);b++){{var pr='';for(var k=0;k<p;k++)pr+=((b>>k)&1)?'P':'F';ix[pr]=hh.length;hh.push(pr)}}}}
  hs.push(hh);hi.push(ix);
 }}
 if(up>=n)up=0;pp=new Array(up).fill(true);sy();rn();cp();
}}

function ns(p,h){{var k=String(N);if(!NH[k])return new Array(169).fill(0.5);var d=NH[k],i=d.histories[p].indexOf(h);return i>=0?d.strategies[p][i]:new Array(169).fill(0.5)}}
function dv(p,h){{var b=ns(p,h),d=dev[p]||0;if(d===0)return b.slice();return b.map(function(v){{return Math.min(1,Math.max(0,v*(1+d)))}})}}
function bA(){{var a=[];for(var p=0;p<N;p++){{var ss=[];for(var h of hs[p])ss.push(p===up?new Array(169).fill(0.5):dv(p,h));a.push({{ix:hi[p],ss:ss}})}};return a}}
function cH(){{var h='';for(var p=0;p<up;p++)h+=pp[p]?'P':'F';return h}}

function s2p(s){{if(N<4){{if(N===2)return{{l:1,r:1,u:-1}}[s];return{{l:(up+1)%3,r:(up+2)%3,u:-1}}[s]}}var m=[[{{r:3,l:1,u:2}}],[{{r:0,l:2,u:3}}],[{{r:1,l:3,u:0}}],[{{r:2,l:0,u:1}}]];return m[up][s]}}
function sy(){{dev={{}};for(var s of['l','r','u']){{var p=s2p(s);if(p>=0&&p<N&&p!==up){{var v=sv[s]||33,nv=Math.round(ap(ns(p,hs[p][0]),CW)*100);dev[p]=v/nv-1}}}}}}

function rn(){{
 var b='';for(var p=0;p<N;p++)b+='<button class="'+(p===up?'sel':'')+'" onclick="sP('+p+')">'+nm[p]+'</button> ';
 document.getElementById('pb').innerHTML=b;

 var sx='';var sl={{l:'左',r:'右',u:'上'}};
 var myVpip=Math.round(ap(ns(up,hs[up][0]),CW)*100);
 sx+='<span style="font-size:10px;color:#888">你</span> ';
 sx+='<input type="number" class="vpi" value="'+myVpip+'" disabled style="opacity:0.4"> ';
 for(var s of['l','r','u']){{var v=sv[s]||33;
  sx+='<span style="font-size:10px;color:#888">'+sl[s]+'</span> ';
  sx+='<input type="number" class="vpi" value="'+v+'" min="0" max="100" step="5" onchange="sv.'+s+'=parseInt(this.value)||33;sy();rn();cp()"> ';
 }}
 document.getElementById('sx').innerHTML=sx;

 var ox='';
 for(var p=0;p<N;p++){{
  if(p===up){{ox+='<button class="tog" style="background:#e94560;opacity:0.7;margin:0 1px" disabled>'+nm[p]+'</button>';}}
  else if(p<up){{var u=pp[p];ox+='<button class="tog" style="background:'+(u?'#4a90d9':'#555')+';margin:0 1px" onclick="pp['+p+']=!pp['+p+'];rn();cp()">'+nm[p]+'</button>';}}
  else{{ox+='<button class="tog" style="background:#333;color:#666;margin:0 1px" disabled>'+nm[p]+'</button>';}}
  var nv=Math.round(ap(ns(p,hs[p][0]),CW)*100),d=dev[p]||0,vv=Math.round(nv*(1+d));
  ox+='<span style="font-size:10px;color:#4a90d9;min-width:24px;display:inline-block;text-align:center" title="Nash '+nv+'%">'+vv+'%</span> ';
 }}
 document.getElementById('ox').innerHTML=ox;

 for(var n of[2,3,4])document.getElementById('b'+n).className=n===N?'sel':'';
}}

function bG(st){{
 function tI(r,c){{var hi=12-r,lo=12-c;if(hi===lo)return 12-hi;var o=0;if(hi>lo){{for(var h=12;h>hi;h--)o+=h;return 13+o+lo}}else{{for(var h=12;h>lo;h--)o+=h;return 13+78+o+hi}}}}
 var h='<table class="grid"><tr><th></th>';for(var c=0;c<13;c++)h+='<th>'+RK[12-c]+'</th>';h+='</tr>';
 for(var r=0;r<13;r++){{
  h+='<tr><th>'+RK[12-r]+'</th>';
  for(var c=0;c<13;c++){{
   var t=tI(r,c),v=(t>=0&&t<169)?st[t]:0,rd=Math.round(180*(1-v)),gn=Math.round(170*v);
   var bd;if(r<c)bd='2px solid #4a90d9';else if(r>c)bd='2px solid #d98a4a';else bd='2px solid #fff';
   var lb='{gl(0,0)}';lb=window.lbArr?window.lbArr[r*13+c]:'';
   h+='<td style="background:rgb('+rd+','+gn+',0);border:'+bd+'" title="'+lb+': '+Math.round(v*100)+'%">'+lb+'<br>'+Math.round(v*100)+'%</td>';
  }}
  h+='</tr>';
 }}
 return h+'</table>';
}}

function cp(){{
 var a=bA(),h=cH(),r=br(N,up,h,a);
 var na=[];for(var p=0;p<N;p++){{var ss=hs[p].map(function(hh){{return ns(p,hh)}});na.push({{ix:hi[p],ss:ss}})}}
 var nr=br(N,up,h,na);
 var hl=h.replace(/P/g,'1').replace(/F/g,'0');
 var hm='<div class="heat"><h3>'+nm[up]+' '+(hl||'开局')+' — Push范围 (剥削)</h3>'+bG(r.br)+'</div>';
 document.getElementById('ch').innerHTML=hm;
 var ga=[];for(var i=0;i<169;i++){{var d=r.vP[i]-nr.vP[i];if(Math.abs(d)>0.005)ga.push({{n:window.tnArr?window.tnArr[i]:'',d,e:r.vP[i],n2:nr.vP[i]}})}}
 ga.sort(function(a,b){{return b.d-a.d}});
 var gh='<b>EV 增益 | Push: '+r.br.filter(function(v){{return v>0.5}}).length+'/169</b><table style="font-size:11px"><tr style="color:#888"><th style="text-align:left;padding:1px 8px 1px 0">牌</th><th style="text-align:right;padding:1px 6px">Nash</th><th style="text-align:right;padding:1px 6px">剥削</th><th style="text-align:right;padding:1px 6px">增益</th></tr>';
 for(var g of ga.slice(0,25)){{var c=g.d>0?'pos':'neg';gh+='<tr><td>'+g.n+'</td><td style="text-align:right;color:#888">'+g.n2.toFixed(1)+'</td><td style="text-align:right;color:#ccc">'+g.e.toFixed(1)+'</td><td class="'+c+'" style="text-align:right">'+(g.d>0?'+':'')+g.d.toFixed(1)+'BB</td></tr>'}}
 gh+='</table>';if(ga.length===0)gh='<b>EV 增益</b><br>无显著偏离';
 document.getElementById('gx').innerHTML=gh;
}}

function sP(p){{up=p;pp=new Array(p).fill(true);rn();cp()}}
function tog(){{for(var p=0;p<up;p++)pp[p]=!pp[p];rn();cp()}}
function uR(v){{rb=parseFloat(v);document.getElementById('rv').textContent=v;cp()}}
function rst(){{sv={{l:33,r:38,u:64}};sy();rn();cp()}}

// Label arrays
window.tnArr = {json.dumps([gl(r,c) for r in range(13) for c in range(13)])};
window.lbArr = window.tnArr;

// Keyboard shortcuts: left=Q/W/E/F, right=U/I/O/J, space=all, 2/3/4=players
document.addEventListener('keydown', function(e){{
 var kc=e.which||e.keyCode;
 // F(70)/J(74): cycle pos; space(32): toggle all; 2/3/4: players
 if(kc===70||kc===74){{e.preventDefault();sP((up-1+N)%N);return;}}
 if(kc===32){{e.preventDefault();tog();return;}}
 if(kc>=50&&kc<=52){{e.preventDefault();setN(kc-48);return;}}
 if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
 // Left hand: Q(81)/W(87)/E(69); Right hand: U(85)/I(73)/O(79)
 if((kc===81||kc===85)&&up>0){{pp[0]=!pp[0];rn();cp();}}
 else if((kc===87||kc===73)&&up>1){{pp[1]=!pp[1];rn();cp();}}
 else if((kc===69||kc===79)&&up>2){{pp[2]=!pp[2];rn();cp();}}
}});

setN(4);
</script></body></html>"""

out = os.path.join(HERE, "aof_dashboard.html")
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Generated: {out} ({os.path.getsize(out)/1024:.0f} KB)")
