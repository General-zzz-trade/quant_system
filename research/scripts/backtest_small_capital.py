#!/usr/bin/env python3
# Small capital backtest: $500/10x/Strategy H
# Uses pickle for sklearn/lightgbm model loading (required by user for ML pipeline)
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")  # noqa: E702
import json, pickle, sys, numpy as np, pandas as pd  # noqa: S403,E401
from pathlib import Path  # noqa: E402
sys.path.insert(0, "/quant_system")
from features.batch_feature_engine import compute_features_batch
from alpha.training.train_multi_horizon import rolling_zscore
DATA_DIR, MODEL_DIR = Path("data_files"), Path("models_v8")
COST = 0.0007
def load_resample(sym, rule):
    df = pd.read_csv(DATA_DIR / f"{sym}_1h.csv"); df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    if rule == "1h": return df
    agg = {"open_time":"first","open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    for c in ["quote_volume","taker_buy_volume","taker_buy_quote_volume","trades"]:
        if c in df.columns: agg[c]="sum"
    return df.set_index("datetime").resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
def add_cm(feat_df, df):
    p = DATA_DIR / "cross_market_daily.csv"
    if not p.exists(): return feat_df
    cm = pd.read_csv(p, parse_dates=["date"]); cm["date"] = cm["date"].dt.date
    dates = pd.to_datetime(df["open_time"], unit="ms").dt.date; ci = cm.set_index("date")
    # T-1 shift: shift index forward by 1 day so bar on date D uses data from D-1
    ci.index = [d + pd.Timedelta(days=1) for d in ci.index]
    for col in ci.columns: feat_df[col] = dates.map(lambda d, c=col: ci[c].get(d, np.nan)).ffill().values
    return feat_df
def load_predict(sym, itv, df):
    suffix = {"4h":"_4h","1h":"_gate_v2"}[itv]; mdir = MODEL_DIR / f"{sym}{suffix}"
    if not mdir.exists(): return None, None
    cfg = json.load(open(mdir / "config.json")); feats = cfg.get("features") or cfg["horizon_models"][0]["features"]
    feat_df = compute_features_batch(sym, df); feat_df = add_cm(feat_df, df)
    for f in feats:
        if f not in feat_df.columns: feat_df[f] = 0.0
    X = feat_df[feats].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)  # Ridge requires no NaN
    for name in ["ridge_model.pkl","lgb_model.pkl","lgbm_v8.pkl"]+[h.get("lgbm","") for h in cfg.get("horizon_models",[])]+[h.get("ridge","") for h in cfg.get("horizon_models",[])]:
        if not name: continue
        p2 = mdir / name
        if p2.exists():
            with open(p2,"rb") as fh: mdl = pickle.load(fh)
            if isinstance(mdl,dict) and "model" in mdl: mdl = mdl["model"]
            try:
                return mdl.predict(X), cfg
            except TypeError:
                import xgboost as xgb
                return mdl.predict(xgb.DMatrix(X)), cfg
    return None, None
def gen_sig(pred, close, cfg):
    dz,mh,maxh=cfg.get("deadzone",1.0),cfg.get("min_hold",6),cfg.get("max_hold",36)
    lo,mg=cfg.get("long_only",False),cfg.get("monthly_gate",False)
    z=rolling_zscore(pred,window=cfg.get("zscore_window",180),warmup=cfg.get("zscore_warmup",45))
    sig=np.zeros(len(z)); sig[z>dz]=1; sig[z<-dz]=-1
    if lo: sig[sig<0]=0
    if mg:
        sw=120 if len(close)<5000 else 480
        sma=pd.Series(close).rolling(sw,min_periods=sw//2).mean().values
        for i in range(len(sig)):
            if not np.isnan(sma[i]) and close[i]<sma[i]: sig[i]=min(sig[i],0)
    cur,hold=0,0
    for i in range(len(sig)):
        s=sig[i]
        if s!=cur and s!=0: cur,hold=s,1
        elif cur!=0 and hold<mh: sig[i]=cur; hold+=1
        elif s!=cur: cur,hold=s,(1 if s!=0 else 0)
    cur,hold=0,0
    for i in range(len(sig)):
        if sig[i]!=0:
            if sig[i]==cur: hold+=1
            else: cur,hold=sig[i],1
            if hold>maxh: sig[i],cur,hold=0,0,0
        else: cur,hold=0,0
    return sig
def align_to_1h(sig, dt_src, dt_1h):
    return pd.Series(sig, index=dt_src).reindex(dt_1h, method="ffill").fillna(0).values
def bb_scale(cl, i, d, w=12):
    if i < w: return 1.0
    r=cl[i-w:i]; ma=np.mean(r); std=np.std(r)
    if std<=0: return 1.0
    bb=(cl[i]-ma)/std
    if d==1: return 1.2 if bb<-1 else (1.0 if bb<-0.5 else (0.7 if bb<0 else (0.5 if bb<0.5 else 0.3)))
    if d==-1: return 1.2 if bb>1 else (1.0 if bb>0.5 else (0.7 if bb>0 else (0.5 if bb>-0.5 else 0.3)))
    return 1.0
def main():
    init, lev = 500.0, 10.0; caps = {"BTCUSDT": 0.15, "ETHUSDT": 0.10}; syms = list(caps.keys())
    print(f"\n{'='*70}"); print(f"  Strategy H: $500 → ? with 10x leverage"); print(f"{'='*70}")
    data = {}
    for sym in syms:
        df1=load_resample(sym,"1h"); df4=load_resample(sym,"4h"); cl=df1["close"].values; dt=df1["datetime"].values
        p1,c1=load_predict(sym,"1h",df1); s1=gen_sig(p1,cl,c1)
        p4,c4=load_predict(sym,"4h",df4); s4r=gen_sig(p4,df4["close"].values,c4); s4=align_to_1h(s4r,df4["datetime"].values,dt)
        data[sym]={"cl":cl,"dt":dt,"s4":s4,"s1":s1}; print(f"  {sym}: {len(cl)} bars")
    ml=min(len(data[s]["cl"]) for s in syms)
    eq=init; pk=eq; mdd=0.0; pos={s:{"p":0,"e":0.0,"sc":1.0} for s in syms}
    tlog=[]; eqh=[eq]; meq={}
    for i in range(1,ml):
        dt_v=pd.Timestamp(data["BTCUSDT"]["dt"][i])
        for sym in syms:
            d=data[sym]; cl,s4,s1=d["cl"],d["s4"],d["s1"]; cap=caps[sym]; c=cl[i]; s=int(s4[i]); p=pos[sym]
            if s!=p["p"]:
                if p["p"]!=0 and p["e"]>0:
                    net=p["p"]*(c/p["e"]-1)-2*COST; pnl=eq*cap*lev*net*p["sc"]; eq+=pnl
                    tlog.append({"t":dt_v,"sym":sym,"dir":p["p"],"ep":p["e"],"ex":c,"net":net*100,"pnl":pnl,"eq":eq,"sc":p["sc"]})
                if s!=0:
                    s1v=int(s1[i]); tf=1.3 if s1v==s else (0.3 if s1v==-s else 0.7)
                    p["p"],p["e"],p["sc"]=s,c,tf*bb_scale(cl,i,s)
                else: p["p"],p["e"],p["sc"]=0,0.0,1.0
        pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd); eqh.append(eq)
        meq[f"{dt_v.year}-{dt_v.month:02d}"]=eq
    T=pd.DataFrame(tlog); nd=ml/24; ny=nd/365; cagr=(eq/init)**(1/max(ny,0.1))-1
    print(f"\n  {'='*50}"); print(f"  PERFORMANCE"); print(f"  {'='*50}")
    print(f"  ${init:,.0f} → ${eq:,.0f}"); print(f"  Return: {(eq/init-1)*100:,.1f}%"); print(f"  CAGR: {cagr*100:.1f}%"); print(f"  MaxDD: {mdd:.1f}%"); print(f"  Trades: {len(T)}")
    if len(T)==0: return
    W=T[T["net"]>0]; L=T[T["net"]<=0]
    print(f"\n  {'='*50}"); print(f"  TRADES"); print(f"  {'='*50}")
    print(f"  WR: {len(W)/len(T)*100:.1f}% ({len(W)}/{len(T)})"); print(f"  Avg Win: {W['net'].mean():.2f}%"); print(f"  Avg Loss: {L['net'].mean():.2f}%" if len(L)>0 else "")
    print(f"  Best: {T['net'].max():.2f}%  Worst: {T['net'].min():.2f}%")
    pf=W['pnl'].sum()/abs(L['pnl'].sum()) if len(L)>0 and L['pnl'].sum()!=0 else float('inf')
    print(f"  Profit Factor: {pf:.2f}" if pf<1000 else f"  Profit Factor: {pf:.0f}")
    for sym in syms:
        st=T[T["sym"]==sym]; sw=st[st["net"]>0]; sl=st[st["net"]<=0]
        lt=st[st["dir"]>0]; sht=st[st["dir"]<0]
        if len(st)==0: continue
        print(f"\n  {sym}: {len(st)} trades WR={len(sw)/len(st)*100:.1f}% PnL=${st['pnl'].sum():,.0f}")
        print(f"    Long: {len(lt)} WR={len(lt[lt['net']>0])/max(len(lt),1)*100:.1f}%", end="")
        if len(sht)>0: print(f"  Short: {len(sht)} WR={len(sht[sht['net']>0])/max(len(sht),1)*100:.1f}%")
        else: print()
    print(f"\n  {'='*50}"); print(f"  YEARLY BREAKDOWN"); print(f"  {'='*50}")
    T["yr"]=T["t"].dt.year; ys=init
    for yr in sorted(T["yr"].unique()):
        yt=T[T["yr"]==yr]; ye=yt["eq"].iloc[-1]; yr_r=(ye/ys-1)*100; yw=len(yt[yt["net"]>0])/len(yt)*100
        print(f"  {yr}: {yr_r:+10.1f}%  ${ys:>14,.0f} → ${ye:>14,.0f}  trades={len(yt):3d} WR={yw:.0f}%"); ys=ye
    ea=np.array(eqh); de=ea[::24]; dr=np.diff(de)/de[:-1]; dr=dr[np.isfinite(dr)]
    sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
    dn=dr[dr<0]; sortino=np.mean(dr)/np.std(dn)*np.sqrt(365) if len(dn)>0 and np.std(dn)>0 else 0
    calmar=cagr/abs(mdd)*100 if mdd!=0 else 0
    ms=sorted(meq.keys()); pm=init; posm=sum(1 for m in ms if meq[m]>(pm:=meq.get(ms[max(0,ms.index(m)-1)],init)))
    # Recompute properly
    posm=0; pm=init
    for m in ms:
        if meq[m]>pm: posm+=1
        pm=meq[m]
    sk=0; msk=0
    for _,t in T.iterrows():
        if t["net"]<=0: sk+=1; msk=max(msk,sk)
        else: sk=0
    print(f"\n  {'='*50}"); print(f"  RISK METRICS"); print(f"  {'='*50}")
    print(f"  Sharpe:     {sharpe:.2f}"); print(f"  Sortino:    {sortino:.2f}"); print(f"  Calmar:     {calmar:.2f}")
    print(f"  Daily Vol:  {np.std(dr)*100:.2f}%"); print(f"  Ann Vol:    {np.std(dr)*np.sqrt(365)*100:.1f}%")
    print(f"  Monthly WR: {posm}/{len(ms)} ({100*posm/max(len(ms),1):.0f}%)"); print(f"  Max Loss Streak: {msk}")
    print(f"\n  {'='*50}"); print(f"  MILESTONES"); print(f"  {'='*50}")
    for m in [1000,2500,5000,10000,25000,50000,100000,250000,500000,1_000_000,5_000_000,10_000_000,50_000_000,100_000_000]:
        for idx,e in enumerate(eqh):
            if e>=m:
                days=idx/24; dt_at=pd.Timestamp(data["BTCUSDT"]["dt"][min(idx,ml-1)])
                print(f"    ${m:>14,}  day {days:5.0f} ({days/30:5.1f}mo)  ~{dt_at.date()}"); break
    print(f"\n  {'='*50}"); print(f"  TOP DRAWDOWNS"); print(f"  {'='*50}")
    pka=np.maximum.accumulate(ea); dda=(ea-pka)/pka*100; idd=False; dds=0; ddl=[]
    for i in range(len(dda)):
        if dda[i]<-1 and not idd: dds=i; idd=True
        elif dda[i]>=-0.1 and idd: ddl.append({"d":dda[dds:i].min(),"dur":(i-dds)/24,"s":dds}); idd=False
    ddl.sort(key=lambda x:x["d"])
    for j,dd in enumerate(ddl[:5]):
        dt_s=pd.Timestamp(data["BTCUSDT"]["dt"][min(dd["s"],ml-1)])
        print(f"    {j+1}. {dd['d']:6.1f}%  {dd['dur']:5.0f} days  ~{dt_s.date()}")
if __name__=="__main__": main()
