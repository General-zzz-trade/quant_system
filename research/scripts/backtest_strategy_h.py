#!/usr/bin/env python3
"""Backtest Strategy H: 4h direction + 1h position scaling + BB entry scaler.

Note: Uses pickle to load sklearn/lightgbm model files (industry standard ML format).
Full WF validation, $500 start, 10x leverage, BTC+ETH portfolio.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")  # noqa: E702
import json, pickle, sys, numpy as np, pandas as pd  # noqa: S403,E401
from pathlib import Path   # noqa: E702
sys.path.insert(0, "/quant_system")
from features.batch_feature_engine import compute_features_batch
from alpha.training.train_multi_horizon import rolling_zscore

DATA_DIR, MODEL_DIR = Path("data_files"), Path("models_v8")
COST = 0.0007

def load_resample(sym, rule):
    df = pd.read_csv(DATA_DIR / f"{sym}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
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

def load_predict(sym, interval, df):
    suffix = {"4h":"_4h","1h":"_gate_v2","15m":"_15m"}[interval]
    mdir = MODEL_DIR / f"{sym}{suffix}"
    if not mdir.exists(): return None, None
    cfg = json.load(open(mdir / "config.json"))
    feats = cfg.get("features") or cfg["horizon_models"][0]["features"]
    feat_df = compute_features_batch(sym, df); feat_df = add_cm(feat_df, df)
    for f in feats:
        if f not in feat_df.columns: feat_df[f] = 0.0
    X = feat_df[feats].values.astype(np.float64)
    for name in ["lgb_model.pkl","lgbm_v8.pkl"]+[h.get("lgbm","") for h in cfg.get("horizon_models",[])]:
        p2 = mdir / name
        if p2.exists():
            with open(p2,"rb") as fh: mdl = pickle.load(fh)
            if isinstance(mdl,dict) and "model" in mdl: mdl = mdl["model"]
            return mdl.predict(X), cfg
    return None, None

def gen_sig(pred, close, cfg):
    dz,mh,maxh = cfg.get("deadzone",1.0),cfg.get("min_hold",6),cfg.get("max_hold",36)
    lo,mg = cfg.get("long_only",False),cfg.get("monthly_gate",False)
    z = rolling_zscore(pred, window=cfg.get("zscore_window",180), warmup=cfg.get("zscore_warmup",45))
    sig = np.zeros(len(z)); sig[z>dz]=1; sig[z<-dz]=-1
    if lo: sig[sig<0]=0
    if mg:
        sw = 120 if len(close)<5000 else 480
        sma = pd.Series(close).rolling(sw, min_periods=sw//2).mean().values
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

def entry_bb_scale(close, idx, d, w=12):
    if idx < w: return 1.0
    r = close[idx-w:idx]; ma=np.mean(r); std=np.std(r)
    if std<=0: return 1.0
    bb = (close[idx]-ma)/std
    if d==1: return 1.2 if bb<-1 else (1.0 if bb<-0.5 else (0.7 if bb<0 else (0.5 if bb<0.5 else 0.3)))
    if d==-1: return 1.2 if bb>1 else (1.0 if bb>0.5 else (0.7 if bb>0 else (0.5 if bb>-0.5 else 0.3)))
    return 1.0

def bt_h(cl, s4, s1, lev, cap, use_bb=True, init=500.0):
    eq,pk,mdd,pos,ep=init,init,0.0,0,0.0
    trades,entry_sc=[],1.0; dpnl_list=[]; dpnl=0.0
    for i in range(1,len(cl)):
        c=cl[i]; s=int(s4[i])
        if s!=pos:
            if pos!=0 and ep>0:
                net=pos*(c/ep-1)-2*COST; pnl=eq*cap*lev*net*entry_sc
                eq+=pnl; dpnl+=pnl; trades.append({"net":net,"dir":pos,"sc":entry_sc})
            if s!=0:
                s1v=int(s1[i]); tf_sc=1.3 if s1v==s else (0.3 if s1v==-s else 0.7)
                bb_sc=entry_bb_scale(cl,i,s) if use_bb else 1.0
                entry_sc=tf_sc*bb_sc; pos,ep=s,c
            else: pos,ep,entry_sc=0,0.0,1.0
        pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd)
        if i%24==0: dpnl_list.append(dpnl/max(eq-dpnl,1) if eq>0 else 0); dpnl=0.0
    dr=np.array(dpnl_list)
    sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
    nd=len(cl)/24; cagr=(eq/init)**(365/max(nd,1))-1
    rets=[t["net"] for t in trades]; wr=np.mean(np.array(rets)>0)*100 if rets else 0
    avg_sc=np.mean([t["sc"] for t in trades]) if trades else 1.0
    l_tr=[t for t in trades if t["dir"]>0]; s_tr=[t for t in trades if t["dir"]<0]
    l_wr=np.mean([t["net"]>0 for t in l_tr])*100 if l_tr else 0
    s_wr=np.mean([t["net"]>0 for t in s_tr])*100 if s_tr else 0
    return {"sharpe":round(sharpe,2),"cagr":round(cagr*100,1),"ret":round((eq/init-1)*100,1),
            "mdd":round(mdd,1),"n":len(trades),"wr":round(wr,1),"eq":round(eq,0),
            "avg_sc":round(avg_sc,2),"l_n":len(l_tr),"s_n":len(s_tr),"l_wr":round(l_wr,1),"s_wr":round(s_wr,1)}

def wf_h(cl, s4, s1, lev, cap, use_bb, fold_d=90, min_d=365):
    mb,fb=min_d*24,fold_d*24; nf=(len(cl)-mb)//fb
    if nf<3: return None
    sharpes,rets,dds=[],[],[]
    for f in range(nf):
        s=mb+f*fb; e=min(s+fb,len(cl))
        r=bt_h(cl[s:e],s4[s:e],s1[s:e],lev,cap,use_bb)
        if r["n"]>0: sharpes.append(r["sharpe"]); rets.append(r["ret"]); dds.append(r["mdd"])
    np_=sum(1 for s in sharpes if s>0)
    return {"pass":f"{np_}/{len(sharpes)}","pr":np_/max(len(sharpes),1),
            "ms":round(np.mean(sharpes),2),"mds":round(np.median(sharpes),2),
            "ss":round(np.std(sharpes),2),"mr":round(np.mean(rets),1),
            "md":round(np.mean(dds),1),"wd":round(min(dds),1) if dds else 0}

def main():
    lev=10.0; all_data={}
    for sym in ["BTCUSDT","ETHUSDT"]:
        print(f"\n{'='*70}"); print(f"  {sym} — Strategy H ($500, {lev}x)"); print(f"{'='*70}")
        df1=load_resample(sym,"1h"); df4=load_resample(sym,"4h")
        cl=df1["close"].values; dt=df1["datetime"].values
        p1,c1=load_predict(sym,"1h",df1); s1=gen_sig(p1,cl,c1)
        p4,c4=load_predict(sym,"4h",df4); s4r=gen_sig(p4,df4["close"].values,c4)
        s4=align_to_1h(s4r,df4["datetime"].values,dt)
        cap=0.15 if "BTC" in sym else 0.10
        # Stats
        agree=((s4!=0)&(s1==s4)).sum(); oppose=((s4!=0)&(s1!=0)&(s1!=s4)).sum()
        active4=(s4!=0).sum()
        print(f"  4h+1h agree: {agree} bars ({100*agree/max(active4,1):.0f}% of active)")
        print(f"  4h+1h oppose: {oppose} bars ({100*oppose/max(active4,1):.0f}% of active)")
        # Variants
        v = {}
        v["1h only"] = bt_h(cl, s1, s1, lev, cap, False)
        v["4h only"] = bt_h(cl, s4, np.zeros_like(s4), lev, cap*0.6, False)
        v["H: no BB"] = bt_h(cl, s4, s1, lev, cap, False)
        v["H: +BB scaler"] = bt_h(cl, s4, s1, lev, cap, True)
        print(f"\n  {'Variant':<20} {'Sharpe':>7} {'CAGR%':>7} {'Ret%':>10} {'MaxDD%':>7} {'Trades':>6} {'WR%':>5} {'Scale':>5} {'L_WR':>5} {'S_WR':>5}")
        print(f"  {'-'*85}")
        for n,r in v.items():
            print(f"  {n:<20} {r['sharpe']:7.2f} {r['cagr']:7.1f} {r['ret']:10.1f} {r['mdd']:7.1f} {r['n']:6d} {r['wr']:5.1f} {r['avg_sc']:5.2f} {r['l_wr']:5.1f} {r['s_wr']:5.1f}")
        # WF
        print("\n  Walk-Forward (90-day folds):")
        for name,bb in [("H: no BB",False),("H: +BB",True)]:
            wf=wf_h(cl,s4,s1,lev,cap,bb)
            if wf:
                print(f"  {name:<15} Pass={wf['pass']} ({wf['pr']*100:.0f}%) Mean={wf['ms']:.2f}±{wf['ss']:.2f} "
                      f"Med={wf['mds']:.2f} MeanDD={wf['md']:.1f}% WorstDD={wf['wd']:.1f}%")
        all_data[sym]={"cl":cl,"s4":s4,"s1":s1,"cap":cap}

    # Portfolio
    print(f"\n{'='*70}"); print(f"  BTC+ETH Portfolio Strategy H ($500, {lev}x)"); print(f"{'='*70}")
    eq,pk,mdd=500.0,500.0,0.0; trades=0; dpnl_list=[]; dpnl=0.0
    syms=["BTCUSDT","ETHUSDT"]; max_len=min(len(all_data[s]["cl"]) for s in syms)
    positions={s:{"pos":0,"ep":0.0,"sc":1.0} for s in syms}
    for i in range(1,max_len):
        for sym in syms:
            d=all_data[sym]; cl,s4,s1,cap=d["cl"],d["s4"],d["s1"],d["cap"]
            c=cl[i]; s=int(s4[i]); p=positions[sym]
            if s!=p["pos"]:
                if p["pos"]!=0 and p["ep"]>0:
                    net=p["pos"]*(c/p["ep"]-1)-2*COST
                    pnl=eq*cap*lev*net*p["sc"]; eq+=pnl; dpnl+=pnl; trades+=1
                if s!=0:
                    s1v=int(s1[i]); tf=1.3 if s1v==s else (0.3 if s1v==-s else 0.7)
                    bb=entry_bb_scale(cl,i,s)
                    p["pos"],p["ep"],p["sc"]=s,c,tf*bb
                else: p["pos"],p["ep"],p["sc"]=0,0.0,1.0
        pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd)
        if i%24==0: dpnl_list.append(dpnl/max(eq-dpnl,1) if eq>0 else 0); dpnl=0.0
    dr=np.array(dpnl_list)
    sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
    nd=max_len/24; cagr=(eq/500)**(365/max(nd,1))-1
    print(f"\n  Sharpe:       {sharpe:.2f}")
    print(f"  CAGR:         {cagr*100:.1f}%")
    print(f"  Total Return: {(eq/500-1)*100:.1f}%")
    print(f"  Max Drawdown: {mdd:.1f}%")
    print(f"  Trades:       {trades}")
    print(f"  $500 → ${eq:,.0f}")

if __name__=="__main__":
    main()
