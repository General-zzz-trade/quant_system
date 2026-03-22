#!/usr/bin/env python3
"""Analyze optimal multi-timeframe integration strategy.

Uses pickle to load sklearn/lightgbm trained model files (standard ML serialization).

Compares 9 approaches for combining 15m, 1h, 4h signals.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json, pickle, sys, numpy as np, pandas as pd  # noqa: S403,E401,E702
from pathlib import Path; from scipy import stats  # noqa: E702
sys.path.insert(0, "/quant_system")
from features.batch_feature_engine import compute_features_batch
from scripts.train_multi_horizon import rolling_zscore

DATA_DIR, MODEL_DIR = Path("data_files"), Path("models_v8")
COST = 0.0007

def load_resample(sym, rule):
    df = pd.read_csv(DATA_DIR / f"{sym}_1h.csv")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    if rule == "1h": return df
    agg = {"open_time":"first","open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    for c in ["quote_volume","taker_buy_volume","taker_buy_quote_volume","trades"]:
        if c in df.columns: agg[c] = "sum"
    return df.set_index("datetime").resample(rule).agg(agg).dropna(subset=["close"]).reset_index()

def add_cm(feat_df, df):
    p = DATA_DIR / "cross_market_daily.csv"
    if not p.exists(): return feat_df
    cm = pd.read_csv(p, parse_dates=["date"]); cm["date"] = cm["date"].dt.date
    dates = pd.to_datetime(df["open_time"], unit="ms").dt.date
    ci = cm.set_index("date")
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
        p = mdir / name
        if p.exists():
            with open(p,"rb") as fh: mdl = pickle.load(fh)
            if isinstance(mdl,dict) and "model" in mdl: mdl = mdl["model"]
            return mdl.predict(X), cfg
    return None, None

def gen_sig(pred, close, cfg):
    dz,mh,maxh = cfg.get("deadzone",1.0), cfg.get("min_hold",6), cfg.get("max_hold",36)
    lo,mg = cfg.get("long_only",False), cfg.get("monthly_gate",False)
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
            if sig[i]==cur: hold+=1; (sig.__setitem__(i,0),cur:=0,hold:=0) if hold>maxh else None  # noqa
            else: cur,hold=sig[i],1
        else: cur,hold=0,0
    # simpler max hold
    cur2,hold2=0,0
    for i in range(len(sig)):
        if sig[i]!=0:
            if sig[i]==cur2: hold2+=1
            else: cur2,hold2=sig[i],1
            if hold2>maxh: sig[i]=0; cur2=0; hold2=0
        else: cur2,hold2=0,0
    return sig

def align_to_1h(sig, dt_src, dt_1h):
    s = pd.Series(sig, index=dt_src)
    return s.reindex(dt_1h, method="ffill").fillna(0).values

def bt(cl, sig, lev, cap, init=500.0):
    eq,pk,mdd,pos,ep,trades,wins=init,init,0.0,0,0.0,0,0
    dpnl_list,dpnl=[],0.0
    for i in range(1,len(cl)):
        c,s=cl[i],int(sig[i])
        if s!=pos:
            if pos!=0 and ep>0:
                net=pos*(c/ep-1)-2*COST; pnl=eq*cap*lev*net; eq+=pnl; dpnl+=pnl; trades+=1
                if net>0: wins+=1
            if s!=0: pos,ep=s,c
            else: pos,ep=0,0.0
        pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd)
        if i%24==0: dpnl_list.append(dpnl/max(eq-dpnl,1) if eq>0 else 0); dpnl=0.0
    dr=np.array(dpnl_list)
    sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
    wr=wins/max(trades,1)*100; nd=len(cl)/24; cagr=(eq/init)**(365/max(nd,1))-1
    return {"sharpe":round(sharpe,2),"cagr":round(cagr*100,1),"ret":round((eq/init-1)*100,1),
            "mdd":round(mdd,1),"n":trades,"wr":round(wr,1),"eq":round(eq,0)}

def main():
    lev=10.0
    for sym in ["BTCUSDT","ETHUSDT"]:
        print(f"\n{'='*70}")
        print(f"  {sym} — Multi-TF Integration Analysis ($500, {lev}x)")
        print(f"{'='*70}")
        df_1h=load_resample(sym,"1h"); df_4h=load_resample(sym,"4h")
        cl=df_1h["close"].values; dt=df_1h["datetime"].values
        # Load signals
        p1,c1=load_predict(sym,"1h",df_1h); s1=gen_sig(p1,cl,c1)
        p4,c4=load_predict(sym,"4h",df_4h); s4r=gen_sig(p4,df_4h["close"].values,c4)
        s4=align_to_1h(s4r,df_4h["datetime"].values,dt)
        # 15m
        p15=DATA_DIR/f"{sym}_15m.csv"; has15=p15.exists()
        if has15:
            df15=pd.read_csv(p15); df15["datetime"]=pd.to_datetime(df15["open_time"],unit="ms")
            pp,cp=load_predict(sym,"15m",df15)
            if pp is not None: s15r=gen_sig(pp,df15["close"].values,cp); s15=align_to_1h(s15r,df15["datetime"].values,dt)
            else: has15=False
        cap1=0.15 if "BTC" in sym else 0.10
        results={}
        # A: 1h standalone
        r=bt(cl,s1,lev,cap1); r["label"]="A: 1h standalone"; results["A"]=r
        # B: 4h standalone
        r=bt(cl,s4,lev,cap1*0.6); r["label"]="B: 4h standalone"; results["B"]=r
        # C: 1h+15m AGREE
        if has15:
            sa=np.where((s1!=0)&(s1==s15),s1,0.0)
            r=bt(cl,sa,lev,cap1); r["label"]="C: 1h+15m AGREE"; results["C"]=r
        # D: 4h→1h cascade (1h only when 4h agrees)
        sd=np.where((s1!=0)&(s4==s1),s1,0.0)
        r=bt(cl,sd,lev,cap1); r["label"]="D: 4h→1h cascade"; results["D"]=r
        # E: 4h+1h+15m triple agree
        if has15:
            se=np.where((s1!=0)&(s15==s1)&(s4==s1),s1,0.0)
            r=bt(cl,se,lev,cap1*1.3); r["label"]="E: Triple agree (x1.3)"; results["E"]=r
        # F: Majority vote (2/3)
        if has15:
            sf=np.zeros(len(cl))
            for i in range(len(cl)):
                l=sum(1 for v in [s1[i],s15[i],s4[i]] if v>0)
                sh=sum(1 for v in [s1[i],s15[i],s4[i]] if v<0)
                if l>=2: sf[i]=1
                elif sh>=2: sf[i]=-1
            r=bt(cl,sf,lev,cap1); r["label"]="F: Majority (2/3)"; results["F"]=r
        # G: IC-weighted blend
        sg=np.zeros(len(cl))
        for i in range(len(cl)):
            score=s4[i]*0.50+s1[i]*0.30+(s15[i]*0.20 if has15 else 0)
            if score>0.3: sg[i]=1
            elif score<-0.3: sg[i]=-1
        r=bt(cl,sg,lev,cap1); r["label"]="G: IC-weighted blend"; results["G"]=r
        # H: 4h primary + 1h scale
        sh=s4.copy()
        scale=np.where(s1==s4,1.3,np.where(s1==-s4,0.3,0.7))
        eq,pk,mdd,pos,ep,trades,wins=500.0,500.0,0.0,0,0.0,0,0
        dpnl_list,dpnl=[],0.0; entry_sc=1.0
        for i in range(1,len(cl)):
            c=cl[i]; s=int(sh[i])
            if s!=pos:
                if pos!=0 and ep>0:
                    net=pos*(c/ep-1)-2*COST; pnl=eq*cap1*0.6*lev*net*entry_sc
                    eq+=pnl; dpnl+=pnl; trades+=1
                    if net>0: wins+=1
                if s!=0: pos,ep,entry_sc=s,c,scale[i]
                else: pos,ep,entry_sc=0,0.0,1.0
            pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd)
            if i%24==0: dpnl_list.append(dpnl/max(eq-dpnl,1) if eq>0 else 0); dpnl=0.0
        dr=np.array(dpnl_list)
        sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
        wr=wins/max(trades,1)*100; cagr=(eq/500)**(365/max(len(cl)/24,1))-1
        results["H"]={"label":"H: 4h dir + 1h scale","sharpe":round(sharpe,2),"cagr":round(cagr*100,1),
            "ret":round((eq/500-1)*100,1),"mdd":round(mdd,1),"n":trades,"wr":round(wr,1),"eq":round(eq,0)}
        # I: Independent portfolio (1h cap1 + 4h cap2, separate positions)
        eq,pk,mdd=500.0,500.0,0.0; p1h,p4h,ep1,ep4=0,0,0.0,0.0; trades,wins=0,0
        dpnl_list,dpnl=[],0.0; cap4v=cap1*0.5
        for i in range(1,len(cl)):
            c=cl[i]
            s_1=int(s1[i]); s_4=int(s4[i])
            if s_1!=p1h:
                if p1h!=0 and ep1>0:
                    net=p1h*(c/ep1-1)-2*COST; eq+=eq*cap1*lev*net; dpnl+=eq*cap1*lev*net; trades+=1
                    if net>0: wins+=1
                if s_1!=0: p1h,ep1=s_1,c
                else: p1h,ep1=0,0.0
            if s_4!=p4h:
                if p4h!=0 and ep4>0:
                    net=p4h*(c/ep4-1)-2*COST; eq+=eq*cap4v*lev*net; dpnl+=eq*cap4v*lev*net; trades+=1
                    if net>0: wins+=1
                if s_4!=0: p4h,ep4=s_4,c
                else: p4h,ep4=0,0.0
            pk=max(pk,eq); dd=(eq-pk)/pk*100 if pk>0 else 0; mdd=min(mdd,dd)
            if i%24==0: dpnl_list.append(dpnl/max(eq,1)); dpnl=0.0
        dr=np.array(dpnl_list)
        sharpe=np.mean(dr)/np.std(dr)*np.sqrt(365) if np.std(dr)>0 else 0
        wr=wins/max(trades,1)*100; cagr=(eq/500)**(365/max(len(cl)/24,1))-1
        results["I"]={"label":"I: 1h+4h independent","sharpe":round(sharpe,2),"cagr":round(cagr*100,1),
            "ret":round((eq/500-1)*100,1),"mdd":round(mdd,1),"n":trades,"wr":round(wr,1),"eq":round(eq,0)}

        # Print
        print(f"\n  {'Strategy':<25} {'Sharpe':>7} {'CAGR%':>7} {'Ret%':>10} {'MaxDD%':>8} {'Trades':>7} {'WR%':>6}")
        print(f"  {'-'*72}")
        for k in sorted(results):
            r=results[k]
            print(f"  {r['label']:<25} {r['sharpe']:7.2f} {r['cagr']:7.1f} {r['ret']:10.1f} {r['mdd']:8.1f} {r['n']:7d} {r['wr']:6.1f}")
        ranked=sorted(results.items(),key=lambda x:x[1]["sharpe"],reverse=True)
        print(f"\n  Ranking by Sharpe:")
        for i,(k,r) in enumerate(ranked):
            mark="  ★ BEST" if i==0 else ""
            print(f"    {i+1}. {r['label']:<25} Sharpe={r['sharpe']:6.2f} MaxDD={r['mdd']:6.1f}%{mark}")

if __name__=="__main__":
    main()
