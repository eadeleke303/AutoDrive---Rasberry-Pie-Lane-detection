"""Annotation-independent candidate pair scoring for offline simulation."""
import numpy as np

def at_y(points, y):
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or len(a)<2 or not np.isfinite(a).all():return None
    a=a[np.argsort(a[:,1])]
    if y<a[0,1] or y>a[-1,1]:return None
    return float(np.interp(y,a[:,1],a[:,0]))

class EgoLaneTracker:
    def __init__(self):
        self.previous=None
        self.age=0
    def reset(self):
        self.previous=None
        self.age=0
    def select(self,candidates,w,h,allow_recovery=False):
        options=[]
        diagnostics={"pairs_considered":0,"pairs_rejected_geometry":0}
        for i,a in enumerate(candidates):
            for j,b in enumerate(candidates):
                if i==j:continue
                ap,bp=np.asarray(a["points"],float),np.asarray(b["points"],float)
                lo=max(float(ap[:,1].min()),float(bp[:,1].min()),.42*h)
                hi=min(float(ap[:,1].max()),float(bp[:,1].max()),.99*h)
                if hi-lo<.09*h:continue
                ys=np.linspace(lo+.05*(hi-lo),hi-.05*(hi-lo),11)
                xs=np.array([(at_y(ap,y),at_y(bp,y)) for y in ys],float)
                if not np.isfinite(xs).all():continue
                diagnostics["pairs_considered"]+=1
                widths=(xs[:,1]-xs[:,0])/w
                if np.any(widths<.055) or np.any(widths>.93):
                    diagnostics["pairs_rejected_geometry"]+=1
                    continue
                ratio=float(np.max(widths)/max(np.min(widths),1e-6))
                if ratio>5:
                    diagnostics["pairs_rejected_geometry"]+=1
                    continue
                mid=xs.mean(axis=1)/w
                kink=float(np.mean(np.abs(np.diff(mid,n=2))))
                kink+=.5*float(np.mean(np.abs(np.diff(widths,n=2))))
                if kink>.11:
                    diagnostics["pairs_rejected_geometry"]+=1
                    continue
                confidence=min(float(a.get("confidence",0)),float(b.get("confidence",0)))
                support=min(float(a.get("support",0)),float(b.get("support",0)))
                center_dist=float(np.min(np.abs(mid-.5)))
                encloses=float(np.mean((xs[:,0]<.5*w)&(xs[:,1]>.5*w)))
                median_width=float(np.median(widths))
                continuity=0.
                if self.previous is not None:
                    oldl,oldr=self.previous
                    errors=[]
                    for y,(xl,xr) in zip(ys,xs):
                        pl,pr=at_y(oldl,y),at_y(oldr,y)
                        if pl is not None and pr is not None:
                            errors.append((abs(xl-pl)+abs(xr-pr))/(2*w))
                    if errors:continuity=.22-.9*float(np.mean(errors))
                score=(confidence+.20*support+.18*encloses-.75*center_dist
                       -.27*median_width-1.8*kink-.10*min(ratio-1,4)+continuity)
                options.append((score,i,j,len(ys)))
        if options:
            score,i,j,nrows=max(options,key=lambda z:z[0])
            left,right=candidates[i],candidates[j]
            self.previous=(left["points"],right["points"])
            self.age=0
            return left["points"],right["points"],dict(reason="selected",selected_channels=[left["channel"],right["channel"]],estimated=False,compared_rows=nrows,pair_score=round(float(score),4),**diagnostics)
        if allow_recovery and self.previous is not None and self.age<3 and candidates:
            oldl,oldr=self.previous
            for anchor in [.82*h,.75*h,.68*h]:
                pl,pr=at_y(oldl,anchor),at_y(oldr,anchor)
                if pl is None or pr is None or not .10*w<=pr-pl<=.80*w:continue
                matches=[]
                for c in candidates:
                    x=at_y(c["points"],anchor)
                    if x is not None:
                        matches.extend([(abs(x-pl),"left",c),(abs(x-pr),"right",c)])
                if matches:
                    error,side,c=min(matches,key=lambda t:t[0])
                    if error<.07*w:
                        pts=np.asarray(c["points"],float)
                        other=pts.copy()
                        other[:,0]+=pr-pl if side=="left" else -(pr-pl)
                        self.age+=1
                        left,right=(pts,other) if side=="left" else (other,pts)
                        return left.tolist(),right.tolist(),dict(reason="single_boundary_estimate",selected_channels=[c["channel"]],estimated=True,compared_rows=0,pair_score=None,**diagnostics)
        self.age+=1
        if self.age>3:self.previous=None
        return [],[],dict(reason="no_usable_pair",selected_channels=[],estimated=False,compared_rows=0,pair_score=None,**diagnostics)
