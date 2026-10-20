"""Centered ego-lane selector for offline video research."""
import numpy as np
from ego_lane_v15 import at_y

class EgoLaneTracker:
    def __init__(self):
        self.previous=None
    def reset(self):
        self.previous=None
    def select(self,candidates,w,h,allow_recovery=False):
        options=[]
        y=.84*h
        for i,a in enumerate(candidates):
            xa=at_y(a["points"],y)
            if xa is None:continue
            for j,b in enumerate(candidates):
                if j==i:continue
                xb=at_y(b["points"],y)
                if xb is None or not xa<w/2<xb:continue
                width=(xb-xa)/w
                if not .12<=width<=.82:continue
                ys=np.linspace(.60*h,.90*h,8)
                widths=[]
                for yy in ys:
                    x1,x2=at_y(a["points"],yy),at_y(b["points"],yy)
                    if x1 is not None and x2 is not None:
                        widths.append((x2-x1)/w)
                if len(widths)<4 or min(widths)<.06 or max(widths)>.95:continue
                confidence=min(float(a.get("confidence",0)),float(b.get("confidence",0)))
                score=2*confidence-abs((xa+xb)/2-w/2)/w-.35*width
                options.append((score,i,j))
        if options:
            score,i,j=max(options,key=lambda p:p[0])
            left,right=candidates[i],candidates[j]
            self.previous=(left["points"],right["points"])
            return left["points"],right["points"],dict(reason="centered",selected_channels=[left["channel"],right["channel"]],estimated=False,pair_score=float(score),pairs_considered=len(options),pairs_rejected_geometry=0)
        return [],[],dict(reason="no_ego_pair",selected_channels=[],estimated=False,pair_score=None,pairs_considered=0,pairs_rejected_geometry=0)
