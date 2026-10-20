"""Hybrid lane selection, preserving centered and curve-aware behavior."""
import numpy as np
from ego_lane_v15 import EgoLaneTracker as CurveTracker, at_y
from ego_lane_v14 import EgoLaneTracker as CenterTracker

class EgoLaneTracker:
    def __init__(self):
        self.curve=CurveTracker()
        self.center=CenterTracker()
        self.previous=None
    def reset(self):
        self.curve.reset(); self.center.reset(); self.previous=None
    def select(self,candidates,w,h,allow_recovery=False):
        curve=CurveTracker()
        center=CenterTracker()
        if self.previous is not None:
            curve.previous=self.previous
            center.previous=self.previous
        cl,cr,cd=curve.select(candidates,w,h,False)
        sl,sr,sd=center.select(candidates,w,h,False)
        chosen=None
        if sl and sr:
            ys=np.linspace(.65*h,.90*h,9)
            valid=[]
            for y in ys:
                a,b=at_y(sl,y),at_y(sr,y)
                if a is not None and b is not None:valid.append((a,b))
            if len(valid)>=6:
                widths=np.array([b-a for a,b in valid])/w
                centered=sum(a<.5*w<b for a,b in valid)/len(valid)
                if centered>=.55 and .08<=float(np.median(widths))<=.68:
                    chosen=(sl,sr,dict(sd,reason="centered_pair",selection_policy="centered"))
        if chosen is None and cl and cr:
            chosen=(cl,cr,dict(cd,selection_policy="curve"))
        if chosen is None and sl and sr:
            chosen=(sl,sr,dict(sd,selection_policy="center_fallback"))
        if chosen is not None:
            left,right,debug=chosen
            self.previous=(left,right)
            debug.setdefault("pairs_considered",cd.get("pairs_considered",0))
            debug.setdefault("pairs_rejected_geometry",cd.get("pairs_rejected_geometry",0))
            debug.setdefault("pair_score",cd.get("pair_score"))
            return left,right,debug
        if allow_recovery and self.previous is not None:
            curve.previous=self.previous
            left,right,debug=curve.select(candidates,w,h,True)
            if debug.get("estimated"):
                return left,right,dict(debug,selection_policy="temporal_estimate")
        return [],[],dict(cd,selection_policy="none")
