"""UFLD-v2 CULane inference and candidate selection, for offline simulation only."""
from pathlib import Path
import sys
import cv2
import numpy as np

class PretrainedLaneDetector:
    def __init__(self, repo, weights, config="configs/culane_res18.py", device="auto"):
        import torch
        from PIL import Image
        self.torch, self.Image = torch, Image
        self.repo, self.weights = Path(repo).resolve(), Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if not (self.repo / "utils" / "common.py").is_file():
            raise FileNotFoundError(self.repo)
        sys.path.insert(0, str(self.repo))
        from utils.common import get_model
        from utils.config import Config
        self.cfg = Config.fromfile(str(self.repo / config))
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.net = get_model(self.cfg)
        checkpoint = torch.load(str(self.weights), map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.net.load_state_dict(state, strict=True)
        self.net.to(self.device).eval()
        from ego_lane_v16 import EgoLaneTracker
        self.tracker = EgoLaneTracker()
        self.last_debug = {}

    @staticmethod
    def _at_y(points, y):
        a = np.asarray(points, float)
        if a.ndim != 2 or a.shape[0] < 2:
            return None
        a = a[np.argsort(a[:,1])]
        if y < a[0,1] or y > a[-1,1]:
            return None
        return float(np.interp(y, a[:,1], a[:,0]))

    def reset_tracking(self):
        self.tracker.reset()

    def predict(self, frame, allow_recovery=False):
        import torchvision.transforms as transforms
        h,w = frame.shape[:2]
        cfg = self.cfg
        transform = transforms.Compose([
            transforms.Resize((int(cfg.train_height/cfg.crop_ratio),cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((.485,.456,.406),(.229,.224,.225))
        ])
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        tensor = transform(self.Image.fromarray(rgb))[:,-cfg.train_height:,:].unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.net(tensor)
        loc = output["loc_row"][0].detach().cpu()
        exist_logits = output["exist_row"][0].detach().cpu()
        existence = exist_logits.softmax(0)[1].numpy()
        num_rows, num_lanes, num_grid = int(loc.shape[1]), int(loc.shape[2]), int(loc.shape[0])
        anchors = np.linspace(.42,1.0,num_rows,dtype=float)
        candidates=[]
        for lane in range(num_lanes):
            points=[]
            support=[]
            for j,y in enumerate(anchors):
                p=float(existence[j,lane])
                if p<.50:continue
                scores=loc[:,j,lane]
                best=int(scores.argmax())
                indices=self.torch.arange(max(0,best-1),min(num_grid-1,best+1)+1)
                x=float((scores[indices].softmax(0)*indices.float()).sum()+.5)/num_grid*w
                points.append([x,float(y*h)])
                support.append(p)
            if len(points)>=2:
                candidates.append(dict(channel=lane,points=points,confidence=float(np.mean(support)),support=len(points)/num_rows))
        valid=[c for c in candidates if len(c["points"])>=6 and c["support"]>=.20]
        left,right,decision=self.tracker.select(valid,w,h,allow_recovery=allow_recovery)
        self.last_debug={
            "candidates":candidates,
            "selected_channels":decision["selected_channels"],
            "candidate_count":len(candidates),
            "valid_candidate_count":len(valid),
            "selected_pair":bool(left and right),
            "estimated":decision["estimated"],
            "num_rows":num_rows,
            "reason":decision["reason"],
            "pair_score":decision.get("pair_score"),
            "pairs_considered":decision.get("pairs_considered"),
            "pairs_rejected_geometry":decision.get("pairs_rejected_geometry")
        }
        return {"left_lane_boundary":left,"right_lane_boundary":right}
