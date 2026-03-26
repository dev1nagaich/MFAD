# -*- coding: utf-8 -*-
"""
Deepfake Detection via Pupil Shape + Corneal Light Reflection Analysis
Install: pip install opencv-python numpy scikit-learn matplotlib mediapipe tqdm

NOTE: This method works best on high-resolution images (512px+).
      On 256px images, corneal IoU signal is weak -- we compensate
      with upscaling and a richer set of eye-shape features.
"""
import os, cv2, numpy as np, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
import mediapipe as mp

CFG = dict(
    real_dir="dataset/real", fake_dir="dataset/fake", max_images=500,
    # Decision thresholds (used only in rule-based mode)
    biou_threshold=0.50, iou_threshold=0.20,
    # Upscale small images before processing
    upscale_to=512,
    reflection_thresh=200, boundary_width=4, results_dir="results",
)
os.makedirs(CFG["results_dir"], exist_ok=True)

LEFT_EYE_IDX  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

# ------------------------------------------------------------------ mediapipe
def make_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.4)

def get_landmarks_mp(img_bgr, face_mesh):
    h, w = img_bgr.shape[:2]
    res  = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x*w, p.y*h] for p in lm], dtype=np.float32)

def eye_bbox(lm, idx, pad=10):
    pts = lm[idx]; x1,y1=pts.min(0)-pad; x2,y2=pts.max(0)+pad
    return int(x1),int(y1),int(x2),int(y2)

def crop_region(img, x1,y1,x2,y2):
    h,w=img.shape[:2]
    return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

# ------------------------------------------------------------------ features
def segment_pupil_otsu(eye_gray):
    """Otsu-based pupil segmentation. Returns (mask, contour) or None."""
    if eye_gray is None or eye_gray.size == 0: return None
    blurred = cv2.GaussianBlur(cv2.equalizeHist(eye_gray),(5,5),0)
    _,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, k,iterations=1)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,k,iterations=2)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    h,w=eye_gray.shape; cx_img,cy_img=w/2.0,h/2.0
    best,best_score=None,-1
    for c in contours:
        area=cv2.contourArea(c)
        if area<20: continue
        perim=cv2.arcLength(c,True)
        if perim==0: continue
        circ=4*np.pi*area/(perim**2)
        M=cv2.moments(c)
        if M["m00"]==0: continue
        cx,cy=M["m10"]/M["m00"],M["m01"]/M["m00"]
        dist=np.sqrt((cx-cx_img)**2+(cy-cy_img)**2)
        score=circ*area/(1.0+dist)
        if score>best_score: best_score=score; best=c
    if best is None: return None
    mask=np.zeros_like(eye_gray)
    cv2.drawContours(mask,[best],-1,255,-1)
    return mask, best

def fit_ellipse_mask(contour, shape):
    if contour is None or len(contour)<5: return None
    try: ellipse=cv2.fitEllipse(contour)
    except cv2.error: return None
    mask=np.zeros(shape[:2],dtype=np.uint8)
    cv2.ellipse(mask,ellipse,255,-1)
    return mask

def boundary_pixels(mask, width):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*width+1,2*width+1))
    return cv2.subtract(cv2.dilate(mask,k),cv2.erode(mask,k))

def compute_biou(pm, em, width=4):
    if pm is None or em is None: return 0.5
    if pm.shape!=em.shape:
        em=cv2.resize(em,(pm.shape[1],pm.shape[0]),interpolation=cv2.INTER_NEAREST)
    bp=boundary_pixels(pm,width); be=boundary_pixels(em,width)
    inter=np.logical_and(bp>0,be>0).sum(); union=np.logical_or(bp>0,be>0).sum()
    return float(inter)/float(union) if union>0 else 0.5

def extract_corneal_limbus(eye_gray):
    blurred=cv2.GaussianBlur(eye_gray,(5,5),0); h,w=eye_gray.shape
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,dp=1.2,
        minDist=max(1,w//2),param1=50,param2=15,
        minRadius=max(4,min(h,w)//8),maxRadius=max(10,min(h,w)//2))
    if circles is None: return None
    cx,cy,r=circles[0][0]; return int(cx),int(cy),int(r)

def reflection_mask(eye_gray, limbus, thresh_val=200):
    if limbus is not None:
        disc=np.zeros_like(eye_gray)
        cv2.circle(disc,(limbus[0],limbus[1]),limbus[2],255,-1)
    else:
        h,w=eye_gray.shape; disc=np.zeros_like(eye_gray)
        cv2.ellipse(disc,(w//2,h//2),(int(w*0.35),int(h*0.45)),0,0,360,255,-1)
    _,bright=cv2.threshold(eye_gray,thresh_val,255,cv2.THRESH_BINARY)
    return cv2.bitwise_and(bright,disc)

def compute_iou(a, b):
    if a is None or b is None: return 0.0
    if a.shape!=b.shape:
        b=cv2.resize(b,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_NEAREST)
    inter=np.logical_and(a>0,b>0).sum(); union=np.logical_or(a>0,b>0).sum()
    return float(inter)/float(union) if union>0 else 0.0

def contour_irregularity(contour):
    """
    Extra shape features that capture GAN artifacts:
      - solidity          : area / convex_hull_area  (dips when contour is jagged)
      - convexity_ratio   : perimeter / convex_hull_perimeter
      - aspect_ratio      : min_axis / max_axis of fitted ellipse
      - hu_moment_1       : first Hu moment (shape complexity)
    """
    if contour is None or len(contour)<5:
        return dict(solidity=1.0,convexity=1.0,aspect=1.0,hu1=0.0)
    area=cv2.contourArea(contour)
    perim=cv2.arcLength(contour,True)
    hull=cv2.convexHull(contour)
    hull_area=cv2.contourArea(hull); hull_perim=cv2.arcLength(hull,True)
    solidity=float(area)/float(hull_area)  if hull_area>0 else 1.0
    convexity=float(hull_perim)/float(perim) if perim>0 else 1.0
    try:
        _,(ma,MA),_=cv2.fitEllipse(contour)
        aspect=float(min(ma,MA))/float(max(ma,MA)) if max(ma,MA)>0 else 1.0
    except: aspect=1.0
    M=cv2.moments(contour)
    hu=cv2.HuMoments(M).flatten()
    hu1=float(-np.sign(hu[0])*np.log10(abs(hu[0])+1e-10))
    return dict(solidity=solidity,convexity=convexity,aspect=aspect,hu1=hu1)

# ------------------------------------------------------------------ per-image
def analyse_image(img_bgr, face_mesh, cfg):
    result = dict(biou_left=0.5,biou_right=0.5,avg_biou=0.5,
                  iou_reflect=0.0,prediction="real",landmarks_found=False,
                  solidity=1.0,convexity=1.0,aspect=1.0,hu1=0.0,
                  reflection_count=0)

    # Upscale small images -- critical for corneal reflection detection
    h0,w0=img_bgr.shape[:2]
    scale=cfg["upscale_to"]/max(h0,w0)
    if scale>1.0:
        img_bgr=cv2.resize(img_bgr,(int(w0*scale),int(h0*scale)),
                           interpolation=cv2.INTER_CUBIC)

    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    lm=get_landmarks_mp(img_bgr,face_mesh)
    if lm is None:
        result["prediction"]="fake"; return result
    result["landmarks_found"]=True

    biou_scores=[]; reflect_masks=[]; shape_feats=[]

    for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
        x1,y1,x2,y2=eye_bbox(lm,eye_idx,pad=10)
        eye_gray=crop_region(gray,x1,y1,x2,y2)
        if eye_gray.size==0: biou_scores.append(0.5); continue

        seg=segment_pupil_otsu(eye_gray)
        if seg is not None:
            pm,contour=seg
            em=fit_ellipse_mask(contour,eye_gray.shape)
            biou=compute_biou(pm,em,cfg["boundary_width"])
            feats=contour_irregularity(contour)
            shape_feats.append(feats)
        else:
            biou=0.5; shape_feats.append(dict(solidity=1.0,convexity=1.0,aspect=1.0,hu1=0.0))
        biou_scores.append(biou)

        limbus=extract_corneal_limbus(eye_gray)
        ref=reflection_mask(eye_gray,limbus,cfg["reflection_thresh"])
        reflect_masks.append(ref)
        result["reflection_count"]+=int((ref>0).sum())

    avg_biou=float(np.mean(biou_scores)) if biou_scores else 0.5
    result["biou_left"] =biou_scores[0] if len(biou_scores)>0 else 0.5
    result["biou_right"]=biou_scores[1] if len(biou_scores)>1 else 0.5
    result["avg_biou"]  =avg_biou

    iou_ref=compute_iou(reflect_masks[0],reflect_masks[1]) if len(reflect_masks)==2 else 0.0
    result["iou_reflect"]=iou_ref

    if shape_feats:
        result["solidity"]  =float(np.mean([f["solidity"]   for f in shape_feats]))
        result["convexity"] =float(np.mean([f["convexity"]  for f in shape_feats]))
        result["aspect"]    =float(np.mean([f["aspect"]     for f in shape_feats]))
        result["hu1"]       =float(np.mean([f["hu1"]        for f in shape_feats]))

    A,B=cfg["biou_threshold"],cfg["iou_threshold"]
    result["prediction"]="fake" if (avg_biou<A and iou_ref<B) else "real"
    return result

def feature_vector(r):
    """Return the full feature vector used for ML-based classification."""
    return [r["avg_biou"], r["iou_reflect"], r["solidity"],
            r["convexity"], r["aspect"], r["hu1"], r["reflection_count"]]

# ------------------------------------------------------------------ dataset
EXTENSIONS={".jpg",".jpeg",".png",".webp",".bmp"}
def load_image_paths(folder, max_n=None):
    paths=sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTENSIONS])
    return paths[:max_n] if max_n else paths

# ------------------------------------------------------------------ diagnostics
def print_score_stats(records):
    rb=[r["avg_biou"]    for r in records if r["true_label"]=="real"]
    fb=[r["avg_biou"]    for r in records if r["true_label"]=="fake"]
    ri=[r["iou_reflect"] for r in records if r["true_label"]=="real"]
    fi=[r["iou_reflect"] for r in records if r["true_label"]=="fake"]
    rs=[r["solidity"]    for r in records if r["true_label"]=="real"]
    fs=[r["solidity"]    for r in records if r["true_label"]=="fake"]
    lf=sum(1 for r in records if not r["landmarks_found"])
    print(f"\n--- Score diagnostics ---")
    print(f"  No-face skipped : {lf}/{len(records)}")
    def row(name,a,b):
        if a and b:
            print(f"  {name:12s}  real mean={np.mean(a):.3f} std={np.std(a):.3f}  |  "
                  f"fake mean={np.mean(b):.3f} std={np.std(b):.3f}  "
                  f"  sep={abs(np.mean(a)-np.mean(b)):.3f}")
    row("BIoU",rb,fb); row("IoU",ri,fi); row("Solidity",rs,fs)
    print("-------------------------")

# ------------------------------------------------------------------ evaluate
def evaluate(cfg):
    print("Initialising MediaPipe FaceMesh ...")
    face_mesh=make_face_mesh()
    real_paths=load_image_paths(cfg["real_dir"],cfg["max_images"])
    fake_paths=load_image_paths(cfg["fake_dir"],cfg["max_images"])
    print(f"Real images : {len(real_paths)}\nFake images : {len(fake_paths)}")
    records=[]
    def process_batch(paths, label):
        for p in tqdm(paths,desc=f"  {label:4s}"):
            img=cv2.imread(str(p))
            if img is None: continue
            res=analyse_image(img,face_mesh,cfg)
            res["true_label"]=label; res["path"]=str(p); records.append(res)
    process_batch(real_paths,"real"); process_batch(fake_paths,"fake")
    face_mesh.close()
    print_score_stats(records)

    y_true=np.array([1 if r["true_label"]=="fake" else 0 for r in records])
    y_pred=np.array([1 if r["prediction"]=="fake"  else 0 for r in records])
    acc=accuracy_score(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred)
    report=classification_report(y_true,y_pred,target_names=["real","fake"],zero_division=0)
    print("\n"+"="*55+f"\n  Rule-based Accuracy : {acc*100:.2f}%\n"+"="*55)
    print(report)
    print(f"  CM: real({cm[0,0]}/{cm[0,0]+cm[0,1]}) fake({cm[1,1]}/{cm[1,0]+cm[1,1]})")
    scores=np.array([1.0-r.get("avg_biou",0.5) for r in records])
    fpr,tpr,_=roc_curve(y_true,scores); roc_auc=auc(fpr,tpr)
    print(f"  AUC (BIoU alone) : {roc_auc:.3f}")

    # ---- Logistic Regression on all features (shows upper bound of signal) ----
    X=np.array([feature_vector(r) for r in records])
    scaler=StandardScaler(); X_s=scaler.fit_transform(X)
    from sklearn.model_selection import cross_val_score
    lr=LogisticRegression(max_iter=1000)
    cv_accs=cross_val_score(lr,X_s,y_true,cv=5,scoring="accuracy")
    print(f"\n  LogReg 5-fold CV accuracy: {cv_accs.mean()*100:.2f}% +/- {cv_accs.std()*100:.2f}%")
    print("  (This shows how much discriminative signal exists in the eye features)")

    _plot_cm(cm,cfg["results_dir"]); _plot_roc(fpr,tpr,roc_auc,cfg["results_dir"])
    _plot_dist(records,cfg["results_dir"])
    return records,acc,roc_auc

# ------------------------------------------------------------------ plots
def _plot_cm(cm, out_dir):
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["real","fake"]); ax.set_yticklabels(["real","fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    [ax.text(j,i,str(cm[i,j]),ha="center",va="center",
     color="white" if cm[i,j]>cm.max()/2 else "black") for i in range(2) for j in range(2)]
    plt.colorbar(im,ax=ax); plt.tight_layout()
    out=os.path.join(out_dir,"confusion_matrix.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[saved] {out}")

def _plot_roc(fpr,tpr,roc_auc,out_dir):
    plt.figure(figsize=(5,4)); plt.plot(fpr,tpr,lw=2,label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    out=os.path.join(out_dir,"roc_curve.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[saved] {out}")

def _plot_dist(records, out_dir):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    features=[
        ("avg_biou","BIoU (Pupil Shape)"),
        ("iou_reflect","IoU (Corneal Reflect)"),
        ("solidity","Pupil Solidity"),
        ("convexity","Pupil Convexity"),
        ("aspect","Pupil Aspect Ratio"),
        ("hu1","Hu Moment 1"),
    ]
    for ax,(key,title) in zip(axes.flat,features):
        rv=[r[key] for r in records if r["true_label"]=="real"]
        fv=[r[key] for r in records if r["true_label"]=="fake"]
        ax.hist(rv,bins=30,alpha=0.6,label="real",color="steelblue")
        ax.hist(fv,bins=30,alpha=0.6,label="fake",color="tomato")
        if rv: ax.axvline(np.mean(rv),color="steelblue",linestyle="--",lw=1.5)
        if fv: ax.axvline(np.mean(fv),color="tomato",   linestyle="--",lw=1.5)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    out=os.path.join(out_dir,"score_distributions.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[saved] {out}")

# ------------------------------------------------------------------ sweep
def sweep_thresholds(records):
    y_true  =np.array([1 if r["true_label"]=="fake" else 0 for r in records])
    biou_arr=np.array([r.get("avg_biou",0.5)    for r in records])
    iou_arr =np.array([r.get("iou_reflect",0.0) for r in records])
    print("\nRunning threshold sweep ...")
    best_acc,best_A,best_B=0.0,0.5,0.2; grid_results=[]
    for A in np.arange(0.05,0.99,0.03):
        for B in np.arange(0.0,0.80,0.03):
            preds=((biou_arr<A)&(iou_arr<B)).astype(int)
            acc=accuracy_score(y_true,preds); grid_results.append((acc,float(A),float(B)))
            if acc>best_acc: best_acc,best_A,best_B=acc,float(A),float(B)
    print(f"\nBest thresholds -> A={best_A:.2f}  B={best_B:.2f}  accuracy={best_acc*100:.2f}%")
    A_vals=sorted(set(round(r[1],4) for r in grid_results))
    B_vals=sorted(set(round(r[2],4) for r in grid_results))
    g=np.zeros((len(B_vals),len(A_vals)))
    for acc,A,B in grid_results:
        g[B_vals.index(round(B,4)),A_vals.index(round(A,4))]=acc
    plt.figure(figsize=(9,6))
    plt.imshow(g,aspect="auto",origin="lower",cmap="RdYlGn",
               extent=[A_vals[0],A_vals[-1],B_vals[0],B_vals[-1]])
    plt.colorbar(label="Accuracy"); plt.xlabel("A (BIoU threshold)"); plt.ylabel("B (IoU threshold)")
    plt.title("Threshold Sweep - Accuracy Heatmap")
    plt.scatter([best_A],[best_B],c="blue",s=80,zorder=5,label=f"best ({best_A:.2f},{best_B:.2f})")
    plt.legend()
    out=os.path.join(CFG["results_dir"],"sweep_heatmap.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[saved] {out}")
    return best_A,best_B,best_acc

# ------------------------------------------------------------------ demo
def demo_single(image_path, cfg):
    face_mesh=make_face_mesh(); img=cv2.imread(image_path)
    assert img is not None,f"Cannot read: {image_path}"
    res=analyse_image(img,face_mesh,cfg); face_mesh.close()
    print(f"\nImage      : {image_path}")
    for k in ["biou_left","biou_right","avg_biou","iou_reflect","solidity","aspect","prediction"]:
        print(f"  {k:15s}: {res[k]}")
    return res

# ------------------------------------------------------------------ entry
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",    default="eval",choices=["eval","demo","sweep"])
    parser.add_argument("--image",   default=None)
    parser.add_argument("--real_dir",default=CFG["real_dir"])
    parser.add_argument("--fake_dir",default=CFG["fake_dir"])
    parser.add_argument("--max",     type=int,default=CFG["max_images"])
    args=parser.parse_args()
    CFG["real_dir"]=args.real_dir; CFG["fake_dir"]=args.fake_dir; CFG["max_images"]=args.max
    if args.mode=="demo":
        assert args.image,"--image required in demo mode"; demo_single(args.image,CFG)
    elif args.mode=="sweep":
        records,_,_=evaluate(CFG); sweep_thresholds(records)
    else:
        records,acc,roc_auc=evaluate(CFG)
        print(f"\nFinal -> Accuracy: {acc*100:.2f}%  |  AUC: {roc_auc:.3f}")
