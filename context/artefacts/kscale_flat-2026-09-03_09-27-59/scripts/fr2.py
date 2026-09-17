import cv2, numpy as np, os
B="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
OUT="/root/.claude/jobs/b40fca2a/tmp/frames"
def sheet(path,out,idxs,cols=4,scale=0.35):
    cap=cv2.VideoCapture(path); imgs=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read()
        if ok: imgs.append(cv2.resize(f,None,fx=scale,fy=scale))
    cap.release()
    if not imgs: return None
    h,w=imgs[0].shape[:2]; rows=(len(imgs)+cols-1)//cols
    canv=np.zeros((rows*h,cols*w,3),np.uint8)
    for k,im in enumerate(imgs):
        r,c=divmod(k,cols); canv[r*h:(r+1)*h,c*w:(c+1)*w]=im
    cv2.imwrite(out,canv); return canv.shape
for tag,run in [("n1015","2026-09-04_10-15-18"),("n1040","2026-09-04_10-40-30")]:
    p=f"{B}/{run}/videos/train/rl-video-step-320000.mp4"
    cap=cv2.VideoCapture(p); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
    print(tag,run,"frames",n,"fps",fps)
    print("  span ",sheet(p,f"{OUT}/{tag}_contact.png",np.linspace(0,n-1,12).astype(int)))
    print("  cycle",sheet(p,f"{OUT}/{tag}_cycle.png",np.arange(600,600+36,3)))
    # last-step video too
    p2=f"{B}/{run}/videos/train/rl-video-step-710000.mp4"
    if os.path.exists(p2):
        cap=cv2.VideoCapture(p2); n2=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        print("  final",sheet(p2,f"{OUT}/{tag}_final_cycle.png",np.arange(600,600+36,3)))
