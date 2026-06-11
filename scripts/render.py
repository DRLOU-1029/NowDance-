import sys,os
sys.path.insert(0,os.getcwd())
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from nowdance.io import load_sequence
from nowdance.schema import POSE_LANDMARK_NAMES
import sys,os
sys.path.insert(0,os.getcwd())
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from nowdance.io import load_sequence
from nowdance.schema import POSE_LANDMARK_NAMES
data=json.load(open("charts/jljt_chart.json",encoding="utf-8"))
seq=load_sequence("charts/jljt.json")
import numpy as np
ts=np.array([f.timestamp for f in seq.frames])
sk=data["steps"]
cap=cv2.VideoCapture("assets/jljt.mp4")
fps=cap.get(cv2.CAP_PROP_FPS)
w=int(cap.get(3));h=int(cap.get(4));tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
wr=cv2.VideoWriter("charts/jljt_overlay.mp4",cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
ft=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",20)
us={"left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"}
CONN=[("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip")]
for fi in range(tot):
    ok,frame=cap.read()
    if not ok:break
    t=fi/fps
    pi=int(np.argmin(np.abs(ts-t)))
    kps=seq.frames[pi].keypoints
    cur=None
    for s in sk:
        if s["start_time"]<=t<=s["end_time"]:cur=s;break
    pp={}
    for i,kp in enumerate(kps):
        if i<len(POSE_LANDMARK_NAMES):pp[POSE_LANDMARK_NAMES[i]]=(int(np.clip(kp[0],0,1)*w),int(np.clip(kp[1],0,1)*h))
    ov=frame.copy()
    col=(150,150,150)
    if cur:col=(77,214,181)
    for a,b in CONN:
        if a in pp and b in pp:cv2.line(ov,pp[a],pp[b],col,4,cv2.LINE_AA)
    for n,p in pp.items():
        if n in us:cv2.circle(ov,p,7,(255,255,255),-1,cv2.LINE_AA);cv2.circle(ov,p,5,col,-1,cv2.LINE_AA)
    res=cv2.addWeighted(frame,0.5,ov,0.5,0)
    cv2.rectangle(res,(0,0),(w,44),(0,0,0),-1)
    pil=Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))
    dr=ImageDraw.Draw(pil)
    if cur:
        sn=str(cur.get("step_number","?"))
        nm=str(cur.get("name","?")[:28])
        dr.text((10,6),"Step "+sn+"/45: "+nm,font=ft,fill=col)
    dr.text((w-125,8),"t="+str(round(t,2))+"s",font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",14),fill=(200,200,200))
    wr.write(cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR))
    if fi%200==0:print("  "+str(fi)+"/"+str(tot))
cap.release();wr.release()
print("Done charts/jljt_overlay.mp4")
