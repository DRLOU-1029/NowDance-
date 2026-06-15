"""NowDance Step Tester - with camera capture mode."""
import sys,os,json,time,argparse,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))

def parse_steps(s):
    steps=[]
    for part in s.split(","):
        if "-" in part:
            a,b=part.split("-")
            steps.extend(range(int(a),int(b)+1))
        else:
            steps.append(int(part))
    return steps

def load_chart_seq(chart_path,seq_path):
    from nowdance.chart import Chart
    from nowdance.io import load_sequence
    return Chart.load(chart_path), load_sequence(seq_path)

def _draw_dual_view(cam_frame,ref_frame,cam_pose,ref_pose,step_num,elapsed,step_name,cv2):
    """绘制双视图：左侧摄像头+测试者骨架，右侧参考视频+骨架指引"""
    h,w=cam_frame.shape[:2]
    # 创建双倍宽度画布
    canvas=np.zeros((h,w*2,3),dtype=np.uint8)
    # 左侧：摄像头画面
    canvas[:,:w]=cam_frame
    # 叠加测试者骨架（绿色）
    if cam_pose is not None:
        canvas=_draw_skeleton_overlay(canvas,cam_pose,0,w,h,cv2,(0,255,0),(0,200,255))
    # 右侧：参考视频
    if ref_frame is not None:
        ref_resized=cv2.resize(ref_frame,(w,h))
        canvas[:,w:]=ref_resized
        # 叠加骨架指引（黄色）
        if ref_pose is not None:
            canvas=_draw_skeleton_overlay(canvas,ref_pose,w,w,h,cv2,(0,255,255),(255,128,0))
    # 绘制分隔线
    cv2.line(canvas,(w,0),(w,h),(255,255,255),2)
    # 标题信息
    cv2.putText(canvas,f"Step {step_num}  {step_name[:30]}",(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(canvas,f"t={elapsed:.2f}s",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.putText(canvas,"Your Camera",(w//2-80,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.putText(canvas,"Reference",(w+w//2-60,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    return canvas

def _draw_skeleton_overlay(canvas,pose_frame,offset_x,width,height,cv2,line_color,point_color):
    """在画布指定区域绘制骨架"""
    from nowdance.schema import LANDMARK_INDEX
    connections=[
        ("left_shoulder","right_shoulder"),
        ("left_shoulder","left_elbow"),
        ("left_elbow","left_wrist"),
        ("right_shoulder","right_elbow"),
        ("right_elbow","right_wrist"),
        ("left_shoulder","left_hip"),
        ("right_shoulder","right_hip"),
        ("left_hip","right_hip"),
    ]
    # 计算关键点坐标
    points={}
    for name,idx in LANDMARK_INDEX.items():
        pt=pose_frame.keypoints[idx]
        if pt[3]>0.3:  # visibility threshold
            x=offset_x+int(pt[0]*width)
            y=int(pt[1]*height)
            points[name]=(x,y)
    # 绘制骨架连线
    for start,end in connections:
        if start in points and end in points:
            cv2.line(canvas,points[start],points[end],line_color,3,cv2.LINE_AA)
    # 绘制关键点
    for name,(x,y) in points.items():
        cv2.circle(canvas,(x,y),6,point_color,-1,cv2.LINE_AA)
    return canvas

def _show_score_screen(cam_frame,ref_frame,score,grade,step_name,cv2,no_pose=False):
    """显示得分画面（大字体）"""
    h,w=cam_frame.shape[:2]
    # 创建双倍宽度画布
    canvas=np.zeros((h,w*2,3),dtype=np.uint8)
    # 左侧：摄像头画面
    canvas[:,:w]=cam_frame
    # 右侧：参考视频
    if ref_frame is not None:
        ref_resized=cv2.resize(ref_frame,(w,h))
        canvas[:,w:]=ref_resized
    # 半透明遮罩
    overlay=canvas.copy()
    cv2.rectangle(overlay,(0,0),(w*2,h),(0,0,0),-1)
    canvas=cv2.addWeighted(canvas,0.4,overlay,0.6,0)
    # 确定颜色
    if no_pose:
        score_color=(100,100,100)
        grade_color=(100,100,100)
    elif grade=="Perfect":
        score_color=(0,255,255)  # 黄色
        grade_color=(0,215,255)
    elif grade=="Great":
        score_color=(0,255,0)  # 绿色
        grade_color=(0,200,0)
    elif grade=="Good":
        score_color=(255,255,0)  # 青色
        grade_color=(200,255,0)
    elif grade=="Okay":
        score_color=(255,165,0)  # 橙色
        grade_color=(255,140,0)
    else:
        score_color=(0,0,255)  # 红色
        grade_color=(0,0,200)
    # 显示步骤名称
    cv2.putText(canvas,step_name[:40],(w-200,80),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
    # 显示分数（超大字体）
    score_text=f"{score:.1f}"
    cv2.putText(canvas,score_text,(w-150,h//2-50),cv2.FONT_HERSHEY_SIMPLEX,4.0,score_color,8,cv2.LINE_AA)
    # 显示评级（超大字体）
    cv2.putText(canvas,grade,(w-150,h//2+80),cv2.FONT_HERSHEY_SIMPLEX,2.5,grade_color,6,cv2.LINE_AA)
    # 显示提示
    cv2.putText(canvas,"Press any key to continue...",(w-180,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2,cv2.LINE_AA)
    # 显示2秒或等待按键
    start=time.perf_counter()
    while time.perf_counter()-start<2.0:
        cv2.imshow("Step Tester",canvas)
        if cv2.waitKey(50)&0xFF in (27,ord('q')): break

def run_camera_test(chart,seq,step_nums,repeats,args):
    import cv2,mediapipe as mp
    from mediapipe.tasks.python import BaseOptions,vision
    from nowdance.collect import _landmarks_to_keypoints,_resize_for_inference
    from nowdance.schema import PoseFrame
    from nowdance.quality import evaluate_step

    camera=cv2.VideoCapture(args.camera_idx)
    if not camera.isOpened():
        print("Cannot open camera",args.camera);return
    ref=cv2.VideoCapture(args.video)
    if not ref.isOpened():
        print("Cannot open video",args.video);camera.release();return

    options=vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=0.5,min_pose_presence_confidence=0.5,min_tracking_confidence=0.5)

    ref_fps=ref.get(cv2.CAP_PROP_FPS)
    ts=np.array([f.timestamp for f in seq.frames])
    all_scores={s:[] for s in step_nums}
    mirror=not args.no_mirror

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for attempt in range(1,repeats+1):
            print(f"\n=== Attempt {attempt}/{repeats} ===")
            for sn in step_nums:
                step=[s for s in chart.steps if s.step_number==sn][0]
                print(f"  Step {sn}: {step.name[:25]} ({step.start_time:.1f}-{step.end_time:.1f}s)")

                # Countdown
                start_t=time.perf_counter()
                while time.perf_counter()-start_t<3:
                    remaining=3-(time.perf_counter()-start_t)
                    ok,f=camera.read()
                    if ok and mirror: f=cv2.flip(f,1)
                    if ok:
                        cv2.putText(f,f"Get Ready... {remaining:.0f}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        cv2.imshow("Step Tester",f)
                    if cv2.waitKey(50)&0xFF in (27,ord('q')): break

                # Capture during step
                cam_frames=[];start=time.perf_counter()
                while time.perf_counter()-start<step.end_time-step.start_time:
                    el=time.perf_counter()-start+step.start_time
                    # Read reference frame
                    ref.set(cv2.CAP_PROP_POS_FRAMES,int(el*ref_fps))
                    ok_r,ref_f=ref.read()
                    ok_c,cam_f=camera.read()
                    cam_pose=None
                    if ok_c:
                        if mirror: cam_f=cv2.flip(cam_f,1)
                        rgb=cv2.cvtColor(_resize_for_inference(cam_f,cv2,args.infer_w),cv2.COLOR_BGR2RGB)
                        img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                        res=landmarker.detect_for_video(img,int(time.perf_counter()*1000))
                        if res.pose_landmarks:
                            cam_pose=PoseFrame(timestamp=el,keypoints=_landmarks_to_keypoints(res.pose_landmarks[0]))
                            cam_frames.append(cam_pose)
                    # Find reference pose frame
                    ref_pose_frame=None
                    idx=np.argmin(np.abs(ts-el))
                    if idx<len(seq.frames): ref_pose_frame=seq.frames[idx]
                    # Display with both skeletons overlay
                    display=_draw_dual_view(cam_f,ref_f,cam_pose,ref_pose_frame,sn,el,step.name,cv2)
                    cv2.imshow("Step Tester",display)
                    if cv2.waitKey(1)&0xFF in (27,ord('q')): break

                # Evaluate
                if cam_frames:
                    report=evaluate_step(step,cam_frames)
                    all_scores[sn].append(round(report.score,1))
                    print(f"    Score: {all_scores[sn][-1]:.1f} ({report.grade})")
                    # 显示得分画面（大字体）
                    _show_score_screen(cam_f,ref_f,all_scores[sn][-1],report.grade,step.name,cv2)
                else:
                    all_scores[sn].append(0.0)
                    print("    No pose detected - 0.0")
                    # 显示无姿态检测画面
                    _show_score_screen(cam_f,ref_f,0.0,"Miss",step.name,cv2,no_pose=True)

    camera.release();ref.release();cv2.destroyAllWindows()

    # Summary
    print("\n"+"="*60)
    print("FINAL SUMMARY (Camera Test)")
    print("="*60)
    for sn in step_nums:
        scores=all_scores[sn]
        avg=np.mean(scores) if scores else 0
        if avg>=90:g="Perfect"
        elif avg>=75:g="Great"
        elif avg>=60:g="Good"
        elif avg>=40:g="Okay"
        else:g="Miss"
        parts="  ".join(f"{s:.1f}" for s in scores)
        print(f"  Step {sn:2d}: {parts}  Avg={avg:.1f} {g}")
    total=np.mean([s for ss in all_scores.values() for s in ss]) if any(all_scores.values()) else 0
    print(f"\nOverall: {total:.1f}")

def run_self_test(chart,seq,step_nums,repeats):
    from nowdance.quality import evaluate_step
    ts=np.array([f.timestamp for f in seq.frames])
    all_scores={s:[] for s in step_nums}
    for attempt in range(1,repeats+1):
        print(f"\n=== Attempt {attempt}/{repeats} ===")
        print("Get ready... (3s countdown)")
        for i in range(3,0,-1):
            print(f"  {i}...");time.sleep(1)
        print("  GO!")
        for sn in step_nums:
            step=[s for s in chart.steps if s.step_number==sn][0]
            mask=(ts>=step.start_time)&(ts<=step.end_time)
            idx=np.where(mask)[0]
            player_frames=[seq.frames[i] for i in idx]
            report=evaluate_step(step,player_frames)
            all_scores[sn].append(round(report.score,1))
        for sn in step_nums:
            print(f"  Step {sn:2d}: {all_scores[sn][-1]:5.1f} ({report.grade})")
    print("\nFINAL SUMMARY")
    for sn in step_nums:
        scores=all_scores[sn];avg=np.mean(scores)
        if avg>=90:g="Perfect"
        elif avg>=75:g="Great"
        elif avg>=60:g="Good"
        elif avg>=40:g="Okay"
        else:g="Miss"
        parts="  ".join(f"{s:5.1f}" for s in scores)
        print(f"  Step {sn:2d}: {parts}  Avg={avg:.1f} {g}")
    total=np.mean([s for ss in all_scores.values() for s in ss])
    print(f"\nOverall Average: {total:.1f}")

def main():
    parser=argparse.ArgumentParser(description="NowDance Step Tester")
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--steps",type=str)
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--camera",action="store_true",help="Camera capture mode")
    parser.add_argument("--camera-idx",type=int,default=0)
    parser.add_argument("--no-mirror",action="store_true")
    parser.add_argument("--video",default="assets/jljt.mp4")
    parser.add_argument("--model",default="models/pose_landmarker_lite.task")
    parser.add_argument("--infer-w",type=int,default=480)
    parser.add_argument("--chart",default="charts/jljt_chart.json")
    parser.add_argument("--sequence",default="charts/jljt.json")
    args=parser.parse_args()

    chart,seq=load_chart_seq(args.chart,args.sequence)

    if args.list:
        for s in chart.steps:
            tol=getattr(s,"tolerance",1.0)
            print(f"  Step {s.step_number:2d}: {s.start_time:5.1f}-{s.end_time:5.1f}s  tol={tol}  {s.name[:25]}")
        return

    if not args.steps:
        parser.print_help();return

    steps=parse_steps(args.steps)
    print(f"Testing: steps {steps}, {args.repeats} repeats")

    if args.camera:
        run_camera_test(chart,seq,steps,args.repeats,args)
    else:
        run_self_test(chart,seq,steps,args.repeats)

if __name__=="__main__":
    main()
