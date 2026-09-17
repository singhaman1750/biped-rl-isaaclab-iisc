CFG={
 "REF 08-28":{"hip_yaw":(15,0.9,60,20),"hip_roll":(150,17.3,60,20),"hip_pitch":(200,22.4,60,20),
              "knee":(200,10.0,60,20),"foot_roll":(20,0.5,17,10),"foot_pitch":(50,1.7,17,10)},
 "A 06-50":  {"hip_yaw":(100,3.3,120,40),"hip_roll":(250,28.0,120,40),"hip_pitch":(200,21.5,120,40),
              "knee":(300,12.0,120,40),"foot_roll":(120,1.0,34,20),"foot_pitch":(120,1.3,34,20)},
 "B 07-39":  {"hip_yaw":(100,3.3,120,40),"hip_roll":(250,28.0,120,40),"hip_pitch":(200,21.5,120,40),
              "knee":(300,12.0,120,40),"foot_roll":(120,1.0,34,20),"foot_pitch":(120,1.3,34,20)},
}
SCALE={"REF 08-28":0.4,"A 06-50":0.25,"B 07-39":0.4}
VSIM=10.0   # PhysX max_joint_velocity from kscale.urdf velocity="10"
ROM={"hip_yaw":3.1416,"hip_roll":2.4784,"hip_pitch":3.2637,"knee":2.7053,"foot_roll":0.5236,"foot_pitch":1.3963}
J=["hip_yaw","hip_roll","hip_pitch","knee","foot_roll","foot_pitch"]
print("TORQUE AVAILABLE AT THE PhysX VELOCITY CLAMP (10 rad/s), per DCMotor four-quadrant curve")
print(f"{'joint':12s}" + "".join(f"{n:>26s}" for n in CFG))
print(f"{'':12s}" + "".join(f"{'tau@10rad/s   (frac tau_lim)':>26s}" for n in CFG))
for j in J:
    row=f"{j:12s}"
    for n,c in CFG.items():
        K,D,el,vl=c[j]
        sat=el
        t=min(sat*(1.0-VSIM/vl), el)
        row+=f"{t:16.2f} ({t/el:5.2f})   "[:26]
    print(row)
print()
print("ACTION AUTHORITY, torque commanded by a unit action, and the target offset it asks for")
print(f"{'joint':12s}{'ROM(rad)':>10s}" + "".join(f"{n:>34s}" for n in CFG))
print(f"{'':12s}{'':>10s}" + "".join(f"{'K*s (Nm/a)  s (rad/a)  s/ROM':>34s}" for n in CFG))
for j in J:
    row=f"{j:12s}{ROM[j]:10.3f}"
    for n,c in CFG.items():
        K,D,el,vl=c[j]; s=SCALE[n]
        row+=f"{K*s:12.1f}{s:11.2f}{s/ROM[j]:11.2f}"
    print(row)
