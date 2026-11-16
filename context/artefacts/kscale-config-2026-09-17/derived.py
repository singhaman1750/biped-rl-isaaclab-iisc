import numpy as np
SCALE=0.4; DT=0.005
# joint: K, D, armature, effort, saturation, v_limit, v_sim, I_swing, I_stance, nominal, lo, hi
NOW={
 'hip_pitch' :(200.0,21.5,0.010, 84.0,120.0,17.488,15.0,1.16161,1.17330,-0.20369,-2.21657,1.04720),
 'hip_roll'  :(250.0,28.0,0.010, 42.0, 60.0,18.849,20.0,1.00422,1.54513, 0.042951,-2.26893,0.20944),
 'hip_yaw'   :(100.0, 3.5,0.010, 42.0, 60.0,18.849,20.0,0.03480,0.53437, 0.0,-1.57080,1.57080),
 'knee'      :(300.0,12.0,0.015, 84.0,120.0,17.488,15.0,0.12066,1.41283, 0.51239, 0.0,2.70526),
 'foot_pitch':(120.0, 1.0,0.005, 11.9, 17.0,37.699,44.0,0.00760,4.70450,-0.30870,-0.87267,0.52360),
 'foot_roll' :(120.0, 1.0,0.005, 11.9, 17.0,37.699,44.0,0.00574,5.15001, 0.041254,-0.26180,0.26180)}
RUNA={
 'hip_pitch' :(200.0,22.4,0.010,60.0,60.0,20.0,20.0,1.17000,1.18166),
 'hip_roll'  :(150.0,17.3,0.010,60.0,60.0,20.0,20.0,1.01697,1.60635),
 'hip_yaw'   :( 15.0, 0.9,0.010,60.0,60.0,20.0,20.0,0.02763,0.61603),
 'knee'      :(200.0,10.0,0.015,60.0,60.0,20.0,20.0,0.12076,1.44496),
 'foot_pitch':( 50.0, 1.7,0.005,17.0,17.0,10.0,10.0,0.00760,4.85781),
 'foot_roll' :( 20.0, 0.5,0.005,17.0,17.0,10.0,10.0,0.00574,5.37942)}
P95={'hip_pitch':3.39,'hip_roll':1.51,'hip_yaw':5.36,'knee':6.40,'foot_pitch':9.64,'foot_roll':9.77}
HOLD={'hip_pitch':3.232,'hip_roll':23.634,'hip_yaw':4.792,'knee':5.184,'foot_pitch':5.589,'foot_roll':19.710}
T1={'hip_pitch':8.323,'hip_roll':4.233,'hip_yaw':1.284,'knee':16.900,'foot_pitch':6.128,'foot_roll':0.0}

def env(sat,E,v,q): return min(sat*(1-abs(q)/v), E)
print("="*104)
print("A. DC MOTOR ENVELOPE, now that saturation_effort and effort_limit are separate")
print("="*104)
print(f"{'joint':<12}{'E':>7}{'sat':>7}{'v_lim':>8}{'v_sim':>7}{'corner v':>10}{'env damp':>10}{'tau@p95':>9}{'tau@clamp':>11}")
for j,(K,D,a,E,S,v,vs,Isw,Ist,nom,lo,hi) in NOW.items():
    print(f"{j:<12}{E:>7.1f}{S:>7.1f}{v:>8.3f}{vs:>7.1f}{v*(1-E/S):>10.2f}{S/v:>10.2f}"
          f"{env(S,E,v,P95[j]):>9.2f}{min(S*(1-vs/v),E):>10.2f}{'!' if S*(1-vs/v)>0 else ' '}")
print("\n  corner v is the speed below which the full soft limit is available.")
print("  env damp is saturation_effort / velocity_limit, the viscous damping a saturated joint shows.")
print("  tau@clamp marked ! means torque is still available at the speed the solver refuses to exceed.")

print("\n"+"="*104)
print("B. JOINT TRAVEL AT THE NEW NOMINAL, right leg, soft factor 0.9")
print("="*104)
print(f"{'joint':<12}{'nominal':>9}{'soft lo':>9}{'soft hi':>9}{'to soft-':>9}{'to soft+':>9}{'to hard-':>9}{'to hard+':>9}{'hold sag':>10}{'sag/travel':>11}")
for j,(K,D,a,E,S,v,vs,Isw,Ist,nom,lo,hi) in NOW.items():
    mid=(lo+hi)/2; half=0.5*(hi-lo)*0.9
    slo,shi=mid-half,mid+half
    sag=HOLD[j]/K
    tight=min(abs(nom-lo),abs(hi-nom))
    print(f"{j:<12}{nom:>9.4f}{slo:>9.4f}{shi:>9.4f}{nom-slo:>9.4f}{shi-nom:>9.4f}"
          f"{nom-lo:>9.4f}{hi-nom:>9.4f}{sag:>10.4f}{sag/tight*100:>10.1f}%")

print("\n"+"="*104)
print("C. ACTION AUTHORITY, joint position action at scale 0.4, clip unset")
print("="*104)
print(f"{'joint':<12}{'K':>7}{'E':>7}{'q_sat':>8}{'a_sat':>8}{'run A a_sat':>13}{'note':>24}")
for j,(K,D,a,E,S,v,vs,Isw,Ist,nom,lo,hi) in NOW.items():
    qa=RUNA[j][3]/RUNA[j][0]/SCALE
    n='saturates inside 1 sigma' if E/K/SCALE<1.0 else ''
    print(f"{j:<12}{K:>7.0f}{E:>7.1f}{E/K:>8.3f}{E/K/SCALE:>8.2f}{qa:>13.2f}{n:>24}")

print("\n"+"="*104)
print("D. AGAINST RUN A, the only configuration this robot has walked under")
print("="*104)
hdr=f"{'joint':<12}{'K now':>7}{'K A':>7}{'D now':>7}{'D A':>6}{'z_sw now':>9}{'z_sw A':>8}{'z_st now':>9}{'z_st A':>8}{'b now':>7}{'b A':>6}"
print(hdr)
for j in NOW:
    K,D,a,E,S,v,vs,Isw,Ist,nom,lo,hi=NOW[j]; K2,D2,a2,E2,S2,v2,vs2,Isw2,Ist2=RUNA[j]
    zsw=D/(2*np.sqrt(K*Isw)); zst=D/(2*np.sqrt(K*Ist))
    zsw2=D2/(2*np.sqrt(K2*Isw2)); zst2=D2/(2*np.sqrt(K2*Ist2))
    print(f"{j:<12}{K:>7.0f}{K2:>7.0f}{D:>7.1f}{D2:>6.1f}{zsw:>9.3f}{zsw2:>8.3f}{zst:>9.3f}{zst2:>8.3f}"
          f"{D*DT/Isw:>7.3f}{D2*DT/Isw2:>6.3f}")
print("\n  b is D*dt/I_swing, the explicit integration stability number, divergent above 2.")

print("\n"+"="*104)
print("E. TORQUE MARGIN, single support")
print("="*104)
print(f"{'joint':<12}{'tau_1sup':>10}{'hold':>9}{'E now':>8}{'worst/E':>9}{'E run A':>9}{'worst/E_A':>11}{'verdict':>22}")
for j,(K,D,a,E,S,v,vs,Isw,Ist,nom,lo,hi) in NOW.items():
    w=max(T1[j],HOLD[j]); EA=RUNA[j][3]
    vd='EXCEEDS' if w>E else ('tight' if w>0.7*E else 'ok')
    print(f"{j:<12}{T1[j]:>10.3f}{HOLD[j]:>9.3f}{E:>8.1f}{w/E:>9.2f}{EA:>9.1f}{w/EA:>11.2f}{vd:>22}")
