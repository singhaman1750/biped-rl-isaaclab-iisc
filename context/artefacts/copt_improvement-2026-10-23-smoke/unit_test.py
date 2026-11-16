import ast, math, torch
B='/workspace/isaaclab/biped/'
def cols_rule(m,C):
    P=-(-C//m); q=torch.arange(P).unsqueeze(1); j=torch.arange(m).unsqueeze(0)
    return (j*P+q)*C//(P*m)
for m,C in [(80,80),(40,80),(48,80),(64,80),(100,80),(16,80),(1,80),(80,64)]:
    c=cols_rule(m,C)
    assert set(c.flatten().tolist())==set(range(C)),(m,C)
    if m<=C: assert all(len(set(r.tolist()))==m for r in c),(m,C)
    cnt=torch.bincount(c.flatten(),minlength=C)
    print(f"m={m:3d} C={C} P={c.shape[0]} twice={(cnt>1).sum().item()} max={cnt.max().item()}")
src=open(B+'environments/environments/tasks/locomotion/mdp/curriculums.py').read()
fns=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in('reduce_tracking_rewards_std','_tracking_kernel_average')]
ns={'torch':torch,'ManagerBasedRLEnv':object,'Sequence':list}
exec(compile(ast.Module(body=fns,type_ignores=[]),'c','exec'),ns)
f=ns['reduce_tracking_rewards_std']
def orig(env, env_ids, term_name, minimum_std, interval, update_threshold=0.8, update_rate=0.995, starting_step=0.0):
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if env.common_step_counter < starting_step: return term_cfg.params["std"]
    if env.common_step_counter % interval == 0 and not getattr(env, "_suppress_curriculum", False):
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if torch.mean(rew) / env.max_episode_length > update_threshold * term_cfg.weight * env.step_dt:
            if term_cfg.params["std"] > minimum_std:
                term_cfg.params["std"] = term_cfg.params["std"] * update_rate
    return term_cfg.params["std"]
class T: pass
def mkenv(n=64):
    e=T(); e.num_envs=n; e.max_episode_length=1000; e.step_dt=0.02; e.common_step_counter=0
    t=T(); t.weight=2.0; t.params={'std':0.4}
    rm=T(); rm._episode_sums={'x':torch.zeros(n)}; rm.get_term_cfg=lambda name:t
    e.reward_manager=rm; return e,t
g=torch.Generator().manual_seed(0)
ea,ta=mkenv(); eb,tb=mkenv()
for step in range(1,1500):
    k=torch.rand(64,generator=g); ids=torch.randperm(64,generator=g)[:5]
    for e in (ea,eb):
        e.common_step_counter=step; e.reward_manager._episode_sums['x']=k*40.0
    assert f(ea,ids,'x',0.09,50,0.5,0.95,100)==orig(eb,ids,'x',0.09,50,0.5,0.95,100)
print('default path identical over 1500 steps, final std %.4f'%ta.params['std'])
kw=dict(smoothing_episodes=64,widen_threshold=0.55,maximum_std=0.4)
e,t=mkenv(); err=0.15; traj=[]
for step in range(1,8000):
    e.common_step_counter=step; s=t.params['std']
    e.reward_manager._episode_sums['x']=torch.full((64,),math.exp(-err**2/s**2)*40.0)+torch.randn(64,generator=g)*5
    f(e,torch.randperm(64,generator=g)[:8],'x',0.10,100,0.67,0.95,0,**kw); traj.append(t.params['std'])
s=traj[-1]; print('settled std %.4f kernel %.3f, distinct widths in last half %d'%(s,math.exp(-err**2/s**2),len(set(round(x,6) for x in traj[4000:]))))
e._suppress_curriculum=True; before=(e.__dict__['_tracking_kernel_average']['x'],t.params['std'])
for step in range(8000,8500,100):
    e.common_step_counter=step; f(e,slice(None),'x',0.10,100,0.67,0.95,0,**kw)
assert before==(e.__dict__['_tracking_kernel_average']['x'],t.params['std']); print('suppression leaves average and width untouched')
e._suppress_curriculum=False; err=0.5
for step in range(8500,20000):
    e.common_step_counter=step; s=t.params['std']
    e.reward_manager._episode_sums['x']=torch.full((64,),math.exp(-err**2/s**2)*40.0)
    f(e,torch.arange(8),'x',0.10,100,0.67,0.95,0,**kw)
print('after the error rises to 0.5, std %.4f (cap 0.4)'%t.params['std'])
e,t=mkenv(); err=0.02
for step in range(1,20000):
    e.common_step_counter=step; s=t.params['std']
    e.reward_manager._episode_sums['x']=torch.full((64,),math.exp(-err**2/s**2)*40.0)
    f(e,torch.arange(8),'x',0.10,100,0.67,0.95,0,**kw)
print('tiny error, std %.4f (floor 0.10 exact)'%t.params['std'])
R,P,m,n=5,2,64,16
ret=torch.randn(R,P,m*n).view(R,P,m,n); col=cols_rule(m,80); even=col%2==0
h=(ret[:,even].mean((0,1)),ret[:,~even].mean((0,1))); print('halves',tuple(h[0].shape),'fitness',tuple(ret.mean((0,1,2)).shape),'even cells',even.sum().item(),'odd',(~even).sum().item())
