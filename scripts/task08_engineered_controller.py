import mujoco
import numpy as np
from scipy.optimize import least_squares


class Controller:
    def reset(self, context, seed):
        self.context=context
        self.m=mujoco.MjModel.from_binary_path(context['model_file'])
        self.d=mujoco.MjData(self.m)
        self.output=np.array([self.m.actuator(n).id for n in context['actuator_names']])
        self.acts=np.array([self.m.actuator(n).id for n in context['arm_actuator_names']])
        self.wheels=np.array([self.m.actuator(n).id for n in context['wheel_actuator_names']])
        joints=np.array([self.m.joint(n).id for n in context['arm_joint_names']])
        self.qa=self.m.jnt_qposadr[joints];self.va=self.m.jnt_dofadr[joints]
        base=self.m.joint(context['base_joint']).id
        self.bq=int(self.m.jnt_qposadr[base]);self.bv=int(self.m.jnt_dofadr[base])
        self.ee=self.m.site(context['ee_site']).id
        self.q0=np.array(context['qpos'],float);self.qstart=self.q0[self.qa].copy()
        self.ctrl0=np.zeros(self.m.nu)
        self.ctrl0[self.output]=context['ctrl']
        self.lo=np.maximum(self.m.jnt_range[joints,0],self.m.actuator_ctrlrange[self.acts,0])+.005
        self.hi=np.minimum(self.m.jnt_range[joints,1],self.m.actuator_ctrlrange[self.acts,1])-.005
        self.target=np.array(context['target'],float)
        if context['arm']=='ur5e':
            self.lo[1]=max(self.lo[1],-3.05);self.hi[1]=min(self.hi[1],.1)
        self.shelf=[i for i in range(self.m.ngeom) if (self.m.geom(i).name or '').startswith('pod_')]
        self.shelf_map={g:i for i,g in enumerate(self.shelf)}
        self.arm_geoms=set()
        for i in range(self.m.ngeom):
            body=int(self.m.geom_bodyid[i])
            if (self.m.body(body).name or '').startswith('arm_') or 'payload' in (self.m.geom(i).name or ''):
                self.arm_geoms.add(i)
        self.m.geom_margin[self.shelf]=.025
        self.basegoal=np.array([.08,np.clip(self.target[1]*.6,-.12,.12)])
        rng=np.random.default_rng(seed)
        seeds=[self.qstart.copy()]
        for _ in range(10):
            seeds.append(np.clip(self.qstart+rng.normal(0,1.5,len(self.qa)),self.lo,self.hi))
        best=None
        for armseed in seeds:
            solution=self._solve(self.target,armseed,variable_base=True,max_nfev=90)
            if best is None or solution['cost']<best['cost']:best=solution
            if solution['error']<.006 and solution['contact']<.002:break
        self.basegoal=best['base']
        finish=best['q']
        path=[finish]
        q=finish
        for x in np.linspace(self.target[0],.22,7)[1:]:
            waypoint=self.target.copy();waypoint[0]=x
            solution=self._solve(waypoint,q,variable_base=False,max_nfev=80)
            q=solution['q'];path.append(q)
        self.path=np.array(path[::-1]);self.qpark=self.path[0]
        self.lastwheel=np.zeros(4);self.lastq=self.qstart.copy()
        self.enter_time=None
        self.plan_info={'final_ik_error_m':best['error'],'planning_contact_m':best['contact'],
            'base_goal':self.basegoal.tolist(),'waypoints':len(self.path)}

    def _state(self,q,base):
        self.d.qpos[:]=self.q0
        self.d.qpos[self.bq:self.bq+2]=base
        self.d.qpos[self.qa]=q
        self.d.qvel[:]=0
        mujoco.mj_forward(self.m,self.d)

    def _clearance(self):
        residual=np.zeros(len(self.shelf)+1)
        for i in range(self.d.ncon):
            contact=self.d.contact[i];a,b=int(contact.geom1),int(contact.geom2)
            if a in self.shelf_map and b in self.arm_geoms:
                j=self.shelf_map[a]
            elif b in self.shelf_map and a in self.arm_geoms:
                j=self.shelf_map[b]
            elif a in self.arm_geoms and self.m.geom_bodyid[b]==0 and b not in self.shelf_map:
                j=len(self.shelf)
            elif b in self.arm_geoms and self.m.geom_bodyid[a]==0 and a not in self.shelf_map:
                j=len(self.shelf)
            else:continue
            residual[j]=max(residual[j],.012-float(contact.dist),0.)
        return residual

    def _solve(self,target,initial,variable_base=False,max_nfev=80):
        n=len(self.qa);base0=self.basegoal.copy()
        x0=np.r_[np.clip(initial,self.lo,self.hi),base0] if variable_base else np.clip(initial,self.lo,self.hi)
        lo=np.r_[self.lo,[-.02,-.16]] if variable_base else self.lo
        hi=np.r_[self.hi,[.10,.16]] if variable_base else self.hi
        def residual(x):
            q=x[:n];base=x[n:] if variable_base else base0
            self._state(q,base)
            values=[20*(self.d.site_xpos[self.ee]-target),25*self._clearance(),.012*(q-initial)]
            if variable_base:values.append(.25*(base-base0))
            return np.concatenate(values)
        solution=least_squares(residual,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5,diff_step=1e-4)
        q=solution.x[:n];base=solution.x[n:] if variable_base else base0
        self._state(q,base)
        error=float(np.linalg.norm(self.d.site_xpos[self.ee]-target));contact=float(np.max(self._clearance()))
        return {'q':q.copy(),'base':base.copy(),'error':error,'contact':contact,'cost':error+3*contact}

    @staticmethod
    def smooth(x):
        x=np.clip(x,0.,1.);return x*x*(3-2*x)

    def act(self, observation):
        t=float(observation['time']);q=np.asarray(observation['qpos'],float);v=np.asarray(observation['qvel'],float)
        self.d.qpos[:]=q;self.d.qvel[:]=v;mujoco.mj_forward(self.m,self.d)
        quat=q[self.bq+3:self.bq+7]
        yaw=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
        base=q[self.bq:self.bq+2]
        park_phase=self.smooth((t-.5)/4.)
        qdes=self.qstart+(self.qpark-self.qstart)*park_phase
        drive_fraction=self.smooth((t-3.5)/7.)
        targetbase=self.q0[self.bq:self.bq+2]*(1-drive_fraction)+self.basegoal*drive_fraction
        error=targetbase-base
        co,si=np.cos(yaw),np.sin(yaw)
        velocity=np.clip(2.5*error,-.16,.16)
        vx=co*velocity[0]+si*velocity[1];vy=-si*velocity[0]+co*velocity[1]
        wz=np.clip(-2.5*yaw,-.3,.3)
        wheel=np.array([vx-vy-.427*wz,vx+vy+.427*wz,vx+vy-.427*wz,vx-vy+.427*wz])/.076
        wheel=np.clip(wheel,self.lastwheel-.15,self.lastwheel+.15);self.lastwheel=wheel
        aligned=np.linalg.norm(base-self.basegoal)<.025 and abs(yaw)<.06
        if self.enter_time is None and t>10.5 and aligned:self.enter_time=t
        if self.enter_time is not None:
            progress=self.smooth((t-self.enter_time)/7.)*(len(self.path)-1)
            index=min(len(self.path)-2,int(progress));fraction=progress-index
            qdes=self.path[index]*(1-fraction)+self.path[index+1]*fraction
            if progress>=len(self.path)-1:qdes=self.path[-1].copy()
            if t-self.enter_time>7.5:
                jac=np.zeros((3,self.m.nv));rot=np.zeros_like(jac)
                mujoco.mj_jacSite(self.m,self.d,jac,rot,self.ee)
                j=jac[:,self.va]
                correction=j.T@np.linalg.solve(j@j.T+np.eye(3)*.003,self.target-self.d.site_xpos[self.ee])
                qdes=np.clip(q[self.qa]+np.clip(correction,-.02,.02),self.path[-1]-.12,self.path[-1]+.12)
        gain=np.maximum(self.m.actuator_gainprm[self.acts,0],1.)
        compensation=self.d.qfrc_bias[self.va]/gain
        ctrl=self.ctrl0.copy();ctrl[self.acts]=np.clip(qdes+compensation,self.lo,self.hi);ctrl[self.wheels]=wheel
        return ctrl[self.output].tolist()


def make_controller():
    return Controller()
