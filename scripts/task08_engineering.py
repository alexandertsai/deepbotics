"""Public component assembly and numerical diagnostics for Task08 development."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

ARMS = {'panda': ('franka_emika_panda', 'panda_nohand.xml'),
        'ur5e': ('universal_robots_ur5e', 'ur5e.xml'),
        'xarm7': ('ufactory_xarm7', 'xarm7_nohand.xml')}
INITIAL = {'panda': [0., -.785, 0., -2.356, 0., 1.571, .785],
           'ur5e': [-.83943, -1.22370, -1.33717, -1.07738, -3.03934, 3.03382],
           'xarm7': [-2.26731, -.62319, -.00416, .22692, .02608, 2.72839, -.08641]}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = ROOT/'vendor/rle-bench/tasks/task08/environment/assets'
DEFAULT_START = ROOT/'outputs/task06-task08-parallel-15m-20261102-01/task08/deepseek/selected'


def public_root(path=None):
    if path is not None:
        return Path(path)
    return Path('/workspace') if Path('/workspace/scene.xml').exists() else DEFAULT_PUBLIC


def prepare(destination, source=None, wheel_x=.20, mount_height=.30, battery_x=-.04):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(source) if source is not None else DEFAULT_START
    root = ET.parse(source/'robot.xml').getroot()
    base = root.find('worldbody/body[@name="base"]')
    for corner in ('FL','FR','RL','RR'):
        wheel = base.find(f'body[@name="wheel_{corner}"]')
        position = np.fromstring(wheel.get('pos'), sep=' ')
        position[0] = wheel_x if corner.startswith('F') else -wheel_x
        wheel.set('pos',' '.join(map(str,position)))
    for name, sign in (('profile_2020_cross_front',1),('profile_2020_cross_rear',-1)):
        geom=base.find(f'geom[@name="{name}"]')
        position=np.fromstring(geom.get('pos'),sep=' ');position[0]=sign*wheel_x
        geom.set('pos',' '.join(map(str,position)))
    for name in ('profile_2040_rail_left','profile_2040_rail_right'):
        geom=base.find(f'geom[@name="{name}"]')
        size=np.fromstring(geom.get('size'),sep=' ');size[0]=wheel_x+.01
        geom.set('size',' '.join(map(str,size)));geom.set('mass',str(.82*2*size[0]))
    battery=base.find('body[@name="battery"]')
    position=np.fromstring(battery.get('pos'),sep=' ');position[0]=battery_x
    battery.set('pos',' '.join(map(str,position)))
    site=base.find('site[@name="arm_mount_site"]')
    position=np.fromstring(site.get('pos'),sep=' ');position[2]=mount_height
    site.set('pos',' '.join(map(str,position)))
    tower=base.find('geom[@name="profile_4040_tower"]')
    if tower is not None:
        half=(mount_height+.014)/2
        tower.set('size',f'.02 .02 {half}')
        tower.set('pos',f'0 0 {(mount_height-.034)/2}')
        tower.set('mass',str(1.5*2*half))
    for name in ('profile_2020_top_x','profile_2020_top_y'):
        geom=base.find(f'geom[@name="{name}"]')
        if geom is not None:
            position=np.fromstring(geom.get('pos'),sep=' ');position[2]=mount_height-.01
            geom.set('pos',' '.join(map(str,position)))
    arm=next(b for b in base.findall('body') if b.get('name','').startswith('arm_'))
    arm.set('pos',f'0 0 {mount_height+.01}')
    ET.indent(root,space=' ')
    ET.ElementTree(root).write(destination/'robot.xml',encoding='utf-8',xml_declaration=True)
    controller=Path(__file__).with_name('task08_engineered_controller.py')
    (destination/'controller.py').write_text(controller.read_text() if controller.exists() else (source/'controller.py').read_text())
    return {'destination':str(destination),'wheel_x':wheel_x,'mount_height':mount_height,'battery_x':battery_x}


def _prefix(tree):
    for mesh in tree.findall('asset/mesh'):
        if not mesh.get('name'):
            mesh.set('name',Path(mesh.get('file')).stem)
    names={e.get('name') for e in tree.iter() if e.get('name')}
    names.update(e.get('class') for e in tree.iter('default') if e.get('class'))
    mapping={name:'arm_'+name for name in names}
    reference={'name','class','childclass','joint','joint1','joint2','mesh','material','site','body','body1','body2','target','tendon','actuator'}
    for element in tree.iter():
        for key,value in list(element.attrib.items()):
            if key in reference:
                element.set(key,' '.join(mapping.get(token,token) for token in value.split()))
    return tree


def assemble(candidate, arm='panda', payload=1., public=None, target_index=0):
    public=public_root(public);candidate=Path(candidate)
    original=ET.parse(candidate/'robot.xml').getroot()
    base=deepcopy(original.find('worldbody/body[@name="base"]'))
    for child in list(base):
        if child.tag=='body' and child.get('name','').startswith('arm_'):
            base.remove(child)
    folder,filename=ARMS[arm]
    canonical=_prefix(ET.parse(public/'assets'/folder/filename).getroot())
    root=ET.Element('mujoco',{'model':'public_task08_development'})
    ET.SubElement(root,'compiler',{'angle':'radian','autolimits':'true','meshdir':str((public/'assets'/folder/'assets').resolve())})
    ET.SubElement(root,'option',{'integrator':'implicitfast','timestep':'0.002','impratio':'10'})
    for tag in ('default','asset'):
        item=canonical.find(tag)
        if item is not None:root.append(item)
    wb=ET.SubElement(root,'worldbody');wb.append(base)
    mount=base.find('site[@name="arm_mount_site"]')
    adapter=ET.SubElement(base,'body',{'name':'public_arm_adapter','pos':mount.get('pos')})
    ET.SubElement(adapter,'geom',{'name':'public_adapter_plate','type':'box','size':'.09 .09 .005','pos':'0 0 .005','mass':'.875'})
    arm_root=canonical.find('worldbody/body');arm_root.set('pos','0 0 .01')
    if arm=='ur5e':
        rotation=np.fromstring(arm_root.get('quat','1 0 0 0'),sep=' ');out=np.empty(4)
        mujoco.mju_mulQuat(out,np.array([0.,0.,0.,1.]),rotation)
        arm_root.set('quat',' '.join(map(str,out)))
    adapter.append(arm_root)
    ee=next(e for e in arm_root.iter('site') if e.get('name')=='arm_attachment_site')
    parent=next(e for e in arm_root.iter() if ee in list(e))
    load=ET.SubElement(parent,'body',{'name':'public_payload','pos':ee.get('pos','0 0 0'),'quat':ee.get('quat','1 0 0 0')})
    ET.SubElement(load,'geom',{'name':'public_payload_geom','type':'sphere','size':'.04','mass':str(payload),'rgba':'.8 .1 .1 1'})
    scene=ET.parse(public/'scene.xml').getroot()
    for child in scene.find('worldbody'):
        wb.append(deepcopy(child))
    contact=canonical.find('contact')
    if contact is not None:root.append(contact)
    acts=ET.SubElement(root,'actuator')
    for act in original.find('actuator'):
        if act.get('name','').startswith('motor_'):acts.append(deepcopy(act))
    for act in canonical.find('actuator'):acts.append(act)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    bid=model.joint('base_free').id;ba=model.jnt_qposadr[bid]
    data.qpos[ba:ba+7]=[-.60,0.,.08,1.,0.,0.,0.]
    arm_acts=[model.actuator(i).name for i in range(model.nu) if model.actuator(i).name.startswith('arm_')]
    arm_joints=[model.joint(int(model.actuator_trnid[model.actuator(n).id,0])).name for n in arm_acts]
    for n,q in zip(arm_joints,INITIAL[arm]):data.qpos[model.jnt_qposadr[model.joint(n).id]]=q
    for n,q in zip(arm_acts,INITIAL[arm]):data.ctrl[model.actuator(n).id]=q
    mujoco.mj_forward(model,data)
    spec=json.loads((public/'shelf_spec.json').read_text())['reach_goal']
    context={'arm':arm,'target':spec['flange_target_points_m'][target_index],
        'target_name':spec['target_names_in_order'][target_index],'payload_kg':payload,'shelf_face_x':.4,
        'actuator_names':[model.actuator(i).name for i in range(model.nu)],'arm_joint_names':arm_joints,
        'arm_actuator_names':arm_acts,'wheel_actuator_names':['motor_FL','motor_FR','motor_RL','motor_RR'],
        'base_joint':'base_free','ee_site':'arm_attachment_site','qpos':data.qpos.tolist(),
        'qvel':data.qvel.tolist(),'ctrl':data.ctrl.tolist(),'control_dt':.02,'duration':30.,'seed':0}
    return model,data,context


def compile_summary(candidate, public=None):
    result={}
    for arm in ARMS:
        model,data,context=assemble(candidate,arm,1.,public)
        base=model.body('base').id
        result[arm]={'nq':model.nq,'nu':model.nu,'mass_kg':float(model.body_subtreemass[base]),
                     'initial_flange':data.site_xpos[model.site(context['ee_site']).id].tolist()}
    return result


def diagnose(candidate, arm='panda', target_index=0, payload=1., seconds=30., public=None, trace_interval=2., target=None, split=None):
    if split is not None:
        began=time.monotonic();cases=[]
        targets=(0,3,5,6,8,11) if split=='validation' else tuple(range(12)) if split=='all' else (0,)
        payloads=(1.,2.) if split=='validation' else (1.,)
        for variant in ARMS:
            for index in targets:
                for load in payloads:
                    row=diagnose(candidate,variant,index,load,seconds,public,trace_interval=seconds+1)
                    row.pop('trace',None);cases.append(row)
        by_arm={variant:float(np.mean([r['selection_score'] for r in cases if r['arm']==variant])) for variant in ARMS}
        return {'split':split,'cases':cases,'by_arm':by_arm,
            'selection_score':.75*min(by_arm.values())+.25*float(np.mean(list(by_arm.values()))),
            'selection_formula':'0.75*minimum_arm_mean+0.25*all_arm_mean of per-case safety/hold/progress scores',
            'wall_seconds':time.monotonic()-began,'public_diagnostic_only':True}
    began=time.monotonic();candidate=Path(candidate)
    if target is not None:
        names=json.loads((public_root(public)/'shelf_spec.json').read_text())['reach_goal']['target_names_in_order']
        target_index=names.index(target) if isinstance(target,str) else int(target)
    model,data,context=assemble(candidate,arm,payload,public,target_index)
    sid=model.site(context['ee_site']).id;target=np.array(context['target'])
    shelf={i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('pod_')}
    wheels={model.body('wheel_'+corner).id for corner in ('FL','FR','RL','RR')}
    with tempfile.TemporaryDirectory(prefix='task08-public-') as temp:
        model_file=Path(temp)/'model.mjb';mujoco.mj_saveModel(model,str(model_file));context['model_file']=str(model_file)
        spec=importlib.util.spec_from_file_location('task08_candidate_controller',candidate/'controller.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        controller=module.make_controller();reset_at=time.monotonic();controller.reset(context,0);reset_seconds=time.monotonic()-reset_at
        trace=[];min_error=1e9;max_contact=0.;hold=best_hold=0.;tip=False;max_roll=0.;support_min=4
        minimum_margin=1e9;contact_pair=None;final_margin=-1.
        nsub=round(context['control_dt']/model.opt.timestep);steps=round(seconds/context['control_dt'])
        force=np.zeros(6)
        for k in range(steps):
            controls=np.asarray(controller.act({'time':float(data.time),'qpos':data.qpos.tolist(),'qvel':data.qvel.tolist(),'ctrl':data.ctrl.tolist()}))
            if controls.shape!=(model.nu,) or not np.all(np.isfinite(controls)):raise ValueError('Invalid full actuator control vector')
            data.ctrl[:]=controls
            for _ in range(nsub):mujoco.mj_step(model,data)
            error=float(np.linalg.norm(data.site_xpos[sid]-target));min_error=min(min_error,error)
            contact_force=0.;wheel_contact=set();ground_points=[]
            for contact_id in range(data.ncon):
                contact=data.contact[contact_id];g1,g2=int(contact.geom1),int(contact.geom2)
                if g1 in shelf or g2 in shelf:
                    mujoco.mj_contactForce(model,data,contact_id,force);value=float(np.linalg.norm(force[:3]));contact_force=max(contact_force,value)
                    if value>max_contact:contact_pair=[model.geom(g1).name or model.body(int(model.geom_bodyid[g1])).name,model.geom(g2).name or model.body(int(model.geom_bodyid[g2])).name]
                for gid in (g1,g2):
                    b=int(model.geom_bodyid[gid])
                    while b and b not in wheels:b=int(model.body_parentid[b])
                    if b in wheels:
                        wheel_contact.add(b)
                        other=g2 if gid==g1 else g1
                        if model.geom(other).name=='floor':ground_points.append(contact.pos[:2].copy())
            final_margin=-1.
            if len(ground_points)>=3:
                try:
                    points=np.array(ground_points);hull=points[ConvexHull(points).vertices]
                    center=data.subtree_com[model.body('base').id,:2]
                    edges=np.roll(hull,-1,axis=0)-hull;offsets=center-hull
                    final_margin=float(np.min((edges[:,0]*offsets[:,1]-edges[:,1]*offsets[:,0])/np.maximum(np.linalg.norm(edges,axis=1),1e-9)))
                except QhullError:pass
            if data.time>1:minimum_margin=min(minimum_margin,final_margin)
            max_contact=max(max_contact,contact_force)
            quat=data.qpos[3:7];roll=float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)));max_roll=max(max_roll,roll)
            tip=tip or roll>.35
            if data.time>1:support_min=min(support_min,len(wheel_contact))
            if error<.06 and contact_force<=2 and roll<.15 and final_margin>0:hold+=context['control_dt']
            else:hold=0.
            best_hold=max(best_hold,hold)
            if k==0 or (k+1)%max(1,round(trace_interval/context['control_dt']))==0:
                trace.append({'t':round(float(data.time),2),'base':np.round(data.qpos[:3],4).tolist(),
                    'flange':np.round(data.site_xpos[sid],4).tolist(),'error_m':round(error,4),
                    'shelf_force_n':round(contact_force,3),'support_wheels':len(wheel_contact)})
        progress=float(np.exp(-error/.10))
        reached=float(best_hold>=1.)
        safety=float(not tip and max_contact<=2.)
        selection_score=safety*(.7*reached+.3*progress)
        return {'arm':arm,'target':context['target_name'],'payload_kg':payload,'duration_s':seconds,
            'reset_seconds':reset_seconds,'wall_seconds':time.monotonic()-began,'final_error_m':error,
            'minimum_error_m':min_error,'longest_hold_s':best_hold,'max_shelf_contact_n':max_contact,
            'peak_shelf_contact_pair':contact_pair,'minimum_stability_margin_m':minimum_margin,'final_stability_margin_m':final_margin,
            'max_tilt_rad':max_roll,'tipped':tip,'minimum_support_wheels':support_min,'trace':trace,
            'selection_score':selection_score,
            'selection_formula':'safe*(0.7*one_second_hold+0.3*exp(-final_error_m/0.10)); safe requires no tip and shelf contact <=2N throughout',
            'planning':getattr(controller,'plan_info',None),'public_diagnostic_only':True}


def wheel_response(candidate, arm='panda', public=None):
    responses=[]
    for pattern in ([2,2,2,2],[-2,2,2,-2],[-2,2,-2,2]):
        model,data,context=assemble(candidate,arm,1.,public)
        wid=[model.actuator(n).id for n in context['wheel_actuator_names']]
        for _ in range(250):mujoco.mj_step(model,data)
        origin=data.qpos[:2].copy();data.ctrl[wid]=pattern
        for _ in range(500):mujoco.mj_step(model,data)
        q=data.qpos[3:7];yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        responses.append({'wheel_rad_s':pattern,'displacement_m':(data.qpos[:2]-origin).tolist(),'yaw_rad':float(yaw),'base_z':float(data.qpos[2])})
    return responses


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('candidate',type=Path);parser.add_argument('--arm',default='panda');parser.add_argument('--target',type=int,default=0);parser.add_argument('--seconds',type=float,default=30)
    args=parser.parse_args();print(json.dumps(diagnose(args.candidate,args.arm,args.target,seconds=args.seconds)))
