import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

SETTINGS={'top_z_tolerance':.005,'occlusion_margin':.012,'loss':'soft_l1','loss_scale':1.5,'max_nfev':24,'reacquire_interval':12,'tracking_angle':.4,'max_yaw_innovation':.7,'identity_lock_frames':2}


def red_mask(rgb):
    a=np.asarray(rgb,dtype=np.int16)
    m=((a[...,0]>65)&(a[...,0]-a[...,1]>35)&(a[...,0]-a[...,2]>35)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        keep=np.zeros(n,np.uint8)
        areas=stats[1:,cv2.CC_STAT_AREA]
        keep[1:]=(areas>max(35,.04*areas.max()))
        m=keep[labels]
    return m


def plane_points(obs,top_filter=True):
    mask=red_mask(obs['rgb']);yy,xx=np.nonzero(mask)
    if len(xx)<30:return np.empty((0,2))
    K=np.asarray(obs['K']);T=np.asarray(obs['T_cam_table'])
    rays=np.column_stack(((xx+.5-K[0,2])/K[0,0],-(yy+.5-K[1,2])/K[1,1],-np.ones(len(xx))))@T[:3,:3].T
    if top_filter and obs.get('depth') is not None:
        z=np.asarray(obs['depth'])[yy,xx]
        p=T[:3,3]+rays*z[:,None]
        good=np.isfinite(z)&(np.abs(p[:,2]-.04)<SETTINGS['top_z_tolerance'])
        if good.sum()>max(80,.3*len(xx)):rays=rays[good]
    s=(.04-T[2,3])/rays[:,2]
    return (T[:3,3]+s[:,None]*rays)[:,:2]


def raster_points(points,center,resolution=.001,size=300):
    image=np.zeros((size,size),np.uint8)
    ij=np.rint((points-center)/resolution+size/2).astype(int)
    ok=((ij>=0)&(ij<size)).all(1)
    image[ij[ok,1],ij[ok,0]]=1
    image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return image


def wrap(x):return (np.asarray(x)+np.pi)%(2*np.pi)-np.pi


def rotate(image,angle):
    n=image.shape[0];m=cv2.getRotationMatrix2D((n/2,n/2),-np.rad2deg(angle),1.)
    return cv2.warpAffine(image,m,(n,n),flags=cv2.INTER_LINEAR)


class Estimator:
    def __init__(self,templates=None,resolution=.001,top_filter=True):
        self.res=resolution;self.top_filter=top_filter
        if templates is None:templates=Path(__file__).with_name('templates.npz')
        data=np.load(templates)
        self.templates=np.asarray(data['templates'],np.float32)
        self.ids=np.asarray(data['ids'],int)
        self.offsets=np.asarray(data.get('offsets',np.zeros((len(self.ids),2))))
        self.size=self.templates.shape[1]
        self.coarse=[]
        self.angles=np.arange(-np.pi,np.pi,np.pi/18)
        for sid,tp in enumerate(self.templates):
            for angle in self.angles:
                im=cv2.resize(rotate(tp,float(angle)),None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
                self.coarse.append((sid,float(angle),im))
        self.reset()

    def reset(self):self.last=None;self.frame=0;self.votes=np.zeros(len(self.ids))

    def update(self,*,rgb,K,T_cam_table,t,depth=None):
        obs={'rgb':rgb,'depth':depth,'K':K,'T_cam_table':T_cam_table}
        points=plane_points(obs,self.top_filter)
        if len(points)<35:return self.last if self.last is not None else (0.,0.,0.,0)
        center=np.median(points,axis=0)
        n=self.size+120
        image=raster_points(points,center,self.res,n).astype(np.float32)
        coarse=cv2.resize(image,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA)
        choices=[]
        tracking=self.last is not None and self.frame%SETTINGS['reacquire_interval']!=0
        for sid,angle,tp in self.coarse:
            if self.last is not None and self.frame>=SETTINGS['identity_lock_frames'] and int(self.ids[sid])!=self.last[3]:continue
            if self.last is not None and abs(float(wrap(angle-self.last[2])))>SETTINGS['max_yaw_innovation']:continue
            if tracking and (self.ids[sid]!=self.last[3] or abs(float(wrap(angle-self.last[2])))>SETTINGS['tracking_angle']):continue
            score=cv2.matchTemplate(coarse,tp,cv2.TM_CCORR_NORMED)
            _,value,_,loc=cv2.minMaxLoc(score)
            choices.append((value,sid,angle,loc))
        choices.sort(reverse=True)
        best=None
        for _,sid,angle,loc in choices[:3]:
            for fine in np.arange(angle-.11,angle+.111,.022):
                tp=rotate(self.templates[sid],float(fine))
                score=cv2.matchTemplate(image,tp,cv2.TM_CCORR_NORMED)
                _,value,_,loc=cv2.minMaxLoc(score)
                if best is None or value>best[0]:best=(value,sid,float(fine),loc,tp)
        if best is None:return self.last if self.last is not None else (0.,0.,0.,0)
        score,sid,theta,loc,tp=best
        xy=center+(np.asarray(loc)+self.size/2-n/2)*self.res
        model=self.templates[sid]
        edges=model-cv2.erode(model,np.ones((3,3),np.uint8))
        y,x=np.nonzero(edges>.3)
        if len(x)>600:
            inds=np.linspace(0,len(x)-1,600).astype(int);x=x[inds];y=y[inds]
        boundary=np.column_stack((x-self.size/2,y-self.size/2))*self.res
        dist=cv2.distanceTransform((1-image.astype(np.uint8)),cv2.DIST_L2,5)-cv2.distanceTransform(image.astype(np.uint8),cv2.DIST_L2,5)
        def residual(v):
            c,s=np.cos(v[2]),np.sin(v[2]);R=np.array([[c,-s],[s,c]])
            q=boundary@R.T+v[:2]
            pix=(q-center)/self.res+n/2
            values=map_coordinates(dist,[pix[:,1],pix[:,0]],order=1,mode='constant',cval=30.)
            if depth is not None:
                world=np.column_stack((q,np.full(len(q),.04)));cam=(world-np.asarray(T_cam_table)[:3,3])@np.asarray(T_cam_table)[:3,:3]
                z=-cam[:,2];uvx=K[0,0]*cam[:,0]/z+K[0,2]-.5;uvy=K[1,2]-K[1,1]*cam[:,1]/z-.5
                observed=map_coordinates(np.asarray(depth),[uvy,uvx],order=0,mode='constant',cval=np.nan)
                hidden=np.isfinite(observed)&(observed<z-SETTINGS['occlusion_margin'])
                values=values*np.where(hidden,.05,1.)
            return values
        init=np.r_[xy,theta]
        starts=[(init,np.array([.010,.010,.09]))]
        if self.last is not None and int(self.ids[sid])==self.last[3]:
            c,s=np.cos(self.last[2]),np.sin(self.last[2]);R=np.array([[c,-s],[s,c]])
            starts.append((np.r_[np.asarray(self.last[:2])-R@self.offsets[sid],self.last[2]],np.array([.025,.025,.30])))
        chosen=None
        for initial,bounds in starts:
            try:
                fit=least_squares(residual,initial,diff_step=1e-3,max_nfev=SETTINGS['max_nfev'],loss=SETTINGS['loss'],f_scale=SETTINGS['loss_scale'],
                    bounds=(initial-bounds,initial+bounds))
                if self.last is not None and abs(float(wrap(fit.x[2]-self.last[2])))>SETTINGS['max_yaw_innovation']:continue
                cost=float(np.mean(np.minimum(residual(fit.x)**2,16.)))
                if chosen is None or cost<chosen[0]:chosen=(cost,fit.x)
            except (ValueError,FloatingPointError):pass
        if chosen is not None:xy=chosen[1][:2];theta=chosen[1][2]
        c,s=np.cos(theta),np.sin(theta);xy=xy+np.array([[c,-s],[s,c]])@self.offsets[sid]
        self.votes[sid]+=.1+score
        shape=int(self.ids[int(np.argmax(self.votes))]) if self.frame>1 else int(self.ids[sid])
        result=(float(xy[0]),float(xy[1]),float(wrap(theta)),shape)
        self.last=result;self.frame+=1
        return result


def load_frames(data='/training',split='validation',max_frames=24):
    root=Path(data);manifest=json.loads((root/'manifest.json').read_text())
    rows=[x for x in manifest['frames'] if split=='all' or x['split']==split]
    if max_frames and len(rows)>max_frames:
        rows=[rows[i] for i in np.linspace(0,len(rows)-1,max_frames).astype(int)]
    out=[]
    for row in rows:
        with np.load(root/row['file']) as a:
            obs={k:a[k] for k in ('rgb','depth','K','T_cam_table') if k in a}
            obs['t']=float(a['t']);gt=a['gt'].copy()
        out.append((row,obs,gt))
    return out


def diagnose(candidate,split='validation',max_frames=24,data='/training',overrides=None):
    candidate=Path(candidate);path=candidate/'estimator.py' if candidate.is_dir() else candidate
    sys.path.insert(0,str(path.parent))
    for name in ('vision','templates','estimator'):sys.modules.pop(name,None)
    spec=importlib.util.spec_from_file_location('measured_estimator',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if overrides and 'vision' in sys.modules:sys.modules['vision'].SETTINGS.update(overrides)
    estimator=module.make_estimator();rows=[];last_episode=None;start=time.monotonic()
    for metadata,obs,gt in load_frames(data,split,max_frames):
        if metadata['episode']!=last_episode:estimator.reset();last_episode=metadata['episode']
        before=time.monotonic()
        try:
            result=np.asarray(estimator.update(**obs),float)
            valid=result.shape==(4,) and np.isfinite(result).all() and 0<=result[3]<=2 and result[3]==int(result[3])
        except Exception as error:result=np.full(4,np.nan);valid=False
        rows.append({'seed':metadata['seed'],'frame':metadata['frame'],'kind':metadata['kind'],'valid':bool(valid),
            'translation_mm':float(np.linalg.norm(result[:2]-gt[:2])*1000) if valid else None,
            'rotation_deg':float(abs(wrap(result[2]-gt[2]))*180/np.pi) if valid else None,
            'predicted_shape':int(result[3]) if valid else None,'seconds':time.monotonic()-before})
    valid=[r for r in rows if r['valid']]
    return {'frames':len(rows),'valid_fraction':len(valid)/max(1,len(rows)),
        'median_translation_mm':float(np.median([r['translation_mm'] for r in valid])) if valid else None,
        'mean_translation_mm':float(np.mean([r['translation_mm'] for r in valid])) if valid else None,
        'median_rotation_deg':float(np.median([r['rotation_deg'] for r in valid])) if valid else None,
        'mean_rotation_deg':float(np.mean([r['rotation_deg'] for r in valid])) if valid else None,
        'mean_frame_seconds':float(np.mean([r['seconds'] for r in rows])) if rows else None,
        'wall_seconds':time.monotonic()-start,'p90_translation_mm':float(np.percentile([r['translation_mm'] for r in valid],90)) if valid else None,
        'p90_rotation_deg':float(np.percentile([r['rotation_deg'] for r in valid],90)) if valid else None,
        'selection_score':float((len(valid)/max(1,len(rows)))/(1+np.median([r['translation_mm'] for r in valid])+.25*np.median([r['rotation_deg'] for r in valid])+.25*np.percentile([r['translation_mm'] for r in valid],90)+.1*np.percentile([r['rotation_deg'] for r in valid],90))) if valid else 0.,'rows':rows,
        'labels':'Public noisy pose labels; shape IDs are not supplied.'}


def fit_templates(output,data='/training',resolution=.001,size=480,cluster_ids=None):
    from scipy.cluster.vq import kmeans2
    frames=[row for row in load_frames(data,'train',0) if row[0]['kind']=='static']
    images=[]
    for metadata,obs,gt in frames:
        points=plane_points(obs)
        c,s=np.cos(gt[2]),np.sin(gt[2]);rotation=np.array([[c,-s],[s,c]])
        q=(points-gt[:2])@rotation
        im=raster_points(q,np.zeros(2),resolution,size).astype(np.float32)
        images.append(im)
    feat=np.array([cv2.resize(im,(50,50),interpolation=cv2.INTER_AREA).ravel() for im in images])
    _,labels=kmeans2(feat,3,minit='++',seed=11,iter=40)
    templates=[];counts=[];offsets=[]
    for cluster in range(3):
        count=int((labels==cluster).sum());counts.append(count)
        mean=np.mean(np.array(images)[labels==cluster],axis=0)
        image=(mean>.45).astype(np.uint8)
        image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        yy,xx=np.nonzero(image);anchor=(np.array([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2])-size/2)*resolution
        transform=np.array([[1.,0.,150.-size/2-anchor[0]/resolution],[0.,1.,150.-size/2-anchor[1]/resolution]])
        image=cv2.warpAffine(image.astype(np.float32),transform,(300,300),flags=cv2.INTER_LINEAR)
        templates.append(cv2.GaussianBlur(image,(3,3),.6));offsets.append(-anchor)
    inferred=[]
    for template in templates:
        yy,xx=np.nonzero(template>.5);lo,hi=yy.min(),yy.max();profile=template[int(lo+.15*(hi-lo)):int(lo+.40*(hi-lo))].mean(axis=0)>.3
        runs=int(np.sum(profile&~np.r_[False,profile[:-1]]))
        top_x=np.flatnonzero(profile);tail=(xx.max()-top_x.max())/max(1,xx.max()-xx.min())
        inferred.append(0 if runs==1 else 2 if tail>.10 else 1)
    ids=np.asarray(inferred if cluster_ids is None else cluster_ids)
    if sorted(ids.tolist())!=[0,1,2]:raise ValueError('Inspect template clusters and provide distinct semantic cluster_ids')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'templates.npz',templates=np.asarray(templates),ids=ids,offsets=np.asarray(offsets))
    montage=np.concatenate([(np.asarray(t)*255).astype(np.uint8) for t in templates],axis=1)
    cv2.imwrite(str(output/'template-clusters.png'),montage)
    assignments=[{'seed':f[0]['seed'],'cluster':int(label)} for f,label in zip(frames,labels)]
    report={'counts':counts,'assignments':assignments,'cluster_ids':ids.tolist(),'resolution_m':resolution,
        'source':'Learned from public RGB-D rendered observations and noisy pose labels; no shape labels provided.'}
    (output/'template-training.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
