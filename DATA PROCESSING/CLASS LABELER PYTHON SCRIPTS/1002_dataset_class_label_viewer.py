"""1002 Apple Watch: 25-meter repetition labeling workflow.

Run beside 1002_LABELED_DATA.csv:
    python -m pip install numpy pandas scipy matplotlib
    python label_1002_25m_methodology.py
For batch export (no interactive window):
    python label_1002_25m_methodology.py --no-show

This is an editable, heuristic annotation aid, NOT verified ground truth.
All 20 bouts use independent stroke-start detection. Stroke ends are estimated
using the last meaningful accelerometer peak, flattening, and 3-second check.
Transitions may straddle the approximate schedule anchors.

Interactive controls (disable Matplotlib zoom/pan before clicking):
  Left / Right arrows   previous / next 25 m rep
  Left click            set swimming start (green)
  Right click           set swimming end / rest start (blue)
  T + click             set PRE-swim transition start (red)
  S                    save boundaries and labeled CSV
  R                    restore this repetition's automatic boundaries

Edits persist in 1002_25M_REVIEWED_BOUNDARIES.csv. Original recording is untouched.
Reps 13,15,16,20 use approximate, screenshot-informed corrections. Review them.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter, binary_closing
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
INPUT = HERE / '1002_LABELED_DATA.csv'
BOUNDARIES = HERE / '1002_25M_REVIEWED_BOUNDARIES.csv'
OUTPUT = HERE / '1002_25M_REVIEWED_LABELS.csv'
OVERVIEW = HERE / '1002_25M_REVIEWED_OVERVIEW.png'
CONTACT = HERE / '1002_25M_REVIEWED_ALL_20.png'
GYRO = ['rotationRateX','rotationRateY','rotationRateZ']
ACCEL = ['accelerationX','accelerationY','accelerationZ']
STROKES = ('butterfly','backstroke','breaststroke','freestyle')
COLORS = {'butterfly':'#9a72bc','backstroke':'#4c9bd5',
          'breaststroke':'#3aaf8c','freestyle':'#e5a55e',
          'transition':'#d9b46a','not_swimming':'#c2c6ce','unknown':'#ffffff'}
# These are approximate activity locations inferred from 1002 plots.
# Not hard start/end labels. Earlier notes say 8 x 25 on ~45s, then 12 on ~40s.
# One-minute note is ambiguous; no invented rest inserted.
ANCHORS = np.array([0.,42.,85.,127.,173.,216.,259.,296.,336.,376.,
                    416.,456.,496.,536.,576.,616.,656.,696.,736.,776.])



# Approximate corrections from the user's annotated 1002 screenshots.
# 13 and 15: include the initial movement burst in transition, not not_swimming.
# 16 and 20: initial burst belongs to transition, and swimming begins on
# the next continuing sequence of stroke cycles, after a meaningful dip.
# These times are provisional (~0.1-0.3 s visually), and remain editable.
SCREENSHOT_CORRECTIONS = {
    13: dict(prep_start_s=498.92, swim_start_s=503.36),
    15: dict(prep_start_s=577.46, swim_start_s=582.04),
    16: dict(prep_start_s=617.56, swim_start_s=620.75),
    20: dict(prep_start_s=777.44, swim_start_s=780.26),
}


def load_data():
    df=pd.read_csv(INPUT,dtype={'time':'string'},low_memory=False)
    for col in ['seconds_elapsed',*GYRO,*ACCEL]:
        if col not in df: raise ValueError(f'Missing required column: {col}')
        df[col]=pd.to_numeric(df[col].astype('string').str.strip().str.lstrip("'"),errors='coerce')
    if df[['seconds_elapsed',*GYRO,*ACCEL]].isna().any().any():
        raise ValueError('Non-numeric or missing time/sensor measurements')
    return df.sort_values('seconds_elapsed',kind='stable').reset_index(drop=True)


def consecutive_regions(mask):
    m=np.asarray(mask,bool)
    edges=np.diff(np.r_[False,m,False].astype(int))
    return list(zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)))


def estimate_start(t, g, a, anchor, upper, stroke, dt):
    """Select earliest CONTINUOUS stroke bout; reject bursts followed by long quiet gaps.

    Gyroscope thresholds here are data-tuned heuristic candidates; consistent
    activity duration and multiple peaks are required for every stroke.
    """
    use=(t>=anchor-.7)&(t<min(anchor+24.,upper-3.))
    ids=np.flatnonzero(use)
    if len(ids)<10:return np.nan,'review-no-data'
    x=t[ids];y=g[ids]
    # Threshold intentionally modest to include muted early cycles.
    threshold={'butterfly':1.75,'backstroke':1.20,
               'breaststroke':1.35,'freestyle':1.25}[stroke]
    active=y>=threshold
    # Bridge only ordinary inter-stroke troughs, not prolonged quiet gaps.
    gap_s={'butterfly':0.85,'backstroke':1.12,
           'breaststroke':1.12,'freestyle':1.0}[stroke]
    active=binary_closing(active,structure=np.ones(max(3,int(round(gap_s/dt)))),border_value=0)
    candidates=[]
    for lo,hi in consecutive_regions(active):
        duration=x[hi-1]-x[lo]
        if duration<7.0:continue
        yy=y[lo:hi]
        # Require repeated distinct stroke peaks, not prolonged single high spike.
        prom={'butterfly':1.5,'backstroke':.75,'breaststroke':1.,'freestyle':.85}[stroke]
        pk,_=find_peaks(yy,prominence=prom,distance=max(1,int(.55/dt)))
        if len(pk)<4:continue
        if np.percentile(yy,75)<{'butterfly':3.2,'backstroke':2.2,
                                 'breaststroke':2.7,'freestyle':2.4}[stroke]:continue
        candidates.append((lo,hi,pk))
    if not candidates:return np.nan,'review-no-sustained-pattern'
    lo,hi,pk=max(candidates,key=lambda c:x[c[1]-1]-x[c[0]])
    # Backtrack from first confirmed bout to beginning of its initial rise.
    # Restrict to local ~0.65 s to prevent swallowing earlier false burst.
    first=lo+int(pk[0]); left=max(0,lo-int(.65/dt))
    local=y[left:first+1]
    trough=left+int(np.argmin(local)) if len(local) else lo
    # If rise extends too far back, clamp to immediately preceding baseline.
    onset=max(left,min(lo,trough))
    # A long bout can still include preparatory movement before a meaningful
    # low-activity dip. Consider splitting it when accelerometer activity
    # grows substantially after the gap (early bursts in 1002 examples).
    aa=a[ids]
    head=onset
    duration_s=min(7.,float(x[hi-1]-x[head]-3.0))
    if duration_s>=3.5:
        early=(x>=x[head]+.5)&(x<x[head]+duration_s*.7)
        later=(x>=x[head]+duration_s+1.0)&(x<x[head]+duration_s+5.0)
        if early.sum() and later.sum():
            early_level=float(np.median(aa[early]))
            later_level=float(np.median(aa[later]))
            if later_level>max(.35,1.6*early_level):
                # Find a gyro trough near the activity change, so the start
                # is at the beginning of the new climb rather than the peak.
                window=(x>=x[head]+2)&(x<x[head]+duration_s+1.5)
                ww=np.flatnonzero(window)
                if len(ww):
                    floor=ww[int(np.argmin(y[ww]))]
                    if x[floor]>x[head]+2.0 and x[floor]<x[head]+11:
                        # Only move if later activity has multiple peaks.
                        later_peaks,_=find_peaks(y[floor:hi],prominence=prom,
                            distance=max(1,int(.55/dt)))
                        if len(later_peaks)>=3:
                            onset=floor
    return float(x[onset]),f'auto-sustained-{stroke}' 


def estimate_end(t, acc, gyro, start, upper, dt):
    """Accelerometer 50%-of-LAST-peak rule and 3 s confirmation.

    A candidate quiet point is a flattening after the last meaningful stroke
    peak. It is only accepted when the subsequent three seconds remain below
    half that peak. Return the flattening time, not +3 s.
    """
    if not np.isfinite(start):return np.nan,'review-no-start'
    use=(t>=start+5)&(t<upper-1)
    ix=np.flatnonzero(use)
    if len(ix)<max(20,int(6/dt)):return np.nan,'review-short'
    y=acc[ix];x=t[ix]
    # Relative to local quiet baseline, to avoid arbitrary fixed magnitude.
    base=float(np.percentile(y,12))
    height=max(base+.08,float(np.percentile(y,82))*.36)
    prom=max(.07,float(np.percentile(y,85)-base)*.22)
    peaks,_=find_peaks(y,height=height,prominence=prom,distance=max(1,int(.55/dt)))
    if len(peaks)<2:return np.nan,'review-weak-accel-peaks'
    # Look for declining accel activity late enough that it cannot be an early
    # normal inter-stroke trough. Peak recovery comparison uses *last* peak.
    span3=max(1,int(round(3/dt)))
    lowrun=max(2,int(round(.35/dt)))
    for k in range(1,len(peaks)):
        p=peaks[k]
        # Need at least a few demonstrated stroke peaks before considering end.
        if k<2 or x[p]<start+7:continue
        last=float(y[p]);previous=p
        lim=base+.5*max(0,last-base)
        search_end=min(len(y)-span3-1,p+int(3.2/dt))
        if search_end<=p+lowrun:continue
        for j in range(p+1,search_end-lowrun):
            if y[j]>lim or np.max(y[j:j+lowrun])>lim:continue
            # No rebound to half the last peak in the next 3 seconds.
            if np.max(y[j:j+span3])>=lim:continue
            # Require meaningful decline and low flat accel, not just a trough.
            if np.median(y[j:j+lowrun])>base+.24*max(last-base,.01):continue
            if x[j]<start+8:continue
            # Earliest valid flattening after last meaningful peak.
            return float(x[j]),'auto-accel-last-peak-3s'
    # Incomplete evidence: do not silently make an overconfident end boundary.
    return np.nan,'review-end-unconfirmed'


def detect_prep(t,g,a,anchor,start,prev_end,dt):
    """Find connected preparatory activity around a swim start across slots.

    Short isolated wrist variation during sustained inactivity does not by
    itself establish transition. Requires nearby elevated movement preceding
    the confirmed stroke, and limits backward extension.
    """
    lower=max(float(t[0]),prev_end,anchor-6.0)
    if not np.isfinite(start):return np.nan
    if start<=lower:return float(start)
    ids=np.flatnonzero((t>=lower)&(t<start))
    if not len(ids):return float(start)
    x=t[ids];z=g[ids];u=a[ids]
    # Short-term combined activity, with small quiet dip allowed.
    act=(z>max(1.1,float(np.percentile(z,15))+.9)) | (u>max(.35,float(np.percentile(u,15))+.27))
    # Near-start activity may include two preparation stages.
    regions=consecutive_regions(binary_closing(act,structure=np.ones(max(3,int(.60/dt))),border_value=0))
    candidates=[(lo,hi) for lo,hi in regions if x[hi-1] >=start-4.0 and hi-lo>=max(2,int(.14/dt))]
    if not candidates:return float(max(lower,start-.75))
    # Avoid labeling unrelated early motion as preparation if separated by
    # >1.5 seconds of stillness from stroke onset.
    last=candidates[-1]
    if start-x[last[1]-1]>1.6:return float(max(lower,start-.75))
    chosen=last[0]
    for lo,hi in candidates[-2::-1]:
        if x[chosen]-x[hi-1]<=1.25:chosen=lo
        else:break
    return float(x[chosen])


def validate(table):
    for i,r in table.iterrows():
        values=[r.prep_start_s,r.swim_start_s,r.swim_end_s]
        if np.isfinite(values).all():
            if not (r.slot_start_s-6<=values[0]<=values[1]<values[2]<=r.slot_end_s+3):
                print(f'WARNING: length {i+1} has out-of-order boundaries.')


def generate_table(t,g,a,dt):
    rows=[]
    endslots=np.r_[ANCHORS[1:],min(float(t[-1]),816.)]
    previous_end=float(t[0])
    for i,(anchor,upper) in enumerate(zip(ANCHORS,endslots)):
        stroke=STROKES[i%4]
        start,sstatus=estimate_start(t,g,a,anchor,upper,stroke,dt)
        end,estatus=estimate_end(t,a,g,start,upper,dt)
        if np.isfinite(start) and not np.isfinite(end):
            # Prior 1002 boundary is only a fallback PROVISIONAL estimate.
            old_ends=[17.49,78.19,111.09,155.66,212.5,239.03,281.99,322.53,
                      356.36,401.36,444.53,479.04,517.10,560.70,602.30,
                      639.71,676.21,723.79,761.86,799.46]
            end=float(min(upper-.5,max(start+.25,old_ends[i])))
            estatus='provisional-previous-estimate'
        prep=detect_prep(t,g,a,anchor,start,previous_end,dt)
        if np.isfinite(end):previous_end=end
        rows.append(dict(length=i+1,stroke=stroke,slot_start_s=anchor,
                         prep_start_s=prep,swim_start_s=start,swim_end_s=end,
                         slot_end_s=upper,prep_source='auto-context',
                         start_source=sstatus,end_source=estatus))
    df=pd.DataFrame(rows)
    for rep, corrections in SCREENSHOT_CORRECTIONS.items():
        i=rep-1
        for field, value in corrections.items():
            df.loc[i,field]=float(value)
        df.loc[i,'prep_source']='provisional-user-screenshot'
        df.loc[i,'start_source']='provisional-user-screenshot'
    # Retain any later hand-corrected boundaries in the output boundary CSV.
    if BOUNDARIES.exists():
        old=pd.read_csv(BOUNDARIES)
        for i,row in df.iterrows():
            m=old.loc[old.length==row.length]
            if m.empty:continue
            prior=m.iloc[0]
            for name,source in [('prep_start_s','prep_source'),('swim_start_s','start_source'),('swim_end_s','end_source')]:
                if str(prior.get(source,''))=='manual' and pd.notna(prior.get(name)):
                    df.loc[i,name]=float(prior[name]);df.loc[i,source]='manual'
    validate(df)
    return df


def assign_labels(df,table):
    t=df.seconds_elapsed.to_numpy(float)
    # Start by labeling the entire recording not_swimming; leave post-workout
    # tail unknown because the notes do not confirm post-session activity.
    labels=np.full(len(df),'not_swimming',dtype=object)
    labels[t>=float(table.slot_end_s.iloc[-1])]='unknown'
    # Each candidate stroke + preceding transition is independently placed.
    # Assign using chronological intervals. No fixed schedule assignment of
    # transition is used; it can start before the nominal rep anchor.
    for r in table.itertuples(index=False):
        pre,st,en=r.prep_start_s,r.swim_start_s,r.swim_end_s
        if not (np.isfinite(pre) and np.isfinite(st) and np.isfinite(en) and pre<=st<en):
            print(f'Length {r.length} needs review: no swimming label assigned')
            continue
        labels[(t>=pre)&(t<st)]='transition'
        labels[(t>=st)&(t<en)]=r.stroke
    out=df.copy();out['label (all lowercase)']=labels
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--no-show',action='store_true')
    opts=p.parse_args()
    df=load_data();t=df.seconds_elapsed.to_numpy(float)
    gxyz=df[GYRO].to_numpy(float);axyz=df[ACCEL].to_numpy(float)
    dt=float(np.median(np.diff(t)))
    gm=np.linalg.norm(gxyz,axis=1);am=np.linalg.norm(axyz,axis=1)
    width=max(3,int(round(.18/dt))|1)
    gs=gaussian_filter1d(median_filter(gm,size=width),sigma=max(1,.075/dt))
    aa=gaussian_filter1d(median_filter(am,size=width),sigma=max(1,.075/dt))
    table=generate_table(t,gs,aa,dt)
    original_auto=table.copy(deep=True)

    def export():
        validate(table)
        table.to_csv(BOUNDARIES,index=False,float_format='%.6f')
        labeled=assign_labels(df,table)
        labeled.to_csv(OUTPUT,index=False)
        print(f'Exported: {OUTPUT.name}, {BOUNDARIES.name}')
        print(labeled['label (all lowercase)'].value_counts().to_string())

    def spans(ax,r):
        a,b,c=r.prep_start_s,r.swim_start_s,r.swim_end_s
        if np.isfinite(a) and np.isfinite(b) and a<b:
            ax.axvspan(a,b,color=COLORS['transition'],alpha=.24)
            ax.axvline(a,color='#d32631',lw=1.5)
        if np.isfinite(b) and np.isfinite(c) and b<c:
            ax.axvspan(b,c,color=COLORS[r.stroke],alpha=.20)
            ax.axvline(b,color='#178b47',lw=1.7)
            ax.axvline(c,color='#2853d7',lw=1.7)

    fig0,ax0=plt.subplots(figsize=(15,5))
    ax0.plot(t,gs,color='black',lw=.65,label='Smoothed gyroscope magnitude')
    ax0.plot(t,aa,color='#e78720',lw=.6,label='Smoothed accelerometer magnitude')
    for r in table.itertuples(index=False):spans(ax0,r)
    ax0.set(title='1002 | 25 m methodology | All 20 reps (candidate labels)',
            xlabel='Elapsed seconds',ylabel='Magnitude',xlim=(0,float(t[-1])))
    ax0.grid(alpha=.16);ax0.legend(loc='upper right');fig0.tight_layout();fig0.savefig(OVERVIEW,dpi=140)

    figgrid,grid=plt.subplots(5,4,figsize=(19,14),sharey=False)
    for i,ax in enumerate(grid.flat):
        r=table.iloc[i];lo=max(float(t[0]),float(r.slot_start_s)-4);hi=float(r.slot_end_s)+2
        m=(t>=lo)&(t<=hi)
        ax.plot(t[m],gs[m],color='black',lw=1)
        ax.plot(t[m],aa[m],color='#e78720',lw=.8)
        spans(ax,r)
        ax.set_xlim(lo,hi);ax.set_ylim(bottom=0)
        ax.set_title(f'{i+1:02d} {r.stroke} | {r.swim_start_s:.2f}–{r.swim_end_s:.2f}s',fontsize=10)
        ax.grid(alpha=.14)
    figgrid.suptitle('1002 | reviewed: red=transition, green=swimming, blue=rest | 13,15,16,20 corrected',fontsize=15)
    figgrid.tight_layout(rect=[0,0,1,.97]);figgrid.savefig(CONTACT,dpi=145)

    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
    axiscolors=['tab:blue','tab:orange','tab:green']
    for j,letter in enumerate('XYZ'):
        axes[0].plot(t,gxyz[:,j],color=axiscolors[j],lw=.65,label='Gyro '+letter)
        axes[1].plot(t,axyz[:,j],color=axiscolors[j],lw=.65,label='Accel '+letter)
    axes[2].plot(t,gm,color='gray',alpha=.3,lw=.5,label='Raw gyro magnitude')
    axes[2].plot(t,gs,color='black',lw=1.8,label='Smoothed gyro magnitude')
    axes[2].plot(t,aa,color='#e78720',lw=1.4,label='Smoothed accel magnitude')
    for ax in axes:ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=8)
    axes[2].set_xlabel('Elapsed seconds');axes[2].set_ylabel('Magnitude')
    axes[0].set_ylabel('Gyroscope');axes[1].set_ylabel('Accelerometer')
    state={'index':0,'mode':'swim_start'}
    overlays=[]
    def draw():
        for obj in overlays:obj.remove()
        overlays.clear()
        r=table.iloc[state['index']]
        for ax in axes:
            a,b,c=r.prep_start_s,r.swim_start_s,r.swim_end_s
            if np.isfinite(a) and np.isfinite(b) and a<b:
                overlays.append(ax.axvspan(a,b,color=COLORS['transition'],alpha=.25))
                overlays.append(ax.axvline(a,color='#d32631',lw=2))
            if np.isfinite(b) and np.isfinite(c) and b<c:
                overlays.append(ax.axvspan(b,c,color=COLORS[r.stroke],alpha=.20))
                overlays.append(ax.axvline(b,color='#178b47',lw=2))
                overlays.append(ax.axvline(c,color='#2853d7',lw=2))
            ax.set_xlim(max(float(t[0]),float(r.slot_start_s)-6.),float(r.slot_end_s)+3.)
        fig.suptitle(f'1002 | rep {int(r.length):02d}/20 | {r.stroke} | '
                     f'prep {r.prep_start_s:.2f} | swim {r.swim_start_s:.2f}–{r.swim_end_s:.2f}\n'
                     'Left click=stroke start | Right click=end/rest | T+click=transition start | '
                     'arrows=navigate | S=save | R=reset',fontsize=11)
        fig.canvas.draw_idle()
    def on_key(e):
        if e.key in ('right','n'):
            state['index']=min(19,state['index']+1);draw()
        elif e.key in ('left','p'):
            state['index']=max(0,state['index']-1);draw()
        elif e.key=='s':export()
        elif e.key=='r':
            i=state['index']
            for col in ('prep_start_s','swim_start_s','swim_end_s','prep_source','start_source','end_source'):
                table.loc[i,col]=original_auto.loc[i,col]
            draw()
    def on_click(e):
        if e.inaxes not in axes or e.xdata is None:return
        toolbar=getattr(fig.canvas.manager,'toolbar',None)
        if toolbar is not None and getattr(toolbar,'mode',''):return
        i=state['index'];r=table.iloc[i];x=float(e.xdata)
        if e.key=='t' and e.button==1:
            if x>r.swim_start_s:return
            table.loc[i,'prep_start_s']=x;table.loc[i,'prep_source']='manual'
        elif e.button==1:
            if not (r.prep_start_s<=x<r.swim_end_s):return
            table.loc[i,'swim_start_s']=x;table.loc[i,'start_source']='manual'
        elif e.button==3:
            if not (r.swim_start_s<x<r.slot_end_s+3):return
            table.loc[i,'swim_end_s']=x;table.loc[i,'end_source']='manual'
        else:return
        draw()
    fig.canvas.mpl_connect('key_press_event',on_key)
    fig.canvas.mpl_connect('button_press_event',on_click)
    draw();fig.tight_layout(rect=[0,0,1,.91])
    export()
    if opts.no_show:plt.close('all')
    else:plt.show()

if __name__=='__main__':main()
