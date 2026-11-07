"""Stroke-specific labeling and interactive plots for Apple Watch TRIAL 2.

Place beside 1003_TRIAL_2_LABELED_DATA.csv; run in VS Code:
    python -m pip install pandas numpy matplotlib scipy
    python trial2_stroke_specific_labeler.py

Controls in detailed figure (turn off Matplotlib zoom/pan before editing):
    Left / Right arrows : previous / next 25-length
    Left click          : correct swimming start
    Right click         : correct swimming end
    S                   : save edited boundaries and labeled CSV
    R                   : restore automatically detected start for this length

Use --no-show to generate files and images without opening plot windows.

CAUTION: detected starts are heuristics and all must be reviewed. Stroke endings
and rest/transition endings are preliminary estimates from Trial 1 and must
also be checked. This script does NOT use the existing labels as ground truth.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / '1003_TRIAL_2_LABELED_DATA.csv'
BOUNDARIES = ROOT / 'trial2_stroke_specific_boundaries.csv'
LABELED = ROOT / '1003_TRIAL_2_STROKE_SPECIFIC_LABELS.csv'
OVERVIEW = ROOT / 'trial2_stroke_specific_overview.png'
CONTACT_SHEET = ROOT / 'trial2_stroke_specific_all_20.png'
LENGTH_SEC, N_LENGTHS, TIME_OFFSET = 30.0, 20, 0.0
STROKES = ('butterfly', 'backstroke', 'breaststroke', 'freestyle')
# The first value is a PROVISIONAL swim end, second the provisional end of rest.
END_TIMES = {'butterfly': (16., 29.), 'backstroke': (16.5, 28.61),
             'breaststroke': (20.4, 28.62), 'freestyle': (15., 28.7)}
COLORS = {'butterfly':'#9467bd','backstroke':'#1f77b4',
          'breaststroke':'#2ca58d','freestyle':'#e58e29',
          'not-swimming':'#8d939e','transition':'#c3a56c'}
GYRO = ['rotationRateX','rotationRateY','rotationRateZ']
ACCEL = ['accelerationX','accelerationY','accelerationZ']


def read_csv():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Put {INPUT.name} next to this script.')
    df = pd.read_csv(INPUT, dtype={'time':'string'}, low_memory=False)
    for col in ['seconds_elapsed',*GYRO,*ACCEL]:
        if col not in df: raise ValueError(f'Missing column: {col}')
        df[col] = pd.to_numeric(df[col].astype('string').str.strip().str.lstrip("'"),errors='coerce')
    if df[['seconds_elapsed',*GYRO,*ACCEL]].isna().any().any():
        raise ValueError('Missing/non-numeric timestamp or sensor values in CSV.')
    df = df.sort_values('seconds_elapsed',kind='stable').reset_index(drop=True)
    return df


def find_start(t, y, slot, stroke):
    """Return start of first stroke rise; stroke-specific first-peak rules.

    Butterfly: first large repeating rise after stillness.
    Backstroke: ACCEPT muted early humps before the high regular peaks.
    Breaststroke: first sharp rise toward a large initial peak.
    Freestyle: start of gradual rising incline toward first large broad peak.
    """
    search = (t >= slot + .35) & (t < slot + 8.0)
    x, z = t[search] - slot, y[search]
    if len(x)<20: return np.nan,'review-no-data'
    dt = np.median(np.diff(x))
    if stroke == 'backstroke':
        height, prom, beginning = 3.3, 1.25, .55
    elif stroke == 'butterfly':
        height, prom, beginning = 6.0, 3.0, 1.4
    elif stroke == 'breaststroke':
        height, prom, beginning = 6.0, 3.0, 1.2
    else:
        height, prom, beginning = 4.8, 2.0, 1.1
    peaks, props = find_peaks(z,height=height,prominence=prom,distance=max(1,int(.6/dt)))
    peaks = [p for p in peaks if x[p] >= beginning]
    if not peaks: return np.nan,'review-no-peak'
    pk = peaks[0]
    # Find a preceding trough. Do not reach back into movement from the previous length.
    backtrack = {'butterfly':2.0,'backstroke':1.5,'breaststroke':1.7,'freestyle':2.0}[stroke]
    left=max(0, pk-int(round(backtrack/dt)))
    trough=left+int(np.argmin(z[left:pk+1]))
    # Freestyle starts at the beginning of the climb toward the first sustained peak.
    # Breaststroke's first spike usually crosses magnitude ~2 very rapidly.
    # Backstroke may have a muted first hump: keep its earlier incline.
    floor=float(z[trough]); top=float(z[pk]); amplitude=top-floor
    if stroke=='breaststroke':
        threshold=max(1.9,floor+min(.6,.12*amplitude))
    elif stroke=='freestyle':
        threshold=max(1.7,floor+min(.55,.14*amplitude))
    elif stroke=='backstroke':
        threshold=floor+min(.45,.16*amplitude)
    else:
        threshold=floor+min(.6,.12*amplitude)
    rise = np.where(z[trough:pk+1]>=threshold)[0]
    if not len(rise): return np.nan,'review-rise'
    index=trough+int(rise[0])
    return float(slot+x[index]),f'auto-{stroke}'


def initial_table(t, smooth):
    rows=[]
    for i in range(N_LENGTHS):
        stroke=STROKES[i%4]
        slot=TIME_OFFSET+i*LENGTH_SEC
        start,source=find_start(t,smooth,slot,stroke)
        end,rest=END_TIMES[stroke]
        rows.append(dict(length=i+1,stroke=stroke,slot_start_s=slot,
                         swim_start_s=round(start,3) if np.isfinite(start) else np.nan,
                         swim_end_s=slot+end,rest_end_s=slot+rest,
                         slot_end_s=slot+LENGTH_SEC,start_source=source,
                         end_source='provisional-template'))
    result=pd.DataFrame(rows)
    # Retain explicitly manual edits from previous run; re-detect all other starts.
    if BOUNDARIES.is_file():
        previous=pd.read_csv(BOUNDARIES)
        for i,row in result.iterrows():
            old=previous.loc[previous.length==row.length]
            if old.empty:continue
            old=old.iloc[0]
            if str(old.get('start_source',''))=='manual' and pd.notna(old.swim_start_s):
                result.loc[i,'swim_start_s']=float(old.swim_start_s)
                result.loc[i,'start_source']='manual'
            if str(old.get('end_source',''))=='manual' and pd.notna(old.swim_end_s):
                result.loc[i,'swim_end_s']=float(old.swim_end_s)
                result.loc[i,'end_source']='manual'
    return result


def make_labels(df, table):
    t=df.seconds_elapsed.to_numpy()
    labels=np.full(len(df),'unknown',dtype=object)
    for row in table.itertuples(index=False):
        a,b,c,d,e=row.slot_start_s,row.swim_start_s,row.swim_end_s,row.rest_end_s,row.slot_end_s
        if not (np.isfinite(b) and a<=b<c<=d<=e):
            print(f'Length {row.length} invalid/review: leaving unknown')
            continue
        labels[(t>=a)&(t<e)]='transition'
        labels[(t>=b)&(t<c)]=row.stroke
        labels[(t>=c)&(t<d)]='not-swimming'
    output=df.copy()
    output['label (all lowercase)']=labels
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-show',action='store_true',help='Save files and plots without GUI')
    args=parser.parse_args()
    df=read_csv(); t=df.seconds_elapsed.to_numpy(dtype=float)
    g=df[GYRO].to_numpy(dtype=float); a=df[ACCEL].to_numpy(dtype=float)
    gm=np.linalg.norm(g,axis=1); am=np.linalg.norm(a,axis=1)
    dt=np.median(np.diff(t)); n=max(3,int(round(.23/dt))|1)
    smooth=gaussian_filter1d(median_filter(gm,size=n),sigma=max(1,.06/dt))
    accel_s=gaussian_filter1d(median_filter(am,size=n),sigma=max(1,.06/dt))
    tab=initial_table(t,smooth)

    def save():
        tab.to_csv(BOUNDARIES,index=False,float_format='%.6f')
        out=make_labels(df,tab)
        out.to_csv(LABELED,index=False)
        print(f'Saved {BOUNDARIES.name} and {LABELED.name}')

    fig0, ax0=plt.subplots(figsize=(15,4.6))
    ax0.plot(t,smooth,color='black',lw=.65)
    for r in tab.itertuples(index=False):
        ax0.axvline(r.slot_start_s,color='gray',alpha=.4,lw=.6)
        if np.isfinite(r.swim_start_s):
            ax0.axvline(r.swim_start_s,color=COLORS[r.stroke],alpha=.8,lw=1.2)
    ax0.set(xlabel='Seconds elapsed',ylabel='Smoothed gyroscope magnitude',
            title='Trial 2: stroke-specific starts for all 20 lengths (provisional)')
    ax0.grid(alpha=.15);fig0.tight_layout();fig0.savefig(OVERVIEW,dpi=160)

    figgrid, grid=plt.subplots(5,4,figsize=(17,12))
    for i,ax in enumerate(grid.flat):
        r=tab.iloc[i];lo=float(r.slot_start_s)
        m=(t>=lo)&(t<lo+9)
        ax.plot(t[m]-lo,gm[m],color='#bbbbbb',alpha=.35,lw=.55)
        ax.plot(t[m]-lo,smooth[m],color='black',lw=1.6)
        ax.plot(t[m]-lo,accel_s[m],color='#d98522',lw=.85)
        if np.isfinite(r.swim_start_s):
            ax.axvline(float(r.swim_start_s)-lo,color='red',ls='--',lw=1.4)
        ax.set_xlim(0,9);ax.set_ylim(bottom=0)
        ax.set_title(f'{i+1:02d} {r.stroke.title()} | {r.swim_start_s:.2f}s',fontsize=10)
        ax.grid(alpha=.18)
    figgrid.suptitle('Trial 2 — 20 individually detected starts (red dashed line)',fontsize=15)
    figgrid.tight_layout();figgrid.savefig(CONTACT_SHEET,dpi=150)

    fig,axes=plt.subplots(3,1,figsize=(14,8.5),sharex=True)
    colors=('tab:blue','tab:orange','tab:green')
    for j,letter in enumerate('XYZ'):
        axes[0].plot(t,g[:,j],color=colors[j],alpha=.7,lw=.55,label=f'Gyro {letter}')
        axes[1].plot(t,a[:,j],color=colors[j],alpha=.7,lw=.55,label=f'Accel {letter}')
    axes[2].plot(t,gm,color='#aaaaaa',alpha=.3,lw=.55,label='Raw gyroscope magnitude')
    axes[2].plot(t,smooth,color='black',lw=1.7,label='Smoothed gyroscope magnitude')
    axes[2].plot(t,accel_s,color='#e28a20',lw=1.2,label='Smoothed accelerometer magnitude')
    axes[0].set_ylabel('Gyroscope axes');axes[1].set_ylabel('Accelerometer axes')
    axes[2].set_ylabel('Magnitude');axes[2].set_xlabel('Elapsed time (seconds)')
    for ax in axes:
        ax.grid(alpha=.17);ax.legend(loc='upper right',fontsize=8)
    current=[0];overlays=[]
    def draw():
        for art in overlays:art.remove()
        overlays.clear()
        row=tab.iloc[current[0]];lo=float(row.slot_start_s);hi=float(row.slot_end_s)
        st=float(row.swim_start_s);en=float(row.swim_end_s);re=float(row.rest_end_s)
        for ax in axes:
            if np.isfinite(st) and lo<=st<en<=re<=hi:
                spans=[(lo,st,'transition'),(st,en,row.stroke),
                       (en,re,'not-swimming'),(re,hi,'transition')]
                for begin,end,label in spans:
                    if end>begin:overlays.append(ax.axvspan(begin,end,color=COLORS[label],alpha=.14))
                overlays.append(ax.axvline(st,color='red',ls='--',lw=1.7))
            overlays.append(ax.axvline(en,color='blue',ls='--',lw=1.25))
            overlays.append(ax.axvline(re,color='gray',ls=':',lw=1.1))
            ax.set_xlim(lo,hi)
        fig.suptitle(f'Trial 2 | Length {int(row.length):02d}/20 | {row.stroke.title()} | '
                     f'Start {st:.3f}s ({row.start_source}) | End {en:.2f}s (provisional)\n'
                     'Left click=start | Right click=end | arrows=navigate | S=save | R=redetect',fontsize=11)
        fig.canvas.draw_idle()
        print(f'Length {row.length:02.0f} {row.stroke}: start {st:.3f}s ({row.start_source})')
    def key(event):
        if event.key in ('right','n'):
            current[0]=min(N_LENGTHS-1,current[0]+1);draw()
        elif event.key in ('left','p'):
            current[0]=max(0,current[0]-1);draw()
        elif event.key=='s':save()
        elif event.key=='r':
            i=current[0];r=tab.iloc[i];start,source=find_start(t,smooth,float(r.slot_start_s),r.stroke)
            tab.loc[i,'swim_start_s']=start;tab.loc[i,'start_source']=source;draw()
    def click(event):
        if event.inaxes not in axes or event.xdata is None or event.button not in (1,3):return
        mgr=plt.get_current_fig_manager()
        if getattr(mgr,'toolbar',None) is not None and getattr(mgr.toolbar,'mode',None):return
        i=current[0];r=tab.iloc[i];x=float(event.xdata)
        if event.button==1:
            if not (r.slot_start_s<=x<r.swim_end_s):return
            tab.loc[i,'swim_start_s']=x;tab.loc[i,'start_source']='manual'
        else:
            if not (np.isfinite(r.swim_start_s) and r.swim_start_s<x<=r.rest_end_s):return
            tab.loc[i,'swim_end_s']=x;tab.loc[i,'end_source']='manual'
        draw()
    fig.canvas.mpl_connect('key_press_event',key)
    fig.canvas.mpl_connect('button_press_event',click)
    draw();fig.tight_layout(rect=[0,0,1,.91])
    save()
    print(f'Overview: {OVERVIEW.name}; all 20 views: {CONTACT_SHEET.name}')
    if not args.no_show:plt.show()
    else:plt.close('all')

if __name__=='__main__':main()
