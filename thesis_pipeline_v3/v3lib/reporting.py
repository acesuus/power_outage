from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1] / '.mpl_cache'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score,precision_recall_curve
from sklearn.calibration import calibration_curve
from lib.reporting import plt,save_figure,top_k_scores


def bootstrap(frame,predictions,main,alternate,config):
    calendar = pd.date_range(frame.date.min(),frame.date.max(),freq='D').strftime('%Y-%m-%d').to_numpy()
    groups = {d:np.flatnonzero(frame.date.to_numpy() == d) for d in calendar}
    width,rng = config['bootstrap_block_days'],np.random.default_rng(config['seed'])
    records = []
    for repeat in range(config['bootstrap_replicates']):
        starts = rng.integers(0,len(calendar)-width+1,size=int(np.ceil(len(calendar)/width)))
        days = np.concatenate([calendar[s:s+width] for s in starts])[:len(calendar)]
        take = np.concatenate([groups[d] for d in days])
        y = frame.outage_label.to_numpy()[take]
        scores = {}
        for label,p in predictions.items():
            probability = np.clip(p[take],1e-7,1-1e-7)
            score = {'average_precision':float(average_precision_score(y,probability)),
                'brier_score':float(brier_score_loss(y,probability)),
                'log_loss':float(log_loss(y,probability,labels=[0,1])),
                'roc_auc':float(roc_auc_score(y,probability)) if len(np.unique(y)) == 2 else np.nan}
            scores[label] = score
            records.append({'repeat':repeat,'model':label,**score})
        for label in ['v2_xgboost',alternate]:
            records.append({'repeat':repeat,'model':'main_minus_'+label,
                **{metric:scores[main][metric]-scores[label][metric] for metric in scores[main]}})
    draws,rows = pd.DataFrame(records),[]
    for label,g in draws.groupby('model'):
        for metric in ['average_precision','brier_score','log_loss','roc_auc']:
            values = g[metric].dropna()
            low,high = np.quantile(values,[0.025,0.975])
            rows.append({'model':label,'metric':metric,'lower_95':low,'upper_95':high,
                'valid_replicates':len(values),'method':'paired moving seven-calendar-day bootstrap; exploratory'})
    return pd.DataFrame(rows)


def figures(frame,predictions,results,folder):
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax = plt.subplots(figsize=(9,5))
    for label,p in predictions.items():
        precision,recall,_ = precision_recall_curve(frame.outage_label,p)
        ax.step(recall,precision,where='post',label=label)
    ax.axhline(frame.outage_label.mean(),linestyle='--',color='gray',label='Positive fraction')
    ax.set(xlabel='Recall',ylabel='Precision',title='Already-seen 2026: exploratory precision–recall',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8)
    save_figure(fig,folder / 'precision_recall.png')
    fig,ax = plt.subplots(figsize=(8,5))
    for label,p in predictions.items():
        observed,predicted = calibration_curve(frame.outage_label,p,n_bins=10,strategy='quantile')
        ax.plot(predicted,observed,marker='o',label=label)
    ax.plot([0,0.3],[0,0.3],color='gray',linestyle='--')
    ax.set(xlabel='Mean predicted probability',ylabel='Recorded outage fraction',
        title='Already-seen 2026: probability reliability',xlim=(0,0.3),ylim=(0,0.3))
    ax.legend(fontsize=8)
    save_figure(fig,folder / 'reliability.png')
    labels = list(results)
    fig,axes = plt.subplots(1,3,figsize=(15,6))
    for ax,metric,title in zip(axes,['average_precision','brier_score','log_loss'],
            ['AP: higher is better','Brier: lower is better','Log loss: lower is better']):
        ax.bar(labels,[results[n][metric] for n in labels],color='#28576c')
        ax.tick_params(axis='x',labelrotation=65,labelsize=8)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    save_figure(fig,folder / 'model_comparison.png')
