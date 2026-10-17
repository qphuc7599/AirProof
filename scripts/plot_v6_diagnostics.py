import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('reports/v6');plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
s=json.loads((root/'release_historical_reanalysis/summary.json').read_text());ys=np.arange(5)
means=np.array([r['P5_minus_P0_mean'] for r in s.values()])*1000
bounds=np.array([r['P5_minus_P0_descriptive_95ci'] for r in s.values()])*1000
axes[0].errorbar(means,ys,xerr=np.array([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color='#205577',capsize=3)
axes[0].axvline(0,color='gray',ls='--',lw=1);axes[0].set_yticks(ys,[c.replace('_',' ') for c in s]);axes[0].invert_yaxis();axes[0].set_xlabel('P5 minus P0 RMSE (x 1,000)');axes[0].set_title('(a) Exposed historical release data')
t=json.loads((root/'transport_fullscale_fixture/summary.json').read_text())['outcomes']
axes[1].barh(np.arange(5),[r['metrics']['raw_timely_delivered']/r['metrics']['raw_generated'] for r in t],color='#4a8097')
axes[1].set_yticks(np.arange(5),['v5 deadline','deadline 6','combined','Spray','capped epidemic']);axes[1].invert_yaxis();axes[1].set_xlim(0,1);axes[1].set_xlabel('Fraction arriving within lag 6');axes[1].set_title('(b) Engineering transport fixture')
r=json.loads((root/'numerical_fullscale_fixture/6006001_severe_clean/transport.json').read_text())
counts=[r['receipt_returned'],r['receipt_buffer_dropped'],r['receipt_expired']+r['receipt_pending']]
axes[2].bar(np.arange(3),np.array(counts)/r['receipt_issued'],color=['#387d65','#bd6458','#84919b'])
axes[2].set_xticks(np.arange(3),['Returned','Queue rejected','Expired/pending'],rotation=25,ha='right');axes[2].set_ylim(0,1);axes[2].set_ylabel('Fraction of issued receipts');axes[2].set_title('(c) Finite return bottleneck')
out=root/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'diagnostics.pdf');fig.savefig(out/'diagnostics.png',dpi=180)
