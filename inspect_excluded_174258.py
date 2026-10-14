exec(open('crop_recording_174258.py',encoding='utf-8-sig').read().split('records=[]')[0])
fig,axs=plt.subplots(5,1,figsize=(15,12))
for ax,p in zip(axs,[1330,1844,4076,7282,10705]):
 a=p-30;b=p+65;ax.plot(c.frame.iloc[a:b],y[a:b]);ax.grid();ax.set_title(str(p));ax2=ax.twinx();ax2.plot(c.frame.iloc[a:b],oa[a:b],color='orange')
fig.tight_layout();fig.savefig(out/'excluded_detail.png',dpi=120)
