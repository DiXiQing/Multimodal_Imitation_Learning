exec(open('inspect_recording_195420.py').read().split('y=c.finger_width')[0])
r=pd.read_csv(out/'crop_ranges.csv')
y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
a=c.object_area.rolling(3,center=True,min_periods=1).median().to_numpy()
print('zero FW',c[c.finger_width<=0].frame.tolist())
for _,q in r[r.status=='excluded'].iterrows():
 k=int(q.peak)
 wins=[(round(np.ptp(y[j:j+16]),1),j,round(np.median(y[j:j+16])/y[k],2),round(np.ptp(a[j:j+16])/max(np.median(a[j:j+16]),1),2)) for j in range(k+8,k+30) if np.median(a[j:j+16])>1.35*np.median(a[int(q.start):int(q.onset)]) and np.median(y[j:j+16])>.3*y[k]]
 print(k,sorted(wins)[:1])
frames=list(range(3050,3600,50))+list(range(11550,12100,50))
fig,axs=plt.subplots(4,6,figsize=(18,10))
for ax,f in zip(axs.flat,frames):
 im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{f:06d}.jpg',np.uint8),1)
 ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(str(f));ax.axis('off')
fig.tight_layout();fig.savefig(out/'size_transitions.jpg',dpi=130)
