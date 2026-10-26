"""Synthetic reporting/inference check; no training or optimizer steps."""
import sys, tempfile, json, csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'New/Program'))
import train_mlp_lstm_comparison as m
import evaluate_distance_window_mlp as evaluator
torch.set_num_threads(2)
with tempfile.TemporaryDirectory(prefix='three_model_check_',dir=ROOT/'tool') as td:
    root=Path(td).resolve()
    assert root.is_relative_to((ROOT/'tool').resolve())
    for part,names in [('Train',['grasp_001','grasp_002','grasp_003']),('Validation',['grasp_004','grasp_005','grasp_006'])]:
        for j,name in enumerate(names):
            folder=root/part/name;folder.mkdir(parents=True)
            (folder/'settings.json').write_text(json.dumps(dict(object_color='black',size_cm=4,width=540,height=480,approach_guide=dict(target_movement_s=j+1))))
            (folder/'segment.json').write_text(json.dumps(dict(onset=8,stable=25)))
            rows=[dict(frame=i,t_camera=100+j*10+i/30+(100 if part=='Validation' else 0),finger_width=10+45*np.sin(i/34*np.pi)+j,
                       hand_object_signed_distance=250-i*8-j,hand_object_distance=abs(250-i*8-j),valid=1) for i in range(35)]
            with (folder/'camera.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def no_training(name,inputs,fit,val,args,out,device,log):
        model=(m.ContinuousDistanceLSTM(args.hidden) if name==m.LSTM_MODEL else m.DistanceMLP(inputs,args.hidden)).to(device)
        scale=m.fit_sequence_scaler(fit) if name==m.LSTM_MODEL else m.fit_scaler(fit)
        torch.save(m.checkpoint(model,name,inputs,scale,args,0,fit),out/'models'/f'{name}.pth')
        history=[dict(model=name,stage='check_only',epoch=0,train_loss=0.,train_mae_px=0.,validation_mae_px=0.)]
        return model,scale,history,dict(model=name,selected_epoch=0,best_validation_mae_px=0.)
    m.train_one=no_training
    sys.argv=['check','--source',str(root),'--output',str(root/'out'),'--device','cpu']
    m.main()
    out=root/'out'
    for filename in ('validation_model_comparison.png','validation_metrics_chart.png','validation_phase_errors.png','validation_metrics_by_condition.png','overall_model_summary.csv'):
        assert (out/filename).is_file(),filename
    with (out/'overall_model_summary.csv').open(encoding='utf-8-sig') as f:
        assert len(list(csv.DictReader(f)))==3
    saved=torch.load(out/'models'/f'{m.LSTM_MODEL}.pth',weights_only=True)
    assert saved['schema']=='distance_lstm_continuous_v1'
    net=m.ContinuousDistanceLSTM(saved['hidden_size']);net.load_state_dict(saved['state_dict']);net.eval()
    x=torch.randn(3,12,1)
    with torch.no_grad():
        whole,_=net(x)
        assert torch.allclose(whole,torch.cat([net(x[i:i+1])[0] for i in range(3)]),atol=1e-6)
        state=None;stream=[]
        for i in range(x.shape[1]):
            value,state=net(x[:,i:i+1],state);stream.append(value)
        assert torch.allclose(whole,torch.cat(stream,dim=1),atol=1e-6)
        altered=x.clone();altered[:,8:]+=100
        assert torch.allclose(whole[:,:8],net(altered)[0][:,:8],atol=1e-6), 'Future leakage'
    assert len(list((out/'validation_lstm_full_predictions').glob('*.csv')))==3
    for name in m.MODEL_SPECS:
        sys.argv=['evaluate','--model',str(out/'models'/f'{name}.pth'),'--source',str(root/'Validation'),'--output',str(root/('test_'+name))]
        evaluator.main()
        result=json.loads((root/('test_'+name)/'evaluation_info.json').read_text(encoding='utf-8'))
        assert result['overall']['frames']==93, 'All three models must use identical eligible targets'
print('PASS: reports, continuous checkpoint loading, full-sequence/streaming equivalence, trial isolation and causality. No training. Temporary outputs removed.')
