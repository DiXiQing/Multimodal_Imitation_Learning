import csv,json
from pathlib import Path
import cv2
root=Path(r'C:/MineApp/Code/Multimodal_Imitation_Learning')
folder=root/'Report/Trial_Procedure'
source=root/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261107_151125_531596/grasp_segments/20261107_151140/camera.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
cap=cv2.VideoCapture(str(folder/'30mm_20261107_151140_annotated_cropped_phase_labels.mp4'))
fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4));output=folder/'30mm_20261107_151140_annotated_cropped_phase_labels_mm.mp4'
writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));assert writer.isOpened()
initial=float(rows[0]['index_mcp_object_distance'])
for i,row in enumerate(rows):
 ok,im=cap.read();assert ok
 # Replace only the numeric region inside the existing dark information panel.
 # Repaint the whole rounded panel, including labels, in one uniform color.
 color=(75,69,61)
 cv2.rectangle(im,(28,10),(292,82),color,-1)
 cv2.rectangle(im,(16,22),(304,70),color,-1)
 for center in [(28,22),(292,22),(28,70),(292,70)]:cv2.circle(im,center,12,color,-1,cv2.LINE_AA)
 cv2.putText(im,'Distance',(30,37),0,.50,(245,245,245),1,cv2.LINE_AA)
 cv2.putText(im,'Aperture',(30,64),0,.50,(245,245,245),1,cv2.LINE_AA)
 distance=float(row['index_mcp_object_distance']);aperture=float(row['finger_width'])
 cv2.putText(im,f'{distance*1.2:.1f} mm ({distance/initial*100:.0f}%)',(141,37),0,.48,(60,225,255),2,cv2.LINE_AA)
 cv2.putText(im,f'{aperture*1.2:.1f} mm',(141,64),0,.48,(190,120,235),2,cv2.LINE_AA)
 writer.write(im)
 if i==0:cv2.imwrite(str(folder/'30mm_20261107_151140_mm_preview.png'),im)
writer.release();cap.release()
(folder/'30mm_20261107_151140_mm_conversion.json').write_text(json.dumps(dict(scale_mm_per_px=1.2,source_csv=str(source),frames=len(rows),distance='index_mcp_object_distance * 1.2',aperture='finger_width * 1.2',original_unchanged=True),indent=2))
print(output)
