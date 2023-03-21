from ultralytics import YOLO

model = YOLO('./weights/yv8s_vehicle_det.pt')

results = model('C:\\Users\\lucas\\OneDrive - ifsp.edu.br\\Projeto OCR\\Bases_Dados\\UFPR-ALPR dataset\\images\\val')
print(results)