from sklearn.linear_model import RANSACRegressor
import torch
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2   
import numpy as np



image_path = "C:\\Ekain\\CTS20DA02\\S080\\20EKIB03_CTS20DA02_S080Z07.jpg"  # ruta de la imagen de prueba

def ransac_line_from_mask(mask):
    # Extract edge points
        mask = (mask > 0).astype(np.uint8) * 255
        y_coords, x_coords = np.where(mask > 0)

        x1 = x_coords.reshape(-1, 1)
        y1 = y_coords.reshape(-1, 1)

    # Fit RANSAC model
        ransac = RANSACRegressor(min_samples=50, residual_threshold=2.0)
        ransac.fit(x1, y1)

    # Get line slope and intercept
        slope = ransac.estimator_.coef_[0][0]
        intercept = ransac.estimator_.intercept_[0]

        return slope, intercept, ransac.inlier_mask_


def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        infer_transform = A.Compose([
            A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]) 

        transformed = infer_transform(image=image)
        image = transformed["image"]

        return image.unsqueeze(0)


def draw_ransac_line(img, slope, intercept, color=(0,0,255)):
        h, w = img.shape[:2]

    # Two endpoints
        x1 = 0
        y1 = int(slope * x1 + intercept)

        x2 = w
        y2 = int(slope * x2 + intercept)

        return cv2.line(img.copy(), (x1, y1), (x2, y2), color, 1)


def draw_paralel(img, slope, intercept,h1, color=(0,0,255)):
        h, w = img.shape[:2]

    # Two endpoints
        x1 = 0
        y1 = int(slope * x1 + intercept)

        x2 = w
        y2 = int(slope * x2 + intercept)

        return cv2.line(img.copy(), (x1, y1-h1), (x2, y2-h1), color, 1)






def process_image_with_unet(image_path):
    pixelratio=1/127.8457   #   mm/pixel  relacion de píxeles a mm





    image_name= image_path.split("\\")[-1]


    model = smp.Unet(                                   # modelo UNet con encoder ResNet34 preentrenado en ImageNet
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,      
    classes=1,
    )
    model.load_state_dict(torch.load("C:\\Industria_Teknologia\\Gradua\\TFG\\best_two_phase_claude_v2.pth", map_location=torch.device('cpu'))) # cargar pesos entrenados
    model.eval()



    


    

    with torch.no_grad():
        input_image = preprocess_image(image_path)
        output = model(input_image)
        predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    ## Quedarse con la region de la máscara más grande  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((predicted_mask > 0.5).astype(np.uint8), connectivity=8)
    if num_labels > 1:  
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
        predicted_mask = np.where(labels == largest_label, predicted_mask, 0)

    ## medidas de desgate en la máscara


    binary_mask= (predicted_mask > 0.5).astype(np.uint8)
    y,x= np.where(binary_mask==1)
    mask_coords=np.column_stack((x,y))
        
    edges= cv2.Canny((predicted_mask > 0.5).astype(np.uint8)*255, 50,150, apertureSize=3) # detección de bordes con Canny
 #usar bordes inferiores a la máscara
    h, w = edges.shape



    # Crear una máscara vacía (todo negro)
    mask_bottom = np.zeros_like(edges)

    # Para cada columna x
    for x in range(w):
        ys = np.where(edges[:, x] > 0)[0]  # todos los y donde hay borde
        if len(ys) > 0:
            y_bottom = ys.max()  # el punto más bajo
            mask_bottom[y_bottom, x] = 255  # dibujarlo en la nueva máscara

   
    
    slope, intercept, inliers = ransac_line_from_mask(mask_bottom)

    """ lines= cv2.HoughLines(mask_bottom,1,np.pi/180,100) ## detección de líneas con transformada de Hough
    if lines is not None:
    rho, theta= lines[0][0]
    a= np.cos(theta)
    b= np.sin(theta)
    x0= a*rho
    y0= b*rho
    pt1= (int(x0 +2000*(-b)), int(y0 +2000*(a)))
    pt2= (int(x0 -2000*(-b)), int(y0 -2000*(a)))
    else:
    print("No se detectó ninguna línea con la transformada de Hough.")
    ymin_contour=[]                       #coordenadas de contorno inferior(y min en cada x)
    x_axis=np.unique(x)
    for xi in x_axis:
        y_min= np.max(y[x==xi])
        ymin_contour.append([xi, y_min])
    ymin_contour= np.array(ymin_contour)
    ## mostrar resultados
    line=cv2.fitLine(ymin_contour, cv2.DIST_L2,0,0.01,0.01)    #linea usando fitline
    vx, vy, x0, y0= line.flatten()                 #vx y vy verctores de direccion/  a y b son vectores normales
    b=vx
    a=-vy
    pt1= (int(x0 +2000*(-b)), int(y0 +2000*(a)))
    pt2= (int(x0 -2000*(-b)), int(y0 -2000*(a)))  
     """




    #vectores unitarios de la línea normal y dirección

    mean_x = np.mean(mask_bottom[:,0])
    mean_y = slope * mean_x + intercept

    pt0 = np.array([mean_x, mean_y], dtype=float)




    line_dir = np.array([1, slope]) / np.linalg.norm([1, slope])
    line_normal = np.array([-slope, 1]) / np.linalg.norm([-slope, 1])



    #coordenadas proyectadas de los puntos de la máscara sobre la línea normal y dirección
    normal_coords= np.dot(mask_coords - np.array(pt0), line_normal) 
    tangent_coords= np.dot(mask_coords - np.array(pt0), line_dir)

    bin_width = 1
    bins = np.arange(np.min(tangent_coords), np.max(tangent_coords), bin_width)

    heights = []
    sum_normal_distance = 0
    max_normal_distance = 0
    max_nor_pos = 0
    max_nor_dis_line=0
    max_nor_dis_line_pos=0
    sum_nor_dis_line=0
    notch_points= []
    for b in bins:
        norm_coords_at_bin = normal_coords[(tangent_coords >= b) & (tangent_coords < b + bin_width)]
        if len(norm_coords_at_bin) > 0:
            nor_dis_line= abs(np.min(norm_coords_at_bin))



            min_dis_line= np.max(norm_coords_at_bin)
            heights.append((b, nor_dis_line))
 
            sum_nor_dis_line= sum_nor_dis_line + nor_dis_line

            if min_dis_line  < -4  :   # umbral para considerar muesca
                notch_points.append((b, min_dis_line))
       
            if nor_dis_line> max_nor_dis_line:
                max_nor_dis_line= nor_dis_line
                max_nor_dis_line_pos= b




    ##    FILTRAR OUTLIERS EN HEIGHTS USANDO IQR  ##


    heights_array = np.array([h[1] for h in heights])

    Q1 = np.percentile(heights_array, 25)
    Q3 = np.percentile(heights_array, 75)
    IQR = Q3 - Q1
    min_h = Q1 - 1.5* IQR  
    max_h = Q3 + 1.5* IQR


        

    filtered_heights = []
    for h in heights:
        if h[1] >= min_h and h[1] <= max_h:
            filtered_heights.append(h)


    #bin minimo y maximo en filtered heights

    min_bin = min([h[0] for h in filtered_heights])
    max_bin = max([h[0] for h in filtered_heights])

    #linea vertical en min_bin y max_bin
    point_min1 = (pt0 + line_dir * min_bin).astype(int)
    point_max1 = (pt0 + line_dir * max_bin).astype(int) 


    ## IDENTIFICAR MUESCAS ##
    #masca de muescas en notch_points
    notch_mask= np.zeros_like(binary_mask)
    for np_bin, np_height in notch_points:
        notch_coords= normal_coords[(tangent_coords >= np_bin) & (tangent_coords < np_bin + bin_width)]
        notch_max= np.max(notch_coords)
        notch_point1= (pt0 + line_dir * np_bin ).astype(int)
        notch_point2= (pt0 + line_dir * np_bin + line_normal * notch_max).astype(int)
        cv2.line(notch_mask, (notch_point1[0], notch_point1[1]), (notch_point2[0], notch_point2[1]), 1, 1)

    n_lab, lab, stat, centr = cv2.connectedComponentsWithStats((notch_mask > 0.5).astype(np.uint8), connectivity=8)

    #areaa de la muesca en píxeles y mm2
    notch_area= np.sum(notch_mask)
    notch_areamm2= notch_area * (pixelratio*pixelratio)


    ## Medidas finales de desgaste ##
    Vbmax_px= max([h[1] for h in filtered_heights])    # altura máxima de desgaste en píxeles
    Vbavg_px= sum([h[1] for h in filtered_heights]) / len(filtered_heights)   # altura media de desgaste en píxeles
    Vbmax_mm= Vbmax_px * pixelratio   # altura máxima de desgaste en mm
    Vbavg_mm= Vbavg_px * pixelratio   # altura media de desgaste en mm

    ## Area
    mask_area= np.sum(binary_mask)
    mask_areamm2= mask_area * (pixelratio**2)

                
    line_normal_unit = line_normal / np.linalg.norm(line_normal)
    offset_min = line_normal_unit * min_h
    offset_max = line_normal_unit * max_h




    # mostrar mascara predicha sobre la imagen original
    mask_image= (predicted_mask >0.5).astype(np.uint8)*255 
    image= cv2.imread(image_path)
    color_mask= np.zeros_like(image)
    color_mask[mask_image==255]= [0,255,0] # máscara verde
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)


    #añadir muescas en naranja
    color_notch= np.zeros_like(image)
    color_notch[notch_mask==1]= [0,165,255] # máscara naranja
    overlay = cv2.addWeighted(overlay, 1.0, color_notch, 0.5, 0)



    #mostrar línea paralela en el punto más bajo de la máscara


    ## MOSTRAR LINEA RANSAC EN IMAGEN
   


    cv2.line(overlay, point_min1, (point_min1[0], point_min1[1]-200), (255,0,255), 1)
    cv2.line(overlay, point_max1, (point_max1[0], point_max1[1]-200), (255,0,255), 1)

    overlay = draw_ransac_line(overlay, slope, intercept, color=(255,0,255))
    #overlay = draw_paralel(overlay, slope, intercept, offset_max, color=(100,0,255))
    #overlay = draw_paralel(overlay, slope, intercept, offset_min, color=(255,0,100)) 
    ## MOSTRAR RESULTADOS CUADRO DE TEXTO##

    cv2.rectangle(overlay, (10,10), (750,130), (211,211,211), -1)
    cv2.putText(overlay, f'Area: {mask_areamm2:.2f} mm2/{mask_area:.2f}px2', (30,30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
    cv2.putText(overlay, f'Vbmax: {Vbmax_mm:.2f} mm/{Vbmax_px:.2f} px', (30,60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
    cv2.putText(overlay, f'Vb: {Vbavg_mm:.2f} mm/{Vbavg_px:.2f} px', (30,90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)      
    cv2.putText(overlay, f'Ratio: {pixelratio}mm/px', (30,120), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)

    cv2.putText(overlay, f'Notch Area: {notch_areamm2:.2f} mm2/{notch_area:.2f}px2', (350,30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)    
    cv2.putText(overlay, f'Notch Count: {n_lab -1}', (350,60),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
    cv2.putText(overlay, f'||', (500,60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,165,255), 2)



    # Guardar imágenes de resultados
    cv2.imwrite("C:\\Ekain\\data\\images1\\results\\" + image_name + "_overlay_final.jpg", overlay)


    ## Mostrar imágenes de resultados
    
    cv2.imshow("Overlay", overlay)
    cv2.imshow("Predicted Mask", mask_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return Vbmax_mm, Vbavg_mm, mask_area, notch_area, mask_image, overlay

if __name__ == "__main__":
    process_image_with_unet(image_path)