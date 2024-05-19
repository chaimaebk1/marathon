from django.shortcuts import render,redirect
from django.http import HttpResponse
import os
import shutil
import numpy as np
import cv2
import supervision as sv
import easyocr
from inference import get_model
import pandas as pd

#Réalisés par : BALAKI CHAIMAE - BARHMI YOUSRA - EJJIYAR YOUSSEF - JARAF SALMA - HARRIZI MARIAM

#page upload
def upload(request):
    return render(request, 'upload.html')
#page search
def search(request):
    return render(request, 'search.html')

#upload d'une ou plusieurs images
def upload_images(request):
    if request.method == 'POST' and request.FILES.getlist('image'):
        for image in request.FILES.getlist('image'):
            save_image(image)
        #return HttpResponse("Images téléchargées avec succès.")
    return redirect('/')

#enregistrer les images au niveau du dossier data
def save_image(image):
    directory = 'marathon/static/img_nontraite'
    directory2 = 'marathon/static/data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    with open(os.path.join(directory, image.name), 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)

# appelle de notre modele
model = get_model(model_id="bib-number-x7gbv/2", api_key='rjoa6cX03rvaUBP29cYH')

# Get the model
def detect_objects_and_ocr(image_file):
    # Perform inference
    image = cv2.imread(image_file)
    results = model.infer(image)
    reader = easyocr.Reader(['en'], gpu=False)
    # Extract labels and detections
    labels = [i.class_name for i in results[0].predictions]
    detections = sv.Detections.from_roboflow(results[0].dict(by_alias=True, exclude_none=True))

    # If no objects are detected, call update_csv_autre function and return
    if not detections:
        update_csv_autre(image_file)
        return

    # Initialize annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate the image
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    dossard_numbers = []
    # Iterate over each detection
    for i in range(len(detections.xyxy)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detections.xyxy[i]

        # Crop the region of interest
        cropped_region = image[int(y1):int(y2), int(x1):int(x2)]

        # Convert cropped region to grayscale
        gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        results = reader.readtext(gray)
        #print(f"results : {results}")
        # Process OCR results
        digits = ''
        for result in results:
            text = result[1]  # Concatenate all detected text
            digits = ''.join(filter(str.isdigit, text))
            if digits:
                dossard_numbers.append(digits)
        #print(f"OCR result for Prediction {i}:\n{digits}")
    if dossard_numbers:
      for bib_number in dossard_numbers:
          update_csv(bib_number, image_file)
    else:
        update_csv_autre(image_file)
    # Display the annotated image
    #sv.plot_image(annotated_image)


#on met à jour notre fichier csv
def update_csv(number, image_name):
    csv_file = "data.csv"

    # Check if the CSV file exists
    if not os.path.isfile(csv_file):
        # If CSV file does not exist, create it with an empty DataFrame
        df = pd.DataFrame()
    else:
        # Load CSV into DataFrame
        df = pd.read_csv(csv_file)
    # Check if the number column exists
    if str(number) in df.columns:
        # Number column exists, check if the image name already exists in the column
        if image_name not in df[str(number)].values:
            # Find the last row index in the column
            last_row_index = df[str(number)].last_valid_index()
            if last_row_index is None:
                # If no valid index found, set it to 0
                last_row_index = 0
            else:
                # Increment the index by 1 to add the new image name in the next row
                last_row_index += 1
            df.at[last_row_index, str(number)] = image_name
    else:
        # Number column does not exist, create a new column with number as header
        if df.empty:
            # If DataFrame is empty, create the new column and initialize the first cell
            df[str(number)] = ""
            df.at[0, str(number)] = image_name
        else:
            # Add the new column and initialize the first cell
            df[str(number)] = ""
            df.at[0, str(number)] = image_name

    # Save DataFrame back to CSV
    df.to_csv(csv_file, index=False)


def update_csv_autre(image_name):
    csv_file = "data.csv"
    df = pd.read_csv(csv_file) if os.path.isfile(csv_file) else pd.DataFrame()

    if 'AUTRE' not in df.columns:
        df['AUTRE'] = ""  # Create 'AUTRE' column if it doesn't exist

    # Convert all filenames in 'AUTRE' column to lowercase for case-insensitive comparison
    df['AUTRE'] = df['AUTRE'].astype(str).str.lower()

    # Check if image_name already exists in the 'AUTRE' column
    if image_name.lower() in df['AUTRE'].values:
        print(f"File '{image_name}' already exists in the 'AUTRE' column. Skipping.")
        return

    # Add the image_name to the 'AUTRE' column
    last_index = df['AUTRE'].last_valid_index()
    if last_index is None:
        last_index = 0
    else:
        last_index += 1
    df.at[last_index, 'AUTRE'] = image_name
    df.to_csv(csv_file, index=False)



#classification des images suivant le numero de dossard
def classify_images(request):
    directory = "marathon/static/data"
    directory2 = "marathon/static/img_nontraite"

    # Parcourir les fichiers dans le répertoire
    for filename in os.listdir(directory2):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            print(f" file : {filename}")
            image_path = os.path.join(directory2, filename)
            shutil.move(image_path, os.path.join(directory, filename))
            image_path = os.path.join(directory, filename)
            detect_objects_and_ocr(image_path)
            
        else:
            print(f"Skipping non-image file: {filename}")
    return redirect('/')



#fonction recherche qui selectionne les urls des images depuis le fichier data.csv
def recherche(request):
    if request.method == 'GET':
        return render(request, 'search.html')
    #on recupere le contenu de notre input
    if request.method == 'POST':
        dossard_number = request.POST.get('dossardNumber', '')
    
    file_path = "data.csv"
    column_name = dossard_number
    try:
        #on lit le fichier csv
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            #si la colonne qu'on recherche existe on élimine par la suite les valeurs nulles NaN
            df_filtered = df.dropna(subset=[column_name])
            #on la transforme sous forme d'une liste
            column_data = df_filtered[column_name].tolist()
            #on change la structure du chemin
            image_paths = ["/"+path.replace('\\', '/').replace('marathon/', '') for path in column_data]
            #image_paths = column_data
        else:
            image_paths = []
    except FileNotFoundError:
        image_paths = []
    
    # Passer la liste au contexte de rendu de la page HTML
    context = {'image_paths': image_paths}
    print(image_paths)
    # Rendre la page HTML avec la liste
    return render(request, 'result.html', context)




