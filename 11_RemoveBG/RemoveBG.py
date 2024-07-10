# Importing Required Modules 
from rembg import remove 
from PIL import Image 

input_path = "OMID.jpg"
output_path = "export.png" 

input = Image.open(input_path) 
output = remove(input) 
output.save(output_path) 
