from functions import *


# Parse cli arguments
parser = argparse.ArgumentParser(description='Classify an image')
parser.add_argument("image_path", help='path to image')
parser.add_argument("saved_model", help= 'name of model')
parser.add_argument('-k', '--top_k', type=int, help='print top k classes')
parser.add_argument('-c', '--category_names', help='name categories')
args = parser.parse_args()

image_path = args.image_path
saved_model = args.saved_model
top_k = args.top_k
category_names = args.category_names

# Load model
model = tf.keras.models.load_model(saved_model, custom_objects = {'KerasLayer':hub.KerasLayer})

# Prediction
if top_k:
    probs, classes = predict(image_path, model, top_k=top_k)
else:
    probs, classes = predict(image_path, model)

if category_names:
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    classes_str = [class_names[str(i)] for i in classes]
else:
    classes_str = classes

print(classes_str)
print(probs)
