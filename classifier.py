import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib
import os

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name).output
    gradcam_model = tf.keras.Model(inputs=model.input, outputs=[last_conv_layer, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# Function to generate explanation
def generate_explanation(prediction, heatmap_path, original_image_path):
    explanation = f"""
    ### Explanation of the Grad-CAM Result

    **Prediction:** The model classified the image as "{prediction}."

    **What is Grad-CAM?**:
    Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that helps us visualize which parts of an image are most influential in the model's classification decision. It provides a visual map highlighting important regions in the image that affected the model's prediction.

    **How Does It Work?**:
    1. **Feature Extraction**: The model extracts features from the last convolutional layer of the neural network, which captures high-level details about the image.
    2. **Gradient Calculation**: Grad-CAM computes gradients of the class output with respect to these features. The gradients indicate how much each feature map contributed to the prediction.
    3. **Heatmap Creation**: The gradients are averaged across all feature maps to create a heatmap. This heatmap is then resized to match the original image's dimensions.
    4. **Overlay**: The heatmap is overlaid on the original image to highlight the regions that were most influential in the classification.

    **Interpreting the Heatmap**:
    - **Brighter Regions**: Areas shown in bright colors (like red or yellow) on the heatmap are the parts of the image that had the most influence on the model’s decision.
    - **Darker Regions**: Areas with darker colors (like blue or black) had less impact on the prediction.

    **Final Notes**:
    The heatmap provides a visual representation of what the model focuses on when making a classification. This helps in understanding the model's decision-making process and can be useful for debugging or improving model performance.

    You can view the heatmap generated for the given image to see which areas were most influential.
    """
    return explanation  # Ensure this returns a string

# Function to load and preprocess the image
def load_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array)

# Function to display the Grad-CAM heatmap
def display_gradcam_heatmap(img_array, original_image_path, base_model, last_conv_layer_name='conv5_block3_out'):
    heatmap = generate_gradcam_heatmap(base_model, img_array, last_conv_layer_name)
    
    # Read and resize the original image
    img = cv2.imread(original_image_path)
    img = cv2.resize(img, (224, 224))

    # Superimpose heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Display the image with heatmap overlay
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Image with Grad-CAM Heatmap")
    plt.show()

    return superimposed_img

# Main function for image classification and explainability
import os

def classify_and_explain(image_file, svm_model_path, save_dir):
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the ResNet50 model with ImageNet weights
    base_model = tf.keras.applications.ResNet50(weights='imagenet')
    
    # Load the ResNet50 model without the classification head, stopping at 'avg_pool'
    feature_extractor_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output  # Use 'avg_pool' layer to get 2048 features
    )

    # Load and preprocess the image
    img_array = load_image(image_file)
    
    # Step 1: Extract features using ResNet50 (before the classification head)
    features = feature_extractor_model.predict(img_array)  # This will give you (1, 2048)

    # Step 2: Flatten the extracted features (if necessary, but it should already be flat with shape (1, 2048))
    flattened_features = features.reshape(1, -1)  # Ensure it's a 2D array with shape (1, 2048)

    # Step 3: Load the pre-trained SVM model and classify the image
    svm_model = joblib.load(svm_model_path)  # Load your pre-trained SVM model
    prediction = svm_model.predict(flattened_features)  # Make prediction

    # Step 4: Grad-CAM heatmap generation
    superimposed_img = display_gradcam_heatmap(img_array, image_file, base_model)

    # Save the heatmap image
    heatmap_save_path = os.path.join(save_dir, 'superimposed_heatmap.jpg')
    cv2.imwrite(heatmap_save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    # Step 5: Generate explanation report
    predicted_label = "Real" if prediction[0] == 1 else "AI-Generated-Image"
    explanation = generate_explanation(predicted_label, heatmap_save_path, image_file)
    
    # Ensure the explanation is a string before writing
    if explanation is None:
        explanation = "No explanation available."

    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:  # Specify encoding explicitly
        f.write(explanation)

    print(f"Prediction: {predicted_label}")
    print(f"Heatmap saved to: {heatmap_save_path}")
    print(f"Report saved to: {report_path}")
    
# Example usage
# classify_and_explain('path/to/uploaded/image.jpg', 'path/to/svm_classifier.pkl', 'path/to/save/directory')
