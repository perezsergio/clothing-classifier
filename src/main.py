from tensorflow.keras.saving import load_model  # pylint: disable=E0401
import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path


def main():
    """Run streamlit app where the user can upload an image and the model will classify it."""
    # Class names
    class_names = [
        "T_shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Top header
    st.title("Clothing Classifier 🛒🤑")
    st.text(
        "This app demonstrates a CNN model which is able to classify images of clothing\n"
        + "items in one of 10 categories:\n"
        + "".join(
            [str(i + 1) + ": " + str(e) + "\n" for i, e in enumerate(class_names)]
        )
        + "Note: The model works best on images that exclusively show the clothing item,\n"
        + "not it being worn by a person. Also it performs best for images with a white \n"
        + "background."
    )

    # File uploader
    st.subheader("Upload an image of a clothing item")
    file = st.file_uploader(
        "Upload an image of a clothing item.",
        type=["jpg", "png"],
    )

    # Load model
    root_dir = Path(__file__).parent.parent.resolve()
    model_path = root_dir / "models" / "final_model.model"
    model = load_model(str(model_path))

    # If a file is uploaded
    if file:
        # Divide UI in 2 columns
        col1, col2 = st.columns(2)

        # Open image from the uploaded file
        image = Image.open(file)

        # Display original image
        with col1:
            st.subheader("Original image")
            st.image(image, use_column_width=True)

        # Convert image to 28x28pixels grayscale
        bw_image = image.convert("L")
        resized_image = bw_image.resize((28, 28))
        with col2:
            st.subheader("Formatted image")
            st.image(resized_image, use_column_width=True)

        # Small explanation of why the image is formatted this way
        st.text(
            "The image must be reformatted as a 28x28 pixel grayscale image because\n"
            + "the model was trained on images with that format"
        )

        # Format data to feed it to the model
        img_array = np.array(resized_image)
        img_array = img_array.reshape((1, 28, 28, 1))
        img_array = 1 - img_array / 255

        # Get model prediction and confidence
        predict_prob = model.predict(img_array)
        prediction = class_names[np.argmax(predict_prob)]
        confidence = np.max(predict_prob)

        # Display it at the bottom of the page
        st.subheader("Prediction:")
        st.text(f"{prediction}, {round(100*confidence, 2)}%")

    # If no file has been uploaded yet
    else:
        st.text("No image has been uploaded yet.")


if __name__ == "__main__":
    main()
