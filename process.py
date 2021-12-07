import glob
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Dict, List

import numpy as np
import pdfplumber
import tensorflow as tf
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

img_height = 180
img_width = 180

class_names = sorted([x.split("/")[-1] for x in glob.glob("combined/train/*")])


def make_disclosure_ranges(cd: Dict) -> List:
    """Make guesses as to where the disclosure's are based on pattern matching

    :param cd:
    :return: List of Page numbers
    """
    keys = [int(k) for k, v in cd.items() if v != "not-disclosures"]

    def group(L):
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last:
                last = n
            elif n - 2 == last:
                last = n
            else:
                yield first, last
                first = last = n
        if last - first > 3:
            yield first, last

    final_range = []
    for rang in group(keys):
        r = list(range(rang[0], rang[1] + 1))
        if len(r) <= 3:
            continue
        try:
            values = [cd[i] for i in r]
            # Sometimes do this sometimes don't
            # if "start" not in values or "certification" not in values:
            #     continue
            final_range.append(r)
        except Exception as e:
            print("ERROR", str(e))
            pass

    return final_range


def output_disclosures(filepath: str, final_range: List) -> None:
    """Write out the disclosures

    :param filepath: To the original PDF
    :type filepath: str
    :param final_range: The ranges of the disclosures to extract
    :type final_range: List
    :return: None
    """
    for r in final_range:
        page_numbers = [x - 1 for x in r]
        pdf = PdfFileReader(filepath)
        pdf_writer = PdfFileWriter()
        output_filename = f"combined/output/disclosure-{page_numbers[1]}.pdf"
        for pgnum in page_numbers:
            pdf_writer.addPage(pdf.getPage(pgnum))
        with open(output_filename, "wb") as out:
            pdf_writer.write(out)


def convert_image_to_numpy_array(filepath, page) -> np.ndarray:
    """Process the extracted image and convert to numpy array


    :param page:
    :return:
    """
    images = convert_from_path(
        filepath,
        dpi=32,
        thread_count=10,
        first_page=page.page_number,
        last_page=page.page_number,
        fmt="png",
    )

    # Load image to temp file and run the model
    with NamedTemporaryFile(suffix=".png") as image:
        images[0].save(image.name)
        # Process the image
        img = keras.preprocessing.image.load_img(
            image.name, target_size=(img_height, img_width)
        )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array


def extract_disclsoures(filepath: str) -> None:
    """Extract disclosure from hearing documents

    :return:
    """
    cd = {}
    # Load the model
    model = keras.models.load_model("combined/disclosure-model.h5")
    # model = keras.models.load_model("/Users/Palin/Code/hearing-extractor/combined/disclosure-model.h5")
    # Extract out the image and label data
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            print(f"{page.page_number} / {len(pdf.pages)} \t", end="")
            img_array = convert_image_to_numpy_array(filepath, page)
            # Run the model
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            pred = class_names[np.argmax(score)]

            # Save the prediction
            score_number = 100 * np.max(score)
            print(pred, "\t", int(score_number))
            cd[page.page_number] = pred

    logging.info("Start PDF generation")
    disclosure_ranges = make_disclosure_ranges(cd)
    output_disclosures(filepath, disclosure_ranges)


if __name__ == "__main__":
    # This should find nine disclosures in the hearing documents from 1991.
    # It misses the final one - (and maybe more)

    filepath = "combined/test/Confirmation_Hearings_on_Federal_Appoint-1991.pdf"
    extract_disclsoures(filepath)
