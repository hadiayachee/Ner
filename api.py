import cv2
import mediapipe as mp
import math
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

app = Flask(__name__)
CORS(app)

# Virtual Keyboard code
@app.route('/virtual_keyboard_english')
def write_english():
    cap = cv2.VideoCapture(0)

    keys = [["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L","DEL"],
            ["CAPS","Z","X","C","V","B","N","M",".","SPACE"]]
    #Initialize mediapipe utilities for drawing and hand detection
    mpDraw = mp.solutions.drawing_utils
    mphand = mp.solutions.hands
    handdetection = mphand.Hands()
    #Define a Button class to represent each button on the virtual keyboard. Each button has a position, size, and text (character or action)
    class Button():
        def __init__(self, pos, text, size):
            self.pos = pos
            self.size = size
            self.text = text

    pri_color = (7,174,246)
    pri_color_hover = (0,150,215)
    pri_color_click = (0,215,129)
    #Create a list of Button objects (buttonList) that represents all the buttons on the virtual keyboard.
    #Define a function drawALL to draw all the buttons on the virtual keyboard on the provided image.

    def drawALL(img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            # Add the condition to check if the button is the "CAPS" button
            if button.text == "CAPS":
                # Change the color of the button based on the caps_lock state
                color = pri_color_click if caps_lock else pri_color
            else:
                color = pri_color

            # Determine the text case based on the caps_lock state
            text_case = button.text.lower() if not caps_lock else button.text

            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.putText(img, text_case, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img

    x_btn = 30
    y_btn = 200
    w_btn = x_btn + 50
    h_btn = y_btn + 50
    buttonList = []
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([60 * j + 160 + 20, 60 * i + 270], key, [55, 55]))  # Shifted the x-coordinate by 20
    #Set up the initial text to be displayed on the virtual keyboard screen (screen_text).
    screen_text = ""
    caps_lock = False
    #Define a function msg_box to display a message box on the provided image.
    def msg_box(img, text):
        cv2.rectangle(img, (20, 10), (20 + 260, 10 + 80), pri_color, -1)
        cv2.putText(img, f"{text}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5)
    #Set the click threshold (click_threshold) for determining whether a hand gesture corresponds to a click or not.
    click_threshold = 30

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (1000, 700), cv2.INTER_LINEAR)
        h_img, w_img, _ = img.shape

        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = handdetection.process(imgrgb)

        img = drawALL(img, buttonList)

        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handlm, mphand.HAND_CONNECTIONS)
                point_8 = handlm.landmark[8]
                point_4 = handlm.landmark[4]
                cx_8, cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
                cx_4, cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
                cv2.circle(img, (cx_8, cy_8), 4, (0, 255, 0), 3)
                cv2.circle(img, (cx_4, cy_4), 4, (0, 255, 0), 3)
                distance = math.sqrt((cx_4 - cx_8) ** 2 + (cy_4 - cy_8) ** 2)
                for button in buttonList:
                    x_hover, y_hover = button.pos
                    w_hover, h_hover = button.size
                    if x_hover < cx_8 < x_hover + w_hover and y_hover < cy_8 < y_hover + h_hover:
                        cv2.rectangle(img, button.pos, (x_hover + w_hover, y_hover + h_hover), pri_color_hover, -1)
                        cv2.putText(img, button.text, (x_hover + 20, y_hover + 40), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (255, 255, 255), 2)
                        if distance <= click_threshold:
                            cv2.rectangle(img, button.pos, (x_hover + w_hover, y_hover + h_hover), pri_color_click, -1)
                            cv2.putText(img, button.text, (x_hover + 20, y_hover + 40), cv2.FONT_HERSHEY_PLAIN, 2,
                                        (255, 255, 255), 2)
                            if button.text == "DEL":
                                screen_text = screen_text[:-1]  # Remove the last character
                            elif button.text == "SPACE":
                                screen_text += " "
                            elif button.text == "CAPS":
                                caps_lock = not caps_lock  # Toggle the Caps Lock state
                            else:
                                if caps_lock:
                                    screen_text += button.text.upper()
                                else:
                                    screen_text += button.text.lower()
                            time.sleep(0.50)

        cv2.rectangle(img, (200, 180), (740, 260), pri_color, -1)
        cv2.putText(img, screen_text, (60, 240), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5)

        cv2.rectangle(img, (530, 10), (630, 110), pri_color, -1)
        cv2.rectangle(img, (420, 10), (520, 110), pri_color, -1)

        if results.multi_hand_landmarks:
            for handlm in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handlm, mphand.HAND_CONNECTIONS)
                point_8 = handlm.landmark[8]
                point_4 = handlm.landmark[4]
                cx_8, cy_8 = int(point_8.x * w_img), int(point_8.y * h_img)
                cx_4, cy_4 = int(point_4.x * w_img), int(point_4.y * h_img)
                cv2.circle(img, (cx_8, cy_8), 4, (0, 255, 0), 3)
                cv2.circle(img, (cx_4, cy_4), 4, (0, 255, 0), 3)
                distance = math.sqrt((cx_4 - cx_8) ** 2 + (cy_4 - cy_8) ** 2)
                if 530 < cx_8 < 630 and 10 < cy_8 < 110:
                    cv2.rectangle(img, (530, 10), (630, 110), pri_color_hover, -1)
                    msg_box(img, "PRINT")
                    if distance <= click_threshold:
                        cv2.rectangle(img, (530, 10), (630, 110), pri_color_click, -1)
                        if len(screen_text) == 0:
                            msg_box(img, "EMPTY")
                        else:
                            file = open("data.txt", "w")
                            file.write(str(screen_text))
                            file.close()

                elif 420 < cx_8 < 520 and 10 < cy_8 < 110:
                    cv2.rectangle(img, (500, 10), (600, 110), pri_color_hover, -1)
                    msg_box(img, "CLEAR")
                    if distance <= click_threshold:
                        cv2.rectangle(img, (420, 10), (520, 110), pri_color_click, -1)
                        screen_text = ""

        cv2.imshow("Virtual Keyboard", img)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Arabic NER API
model_name_arabic = 'aubmindlab/bert-base-arabertv02'
arabert_model = AutoModelForTokenClassification.from_pretrained('modelarabert')
tokenizer = AutoTokenizer.from_pretrained(model_name_arabic)
entity_mapping = {
    "LABEL_0": "B-LOC",
    "LABEL_1": "NONE",
    "LABEL_2": "B-PERS",
    "LABEL_3": "I-PERS",
    "LABEL_4": "B-ORG",
    "LABEL_5": "I-LOC",
    "LABEL_6": "I-ORG",
    "LABEL_7": "B-MISC",
    "LABEL_8": "I-MISC",
}

pipe_arabic = pipeline("ner", model=arabert_model, tokenizer=tokenizer)


@app.route('/predictarabic', methods=['POST'])
def predict_arabic():
    text = request.json['text']
    print(text)
    results = pipe_arabic(text)
    
    merged_results = []
    current_word = ""
    current_label = ""
    
    for result in results:
        word = result['word']
        label = result['entity']
    
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_word:
                merged_results.append({'word': current_word, 'entity': current_label})
                current_word = ""
            current_word = word
            current_label = label
    
    # Add the last merged token if it exists
    if current_word:
        merged_results.append({'word': current_word, 'entity': current_label})
    
    # Construct the response
    predictions = []
    for result in merged_results:
        word = result['word']
        label = entity_mapping.get(result['entity'])
        if label != "NONE":
            predictions.append({
                'word': word,
                'entity': label
            })
    
    response = {
        'predictions': predictions
    }
    
    return jsonify(response)


# English NER API
#model and tokenizer are loaded using the pipeline function from the transformers library.
model_name_english = 'bert-base-cased'
model_path = 'outputs'

# Load the model and tokenizer
ner = pipeline('ner', model=model_path, tokenizer=model_name_english)

@app.route('/predictenglish', methods=['POST'])
def predict_english():
    input_text = request.json['text']
    # (ner) to extract named entities from the input_text using the loaded BERT model and tokenizer.
    # Perform the named entity recognition
    results = ner(input_text)

    # Merge tokens containing "##" with their preceding tokens
    merged_results = []
    current_word = ""
    current_label = ""
    for result in results:
        word = result['word']
        label = result['entity']
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_word:
                merged_results.append({'word': current_word, 'entity': current_label})
                current_word = ""
            current_word = word
            current_label = label

    # Add the last merged token if exists
    if current_word:
        merged_results.append({'word': current_word, 'entity': current_label})

    # Prepare the response
    response = {'results': merged_results}

    return jsonify(response)

if __name__ == '__main__':
    app.run()
