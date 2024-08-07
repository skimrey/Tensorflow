import tkinter as tk
from tkinter import ttk
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Function to plot image
def plot_image(i, predictions_array, true_labels, images, class_names):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})", color=color)

# Function to plot value array
def plot_value_array(i, predictions_array, true_labels):
    predictions_array, true_label = predictions_array[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    bar = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    bar[predicted_label].set_color('red')
    bar[true_label].set_color('blue')

# Initialize the main window
root = tk.Tk()
root.title("TensorFlow Model Builder")

# Dataset categories and items
datasets = {
    "Dataset Collections": ["longt5", "xtreme", "3d", "aflw2k3d", "smallnorb", "smartwatch_gestures"],
    "Abstractive text summarization": ["aeslc", "billsum", "booksum", "newsroom", "reddit", "reddit_tifu", "samsum", "scientific_papers"],
    "Age": ["dices"],
    "Anomaly detection": ["ag_news_subset", "caltech101", "kddcup99", "lost_and_found", "stl10"],
    "Audio": ["accentdb", "common_voice", "crema_d", "dementiabank", "fuss", "groove", "gtzan", "gtzan_music_speech", "librispeech", "libritts", "ljspeech", "nsynth", "savee", "speech_commands", "spoken_digit", "tedlium", "user_libri_audio", "vctk", "voxceleb", "voxforge", "xtreme_s", "yes_no"],
    "Biology": ["ogbg_molpcba"],
    "Captioning": ["coherency_en", "coherency_zh", "common_gen", "image_captioning", "nocaps", "nlvr", "visual7w", "wmt", "xquad"],
    "Classification": ["affirmation", "amazon_polarity", "civil_comments", "imdb", "sogou_news", "xglue", "yahoo_answers"],
    "Computer vision": ["bigearthnet", "cityscapes", "coco", "eurosat", "imagenet2012", "kitti", "mnist", "object_detection", "quickdraw", "voc", "waymo_open_dataset"],
    "Contrastive learning": ["audio_set", "finetune", "melodic", "scene", "vggface2"],
    "Conversion": ["cmu_arctic", "cstr_vctk", "libritts_r", "mfa"],
    "Data augmentation": ["ctb", "ocr", "wmt"],
    "Dialog": ["airdialogue", "babi_dialogue", "camrest676", "dstc2", "frame_semantic_parsing", "kvr", "multiwoz", "taskmaster1", "taskmaster2"],
    "Disentangled representation": ["3dshapes", "disentanglement_lib", "factor"],
    "Emotion": ["emotone"],
    "Factual verification": ["danish", "fever", "kilt_tasks", "tab_fact", "wmt", "xnli"],
    "Facial recognition": ["calfw", "cfw", "dfw", "glint360k", "glintasia", "lfw"],
    "General domain": ["arc", "cosmosqa", "hsnc", "msmarco"],
    "Handwritten text recognition": ["audiomnist", "kondate"],
    "Image classification": ["fashion_mnist", "mnist", "oxford_flowers102"],
    "Image enhancement": ["ffhq", "mapillary_vistas"],
    "Image generation": ["biggan", "dtd", "imaginaire", "nucleus", "rafsimons"],
    "Image segmentation": ["penn_fudan_ped", "voc", "ycb_video"],
    "Language modeling": ["open_subtitles", "para_crawl", "ted_hrlr"],
    "Machine translation": ["opus_books", "ted_hrlr", "wmt"],
    "Named entity recognition": ["conll2003"],
    "Natural language processing": ["snli", "squad", "super_glue", "tydiqa"],
    "Object detection": ["coco", "open_images", "voc"],
    "Paraphrase identification": ["mrpc", "paws", "quora"],
    "POS tagging": ["conll2000", "ptb", "universal_dependencies"],
    "Question answering": ["boolq", "complexwebquestions", "cosmosqa", "hotpotqa", "narrativeqa", "natural_questions", "newsqa", "openbookqa", "squad", "wikisql"],
    "Relational reasoning": ["analogies", "babi", "clutrr"],
    "Text classification": ["ag_news", "dbpedia", "emotion", "imdb", "sms_spam", "yahoo_answers"],
    "Text similarity": ["sts", "word_similarity"],
    "Translation": ["europarl", "iwslt", "ted_hrlr", "wmt"],
    "Voice cloning": ["cmu_arctic", "lj_speech"],
    "Word sense disambiguation": ["semeval2015", "semeval2018"],
}

# Function to update the dropdown based on category selection
def update_dropdown(*args):
    category = category_var.get()
    dataset_dropdown['values'] = datasets.get(category, [])

# Function to create and train the model based on user inputs
def build_model():
    # Get user inputs
    num_layers = int(num_layers_var.get())
    num_neurons = int(num_neurons_var.get())
    dataset_name = dataset_var.get()
    
    # Load dataset
    dataset, metadata = tfds.load(dataset_name, as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    
    def normalize(images, labels):
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, labels

    train_dataset = train_dataset.map(normalize)
    test_dataset = test_dataset.map(normalize)
    train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(32)
    test_dataset = test_dataset.cache().batch(32)

    # Build model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    model.fit(train_dataset, epochs=5, steps_per_epoch=num_train_examples//32)
    
    test_loss, test_accuracy = model.evaluate(test_dataset)
    result_label.config(text=f'Test accuracy: {test_accuracy:.4f}')
    
    for test_images, test_labels in test_dataset.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)
        
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(0, predictions, test_labels, test_images, class_names)
        plt.subplot(1,2,2)
        plot_value_array(0, predictions, test_labels)
        plt.show()

# Create GUI elements
num_layers_label = ttk.Label(root, text="Number of Layers:")
num_layers_label.grid(column=0, row=0, padx=10, pady=10)
num_layers_var = tk.StringVar(value="3")
num_layers_entry = ttk.Entry(root, textvariable=num_layers_var)
num_layers_entry.grid(column=1, row=0, padx=10, pady=10)

num_neurons_label = ttk.Label(root, text="Number of Neurons per Layer:")
num_neurons_label.grid(column=0, row=1, padx=10, pady=10)
num_neurons_var = tk.StringVar(value="128")
num_neurons_entry = ttk.Entry(root, textvariable=num_neurons_var)
num_neurons_entry.grid(column=1, row=1, padx=10, pady=10)

# Create category dropdown
category_var = tk.StringVar()
category_label = ttk.Label(root, text="Category:")
category_label.grid(column=0, row=2, padx=10, pady=10)
category_dropdown = ttk.Combobox(root, textvariable=category_var)
category_dropdown['values'] = list(datasets.keys())
category_dropdown.grid(column=1, row=2, padx=10, pady=10)
category_dropdown.bind("<<ComboboxSelected>>", update_dropdown)

# Create dataset dropdown
dataset_var = tk.StringVar()
dataset_label = ttk.Label(root, text="Dataset:")
dataset_label.grid(column=0, row=3, padx=10, pady=10)
dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var)
dataset_dropdown.grid(column=1, row=3, padx=10, pady=10)

build_button = ttk.Button(root, text="Build and Train Model", command=build_model)
build_button.grid(column=0, row=4, columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=5, columnspan=2, padx=10, pady=10)

# Start the main loop
root.mainloop()
