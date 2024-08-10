import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
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
    num_layers = int(num_layers_var.get())
    layer_configs = [layer_config[i] for i in range(num_layers)]
    
    dataset_name = dataset_var.get()
    epochs = int(epochs_var.get())
    batch_size = int(batch_size_var.get())
    steps_per_epoch = int(steps_per_epoch_var.get())

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
    train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(batch_size)
    test_dataset = test_dataset.cache().batch(batch_size)

    # Build model
    model = tf.keras.Sequential()
    first_layer_config = layer_configs[0]

    # Input shape for the first layer
    model.add(tf.keras.layers.Input(shape=(int(first_layer_config['input_shape_x'].get()),
                                            int(first_layer_config['input_shape_y'].get()),
                                            int(first_layer_config['input_shape_z'].get()))))

    for config in layer_configs:
        layer_type = config['type'].get()
        units = int(config['units'].get())
        activation = config['activation'].get()

        if layer_type == "Dense":
            model.add(tf.keras.layers.Dense(units, activation=activation))
        elif layer_type == "Conv2D":
            kernel_size = (int(config['kernel_size_x'].get()), int(config['kernel_size_y'].get()))
            model.add(tf.keras.layers.Conv2D(filters=units, kernel_size=kernel_size, activation=activation))
        elif layer_type == "MaxPooling2D":
            pool_size = (int(config['pool_size_x'].get()), int(config['pool_size_y'].get()))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
        elif layer_type == "Flatten":
            model.add(tf.keras.layers.Flatten())
        elif layer_type == "SimpleRNN":
            model.add(tf.keras.layers.SimpleRNN(units, activation=activation))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset,
                        steps_per_epoch=steps_per_epoch)
    
    root.model = model  # Save the trained model in the root widget
    messagebox.showinfo("Model Built", "Model has been built and trained successfully.")
    return model
    
    
def save_model():
    if hasattr(root, 'model'):
        model = root.model
        save_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
        if save_path:
            model.save(save_path)
            messagebox.showinfo("Model Saved", f"Model successfully saved to {save_path}")
    else:
        messagebox.showerror("Error", "No model to save. Train a model first.")

# Update this function to add pooling and kernel size inputs
def update_neuron_entries(*args):
    num_layers = int(num_layers_var.get())
    for widget in neuron_frame.winfo_children():
        widget.destroy()
    layer_config.clear()

    for i in range(num_layers):
        config = {}

        # Input Shape for the first Layer only
        if i == 0:
            input_shape_frame = ttk.Frame(neuron_frame)
            input_shape_frame.grid(column=0, row=i*10, columnspan=8, padx=10, pady=5, sticky='w')

            input_shape_label = ttk.Label(input_shape_frame, text=f"Input Shape (Layer {i+1}):")
            input_shape_label.grid(column=0, row=0, padx=10, pady=5, sticky='w')

            input_shape_x_label = ttk.Label(input_shape_frame, text="Width:")
            input_shape_x_label.grid(column=1, row=0, padx=5, pady=5, sticky='w')
            input_shape_x_var = tk.StringVar(value="28")
            input_shape_x_entry = ttk.Entry(input_shape_frame, textvariable=input_shape_x_var, width=10)
            input_shape_x_entry.grid(column=2, row=0, padx=10, pady=5)

            input_shape_y_label = ttk.Label(input_shape_frame, text="Height:")
            input_shape_y_label.grid(column=3, row=0, padx=5, pady=5, sticky='w')
            input_shape_y_var = tk.StringVar(value="28")
            input_shape_y_entry = ttk.Entry(input_shape_frame, textvariable=input_shape_y_var, width=10)
            input_shape_y_entry.grid(column=4, row=0, padx=10, pady=5)

            input_shape_z_label = ttk.Label(input_shape_frame, text="Channels:")
            input_shape_z_label.grid(column=5, row=0, padx=5, pady=5, sticky='w')
            input_shape_z_var = tk.StringVar(value="1")
            input_shape_z_entry = ttk.Entry(input_shape_frame, textvariable=input_shape_z_var, width=10)
            input_shape_z_entry.grid(column=6, row=0, padx=10, pady=5)

            config['input_shape_x'] = input_shape_x_var
            config['input_shape_y'] = input_shape_y_var
            config['input_shape_z'] = input_shape_z_var

        # Layer Type
        layer_label = ttk.Label(neuron_frame, text=f"Layer {i+1} Type:")
        layer_label.grid(column=0, row=i*10+1, padx=10, pady=5, sticky='w')
        layer_type_var = tk.StringVar(value="Flatten")
        layer_type_dropdown = ttk.Combobox(neuron_frame, textvariable=layer_type_var)
        layer_type_dropdown['values'] = ["Dense", "Conv2D", "MaxPooling2D", "Flatten", "SimpleRNN"]
        layer_type_dropdown.grid(column=1, row=i*10+1, padx=10, pady=5)
        layer_type_dropdown.bind("<<ComboboxSelected>>", lambda e, idx=i: update_layer_specific_entries(idx))
        config['layer_type_var'] = layer_type_var

        # Units/Filters
        units_label = ttk.Label(neuron_frame, text=f"Units/Filters (Layer {i+1}):")
        units_label.grid(column=2, row=i*10+1, padx=10, pady=5, sticky='w')
        units_var = tk.StringVar(value="128")
        units_entry = ttk.Entry(neuron_frame, textvariable=units_var, width=10)
        units_entry.grid(column=3, row=i*10+1, padx=10, pady=5)
        config['units'] = units_var

        # Activation Function
        activation_label = ttk.Label(neuron_frame, text=f"Activation (Layer {i+1}):")
        activation_label.grid(column=4, row=i*10+1, padx=10, pady=5, sticky='w')
        activation_var = tk.StringVar(value="relu")
        activation_dropdown = ttk.Combobox(neuron_frame, textvariable=activation_var)
        activation_dropdown['values'] = ["relu", "sigmoid", "tanh", "softmax", "softplus", "softsign", "swish"]
        activation_dropdown.grid(column=5, row=i*10+1, padx=10, pady=5)
        config['activation'] = activation_var

        # Kernel Size Frame
        kernel_size_frame = ttk.Frame(neuron_frame)
        kernel_size_frame.grid(column=0, row=i*10+2, columnspan=8, padx=10, pady=5, sticky='w')
        kernel_size_frame.grid_forget()  # Hidden by default

        # Pool Size Frame
        pool_size_frame = ttk.Frame(neuron_frame)
        pool_size_frame.grid(column=0, row=i*10+3, columnspan=8, padx=10, pady=5, sticky='w')
        pool_size_frame.grid_forget()  # Hidden by default

        config['kernel_size_frame'] = kernel_size_frame
        config['pool_size_frame'] = pool_size_frame

        layer_config.append(config)

    def update_layer_specific_entries(layer_index):
        layer_type = layer_config[layer_index]['layer_type_var'].get()

        kernel_frame = layer_config[layer_index]['kernel_size_frame']
        pool_frame = layer_config[layer_index]['pool_size_frame']

        # Hide both frames initially
        kernel_frame.grid_forget()
        pool_frame.grid_forget()

        if layer_type == "Conv2D":
            kernel_frame.grid(row=layer_index*10+2, column=0, columnspan=8, padx=10, pady=5, sticky='w')  # Show kernel size frame
            pool_frame.grid_forget()  # Hide pool size frame

            # Kernel Size
            for widget in kernel_frame.winfo_children():
                widget.destroy()

            ttk.Label(kernel_frame, text=f"Kernel Size (Layer {layer_index + 1}):").grid(column=0, row=0, padx=10, pady=5, sticky='w')
            ttk.Label(kernel_frame, text="Width:").grid(column=1, row=0, padx=5, pady=5, sticky='w')
            kernel_size_x_var = tk.StringVar(value="3")
            ttk.Entry(kernel_frame, textvariable=kernel_size_x_var, width=5).grid(column=2, row=0, padx=10, pady=5)
            ttk.Label(kernel_frame, text="Height:").grid(column=3, row=0, padx=5, pady=5, sticky='w')
            kernel_size_y_var = tk.StringVar(value="3")
            ttk.Entry(kernel_frame, textvariable=kernel_size_y_var, width=5).grid(column=4, row=0, padx=10, pady=5)
            layer_config[layer_index]['kernel_size_x'] = kernel_size_x_var
            layer_config[layer_index]['kernel_size_y'] = kernel_size_y_var

        elif layer_type == "MaxPooling2D":
            kernel_frame.grid_forget()  # Hide kernel size frame
            pool_frame.grid(row=layer_index*10+2, column=0, columnspan=8, padx=10, pady=5, sticky='w')  # Show pool size frame

            # Pool Size
            for widget in pool_frame.winfo_children():
                widget.destroy()

            ttk.Label(pool_frame, text=f"Pool Size (Layer {layer_index + 1}):").grid(column=0, row=0, padx=10, pady=5, sticky='w')
            ttk.Label(pool_frame, text="Width:").grid(column=1, row=0, padx=5, pady=5, sticky='w')
            pool_size_x_var = tk.StringVar(value="2")
            ttk.Entry(pool_frame, textvariable=pool_size_x_var, width=5).grid(column=2, row=0, padx=10, pady=5)
            ttk.Label(pool_frame, text="Height:").grid(column=3, row=0, padx=5, pady=5, sticky='w')
            pool_size_y_var = tk.StringVar(value="2")
            ttk.Entry(pool_frame, textvariable=pool_size_y_var, width=5).grid(column=4, row=0, padx=10, pady=5)
            layer_config[layer_index]['pool_size_x'] = pool_size_x_var
            layer_config[layer_index]['pool_size_y'] = pool_size_y_var




# Create GUI elements
num_layers_label = ttk.Label(root, text="Number of Layers:")
num_layers_label.grid(column=0, row=0, padx=10, pady=10)
num_layers_var = tk.StringVar(value="1")
num_layers_entry = ttk.Entry(root, textvariable=num_layers_var)
num_layers_entry.grid(column=1, row=0, padx=10, pady=10)
num_layers_var.trace_add("write", update_neuron_entries)

# Frame to hold neuron entries
neuron_frame = ttk.Frame(root)
neuron_frame.grid(column=0, row=1, columnspan=2)

layer_config = []
update_neuron_entries()

# Additional Model Parameters
epochs_label = ttk.Label(root, text="Epochs:")
epochs_label.grid(column=0, row=2, padx=10, pady=10)
epochs_var = tk.StringVar(value="5")
epochs_entry = ttk.Entry(root, textvariable=epochs_var)
epochs_entry.grid(column=1, row=2, padx=10, pady=10)

batch_size_label = ttk.Label(root, text="Batch Size:")
batch_size_label.grid(column=0, row=3, padx=10, pady=10)
batch_size_var = tk.StringVar(value="32")
batch_size_entry = ttk.Entry(root, textvariable=batch_size_var)
batch_size_entry.grid(column=1, row=3, padx=10, pady=10)

steps_per_epoch_label = ttk.Label(root, text="Steps per Epoch:")
steps_per_epoch_label.grid(column=0, row=4, padx=10, pady=10)
steps_per_epoch_var = tk.StringVar(value="1000")
steps_per_epoch_entry = ttk.Entry(root, textvariable=steps_per_epoch_var)
steps_per_epoch_entry.grid(column=1, row=4, padx=10, pady=10)

# Dataset Selection
category_label = ttk.Label(root, text="Dataset Category:")
category_label.grid(column=0, row=5, padx=10, pady=10)
category_var = tk.StringVar(value="Dataset Collections")
category_dropdown = ttk.Combobox(root, textvariable=category_var)
category_dropdown['values'] = list(datasets.keys())
category_dropdown.grid(column=1, row=5, padx=10, pady=10)
category_var.trace("w", update_dropdown)

dataset_label = ttk.Label(root, text="Dataset:")
dataset_label.grid(column=0, row=6, padx=10, pady=10)
dataset_var = tk.StringVar()
dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var)
dataset_dropdown.grid(column=1, row=6, padx=10, pady=10)

update_dropdown()

# Buttons
train_button = ttk.Button(root, text="Build & Train Model", command=build_model)
train_button.grid(column=0, row=7, padx=10, pady=20, columnspan=2)

save_button = ttk.Button(root, text="Save Model", command=save_model)
save_button.grid(column=0, row=8, padx=10, pady=10, columnspan=2)

result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=9, columnspan=2, pady=10)

root.mainloop()