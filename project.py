import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _(mo):
    mo.md(r"""# Project: Sign Language Recognition""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Importing libraries""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import os
    import re
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101V2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    import pandas as pd
    import time
    import matplotlib.pyplot as plt
    import splitfolders
    import json
    return (
        Conv2D,
        Dense,
        EarlyStopping,
        Flatten,
        GlobalAveragePooling2D,
        Input,
        LSTM,
        MaxPooling2D,
        Model,
        ResNet101V2,
        ResNet50,
        ResNet50V2,
        Reshape,
        json,
        mnist,
        np,
        os,
        pd,
        plt,
        re,
        splitfolders,
        tf,
        time,
        to_categorical,
    )


@app.cell
def _(mo):
    mo.md(r"""### Loading the ASL dataset and splitting into train and test""")
    return


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset")

    print("Path to dataset files:", path)
    return kagglehub, path


@app.cell
def _(os, path):
    os.chdir(path)
    return


@app.cell
def _(path, pd):
    from pathlib import Path

    # Path where our data is located
    base_path = Path(path) / "ASL_Alphabet_Dataset" / "asl_alphabet_train"

    # Dictionary to save our 29 classes
    categories = {  0: "A",
                    1: "B",
                    2: "C",
                    3: "D",
                    4: "E",
                    5: "F",
                    6: "G",
                    7: "H",
                    8: "I",
                    9: "G",
                    10: "K",
                    11: "L",
                    12: "M",
                    13: "N",
                    14: "O",
                    15: "P",
                    16: "Q",
                    17: "R",
                    18: "S",
                    19: "T",
                    20: "U",
                    21: "V",
                    22: "W",
                    23: "X",
                    24: "Y",
                    25: "Z",
                    26: "del",
                    27: "nothing",
                    28: "space",
                }

    def add_class_name_prefix(df, col_name):
        df[col_name] = df[col_name].apply(lambda x: f"class_{x}")
        return df

    # List containing all the filenames in the dataset
    filenames_list = []
    # List to store the corresponding category; note that each folder of the dataset has one class of data
    categories_list = []

    for category_id, category_name in categories.items():
        category_path = base_path / category_name
        if category_path.exists() and category_path.is_dir():
            filenames = [file.name for file in category_path.iterdir() if file.is_file()]
            filenames_list.extend(filenames)
            categories_list.extend([category_id] * len(filenames))

    df = pd.DataFrame({"filename": filenames_list, "category": categories_list})
    df = add_class_name_prefix(df, "filename")

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return (
        Path,
        add_class_name_prefix,
        base_path,
        categories,
        categories_list,
        category_id,
        category_name,
        category_path,
        df,
        filenames,
        filenames_list,
    )


@app.cell
def _(Path, base_path, splitfolders):
    output_path = Path('./asl_dataset_split')
    if not output_path.exists():
        splitfolders.ratio(base_path, output=output_path, seed=1333, ratio=(0.8, 0.1, 0.1))
    return (output_path,)


@app.cell
def _(output_path):
    # Define explicit paths for train, val, and test
    train_path = output_path / 'train'
    val_path = output_path / 'val'
    test_path = output_path / 'test'

    print(f"Train data saved at: {train_path}")
    print(f"Validation data saved at: {val_path}")
    print(f"Test data saved at: {test_path}")
    return test_path, train_path, val_path


@app.cell
def _(test_path, train_path):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Parameters
    image_size = 64  # resizing images to 64x64
    batch_size = 32  

    # data augmentation and normalization
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # normalize pixel values to [0, 1]
        validation_split=0.2  # split data into training and validation
    )

    # training data
    train_data = datagen.flow_from_directory(
        directory=train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # validation data
    val_data = datagen.flow_from_directory(
        directory=train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # test data
    test_data = datagen.flow_from_directory(
        directory=test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Verifying shapes
    print(f"Number of training samples: {train_data.samples}")
    print(f"Number of validation samples: {val_data.samples}")
    print(f"Number of test samples: {test_data.samples}")
    return (
        ImageDataGenerator,
        batch_size,
        datagen,
        image_size,
        test_data,
        train_data,
        val_data,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load pretrained ResNet-50, building rnn, cnn and hybrid model""")
    return


@app.cell
def _(ResNet50):
    # loading pretrained ResNet-50

    def load_resnet_base(input_shape=(64, 64, 3)):
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        return base_model

    resnet_base_model = load_resnet_base()
    return load_resnet_base, resnet_base_model


@app.cell
def _(
    Dense,
    GlobalAveragePooling2D,
    Input,
    LSTM,
    Model,
    Reshape,
    resnet_base_model,
):
    def resnet_with_rnn(base_model, rnn_layers=2):
        base_model.trainable = False # we are freezing the base_model
        inputs = Input(shape=(64, 64, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, -1))(x)
        for _ in range(rnn_layers):
            x = LSTM(64, return_sequences=True)(x)
        x = LSTM(64)(x)
        outputs = Dense(29, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model 

    resnet_rnn_model = resnet_with_rnn(resnet_base_model)
    return resnet_rnn_model, resnet_with_rnn


@app.cell
def _(mo):
    mo.md(r"""## Building the CNN model""")
    return


@app.cell
def _(
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Model,
    resnet_base_model,
):
    def resnet_with_cnn(base_model, cnn_layers=2):
        base_model.trainable = False # we are freezing the base_model
        inputs = Input(shape=(64, 64, 3))
        x = base_model(inputs, training=False)
        for _ in range(cnn_layers - 1):
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(29, activation='softmax')(x)  # Updated for 29 classes
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    resnet_cnn_model = resnet_with_cnn(resnet_base_model)
    return resnet_cnn_model, resnet_with_cnn


@app.cell
def _(mo):
    mo.md(r"""## Training the models""")
    return


@app.cell
def _(
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    LSTM,
    MaxPooling2D,
    Model,
    Reshape,
    resnet_base_model,
):
    def hybrid_resnet_cnn_rnn(base_model):
        base_model.trainable = False
        inputs = Input(shape=(64, 64, 3))
        x = base_model(inputs, training=False)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, -1))(x)
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(64)(x)
        outputs = Dense(29, activation='softmax')(x)  # Updated for 29 classes
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    hybrid_model = hybrid_resnet_cnn_rnn(resnet_base_model)
    return hybrid_model, hybrid_resnet_cnn_rnn


@app.cell
def _(mo):
    mo.md(r"""## Evaluating the models""")
    return


@app.cell
def _(EarlyStopping, Path, json, mo, plt, time):
    from types import SimpleNamespace

    def evaluate_model(model, train_data, val_data, test_data, model_name, results, model_filename):
        weights_dir = Path(mo.notebook_dir()) / "weights"
        # Define file paths based on model_name
        weights_file = weights_dir / f"{model_filename}.weights.h5"
        history_file = weights_dir / f"{model_filename}_history.json"

        # Check if both weights and history files exist
        if weights_file.exists() and history_file.exists():
            print(f"Weights file found: {weights_file}. Loading weights...")
            model.load_weights(weights_file)

            print(f"History file found: {history_file}. Loading history...")
            
            with history_file.open('r') as f:
                history = SimpleNamespace(history = json.load(f))

            training_time = "-"
        else:

            # Early stopping callback to avoid overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
            # Train the model
            start_time = time.time()
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=30,
                callbacks=[early_stopping],
                verbose=1
            )
            training_time = time.time() - start_time

        # Evaluate on test data
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)

        # Log results
        results.append({
            'Model': model_name,
            'Validation Accuracy': max(history.history['val_accuracy']),
            'Test Accuracy': test_accuracy,
            'Training Time (s)': training_time
        })

        # Plot training history
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Training/Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

        return history
    return SimpleNamespace, evaluate_model


@app.cell
def _(mo):
    mo.md(r"""### Resnet50""")
    return


@app.cell
def _(
    evaluate_model,
    hybrid_model,
    pd,
    resnet_cnn_model,
    resnet_rnn_model,
    test_data,
    train_data,
    val_data,
):
    results = []

    # evaluate all models
    history_resnet_rnn = evaluate_model(resnet_rnn_model, train_data, val_data, test_data, "ResNet50 + RNN", results, "r50_rnn")
    history_resnet_cnn = evaluate_model(resnet_cnn_model, train_data, val_data, test_data, "ResNet50 + CNN", results, "r50_cnn")
    history_hybrid = evaluate_model(hybrid_model, train_data, val_data, test_data, "Hybrid ResNet50 (CNN + RNN)", results, "r50_hybrid")

    # convert results to DataFrame
    results_df = pd.DataFrame(results)

    # display the results
    results_df
    return (
        history_hybrid,
        history_resnet_cnn,
        history_resnet_rnn,
        results,
        results_df,
    )


@app.cell
def _(mo):
    mo.md(r"""### Resnetv2""")
    return


@app.cell
def _(ResNet50V2, hybrid_resnet_cnn_rnn, resnet_with_cnn, resnet_with_rnn):
    v2_base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

    v2_rnn_model = resnet_with_rnn(v2_base_model)
    v2_cnn_model = resnet_with_cnn(v2_base_model)
    v2_hybrid = hybrid_resnet_cnn_rnn(v2_base_model)
    return v2_base_model, v2_cnn_model, v2_hybrid, v2_rnn_model


@app.cell
def _(
    evaluate_model,
    pd,
    test_data,
    train_data,
    v2_cnn_model,
    v2_hybrid,
    v2_rnn_model,
    val_data,
):
    v2_results = []

    # evaluate all models
    history_r50v2_rnn = evaluate_model(v2_rnn_model, train_data, val_data, test_data, "ResNet50v2 + RNN", v2_results, "r50v2_rnn")
    history_r50v2_cnn = evaluate_model(v2_cnn_model, train_data, val_data, test_data, "ResNet50v2 + CNN", v2_results, "r50v2_cnn")
    history_r50v2_hybrid = evaluate_model(v2_hybrid, train_data, val_data, test_data, "Hybrid ResNet50v2 (CNN + RNN)", v2_results, "r50v2_hybrid")

    # convert results to DataFrame
    v2_results_df = pd.DataFrame(v2_results)

    # display the results
    v2_results_df
    return (
        history_r50v2_cnn,
        history_r50v2_hybrid,
        history_r50v2_rnn,
        v2_results,
        v2_results_df,
    )


@app.cell
def _(mo):
    mo.md("""### Resnet101v2""")
    return


@app.cell
def _(
    ResNet101V2,
    hybrid_resnet_cnn_rnn,
    resnet_with_cnn,
    resnet_with_rnn,
):
    r101_base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=(64, 64, 3))

    r101_rnn_model = resnet_with_rnn(r101_base_model)
    r101_cnn_model = resnet_with_cnn(r101_base_model)
    r101_hybrid = hybrid_resnet_cnn_rnn(r101_base_model)
    return r101_base_model, r101_cnn_model, r101_hybrid, r101_rnn_model


@app.cell
def _(
    evaluate_model,
    pd,
    r101_cnn_model,
    r101_hybrid,
    r101_rnn_model,
    test_data,
    train_data,
    val_data,
):
    r101_results = []

    # evaluate all models
    history_r101v2_rnn = evaluate_model(r101_rnn_model, train_data, val_data, test_data, "ResNet101v2 + RNN", r101_results, "r101v2_rnn")
    history_r101v2_cnn = evaluate_model(r101_cnn_model, train_data, val_data, test_data, "ResNet101v2 + CNN", r101_results, "r101v2_cnn")
    history_r101v2_hybrid = evaluate_model(r101_hybrid, train_data, val_data, test_data, "Hybrid ResNet101v2 (CNN + RNN)", r101_results, "r101v2_hybrid")

    # convert results to DataFrame
    r101_results_df = pd.DataFrame(r101_results)

    # display the results
    r101_results_df
    return (
        history_r101v2_cnn,
        history_r101v2_hybrid,
        history_r101v2_rnn,
        r101_results,
        r101_results_df,
    )


@app.cell
def _(mo):
    mo.md(r"""## Saving Select Models""")
    return


@app.cell
def _(
    Path,
    history_hybrid,
    history_r101v2_cnn,
    history_r101v2_hybrid,
    history_r101v2_rnn,
    history_r50v2_cnn,
    history_r50v2_hybrid,
    history_r50v2_rnn,
    history_resnet_cnn,
    history_resnet_rnn,
    hybrid_model,
    json,
    mo,
    r101_cnn_model,
    r101_hybrid,
    r101_rnn_model,
    resnet_cnn_model,
    resnet_rnn_model,
    v2_cnn_model,
    v2_hybrid,
    v2_rnn_model,
):
    weights_dir = Path(mo.notebook_dir()) / "weights"
    weights_dir.mkdir(exist_ok=True)

    resnet_rnn_model.save_weights(weights_dir / "r50_rnn.weights.h5")
    resnet_cnn_model.save_weights(weights_dir / "r50_cnn.weights.h5")
    hybrid_model.save_weights(weights_dir / "r50_hybrid.weights.h5")

    json.dump(history_resnet_rnn.history, open(weights_dir / "r50_rnn_history.json", 'w' ))
    json.dump(history_resnet_cnn.history, open(weights_dir / "r50_cnn_history.json", 'w' ))
    json.dump(history_hybrid.history, open(weights_dir / "r50_hybrid_history.json", 'w' ))

    v2_rnn_model.save_weights(weights_dir / "r50v2_rnn.weights.h5")
    v2_cnn_model.save_weights(weights_dir / "r50v2_cnn.weights.h5")
    v2_hybrid.save_weights(weights_dir / "r50v2_hybrid.weights.h5")

    json.dump(history_r50v2_rnn.history, open(weights_dir / "r50v2_rnn_history.json", 'w' ))
    json.dump(history_r50v2_cnn.history, open(weights_dir / "r50v2_cnn_history.json", 'w' ))
    json.dump(history_r50v2_hybrid.history, open(weights_dir / "r50v2_hybrid_history.json", 'w' ))

    r101_rnn_model.save_weights(weights_dir / "r101v2_rnn.weights.h5")
    r101_cnn_model.save_weights(weights_dir / "r101v2_cnn.weights.h5")
    r101_hybrid.save_weights(weights_dir / "r101v2_hybrid.weights.h5")

    json.dump(history_r101v2_rnn.history, open(weights_dir / "r101v2_rnn_history.json", 'w' ))
    json.dump(history_r101v2_cnn.history, open(weights_dir / "r101v2_cnn_history.json", 'w' ))
    json.dump(history_r101v2_hybrid.history, open(weights_dir / "r101v2_hybrid_history.json", 'w' ))
    return (weights_dir,)


@app.cell
def _(mo):
    mo.md("""### Loading Weights Example""")
    return


@app.cell
def _(r101_base_model, resnet_with_cnn, test_data, weights_dir):
    # Create a new model instance
    test_model = resnet_with_cnn(r101_base_model)

    # Restore the weights
    test_model.load_weights(weights_dir / "r101v2_cnn.weights.h5")

    print(test_model.evaluate(test_data, verbose=0))
    return (test_model,)


if __name__ == "__main__":
    app.run()
