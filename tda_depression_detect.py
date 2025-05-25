# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.subsampling import choose_n_farthest_points
from ripser import ripser
from persim import PersistenceImager
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# import tensorflow as tf
# from tensorflow.keras import layers, models
# %%
# ========================================================================= #
# Step 1: Process all audio files and create train/test point clouds
# ========================================================================= #

df_splits = pd.read_csv('./df.csv')

def get_all_audio_files():
    audio_files = []
    
    # Check audio directory
    audio_dir = './audio'
    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(audio_dir, f))
    
    return audio_files

def extract_participant_id(filename):
    basename = os.path.basename(filename)
    # Remove _AUDIO.wav or .wav extension and get the participant ID
    if '_AUDIO.wav' in basename:
        return int(basename.replace('_AUDIO.wav', ''))
    elif '.wav' in basename:
        return int(basename.replace('.wav', ''))
    else:
        raise ValueError(f"Unexpected filename format: {basename}")

# Get all audio files
all_audio_files = get_all_audio_files()
print(f"Found {len(all_audio_files)} audio files")

# Process each audio file
point_clouds = {}
labels = {}
participant_splits = {}

def load_audio(audio_file, sr_target=16000, max_duration=None):    
    if max_duration is None:
        y, sr = librosa.load(audio_file, sr=sr_target)
    else:
        y, sr = librosa.load(audio_file, sr=sr_target, duration=max_duration)
    return y, sr

def create_point_cloud_from_audio(audio, window_size=1024, delay=1, hop_size=512):    
    point_cloud = sliding_window_view(audio, window_shape=window_size)[::hop_size]

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    point_cloud = pca.fit_transform(point_cloud)
        
    # Subsampling (for computational efficiency)
    point_cloud = choose_n_farthest_points(points=point_cloud, nb_points=4000)

    return point_cloud

print("\nProcessing audio files...")
for i, audio_file in enumerate(all_audio_files):
    try:
        participant_id = extract_participant_id(audio_file)
        
        # Get split information for this participant
        participant_row = df_splits[df_splits['Participant_ID'] == participant_id]
        if len(participant_row) == 0:
            print(f"Warning: No split information found for participant {participant_id}")
            continue
            
        split = participant_row['Split'].iloc[0]
        phq8_binary = participant_row['PHQ8_Binary'].iloc[0]  # Depression label
        
        print(f"Processing {audio_file} (Participant {participant_id}, Split: {split})")
        
        # Load audio
        audio, sr = load_audio(audio_file)  # Load full audio file
        
        # Create point cloud
        point_cloud = create_point_cloud_from_audio(audio)
        
        # Store results
        point_clouds[participant_id] = point_cloud
        labels[participant_id] = phq8_binary
        participant_splits[participant_id] = split
        
        # print(f"Point cloud shape: {point_cloud.shape}")
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        continue

print(f"\nSuccessfully processed {len(point_clouds)} audio files")

train_point_clouds = {}
test_point_clouds = {}
dev_point_clouds = {}

train_labels = {}
test_labels = {}
dev_labels = {}

for participant_id, point_cloud in point_clouds.items():
    split = participant_splits[participant_id]
    label = labels[participant_id]
    
    if split == 'train':
        train_point_clouds[participant_id] = point_cloud
        train_labels[participant_id] = label
    elif split == 'test':
        test_point_clouds[participant_id] = point_cloud
        test_labels[participant_id] = label
    elif split == 'dev':
        dev_point_clouds[participant_id] = point_cloud
        dev_labels[participant_id] = label

print(f"Train set: {len(train_point_clouds)} samples")
print(f"Test set: {len(test_point_clouds)} samples") 
print(f"Dev set: {len(dev_point_clouds)} samples")
# %%
# ========================================================================= #
# Step 2: Compute Persistence Diagrams for Train/Test Point Clouds
# ========================================================================= #

def compute_persistence_diagrams(point_clouds_dict):
    diagrams_0d = []
    diagrams_1d = []
    for participant_id, pc in point_clouds_dict.items():
        result = ripser(np.array(pc), maxdim=1)
        dgms = result['dgms']
        diagrams_0d.append(dgms[0])
        diagrams_1d.append(dgms[1])
    return diagrams_0d, diagrams_1d

print("\nComputing persistence diagrams for train set...")
train_diagrams_0d, train_diagrams_1d = compute_persistence_diagrams(train_point_clouds)
print(f"Computed {len(train_diagrams_0d)} train 0D diagrams and {len(train_diagrams_1d)} train 1D diagrams.")

print("\nComputing persistence diagrams for test set...")
test_diagrams_0d, test_diagrams_1d = compute_persistence_diagrams(test_point_clouds)
print(f"Computed {len(test_diagrams_0d)} test 0D diagrams and {len(test_diagrams_1d)} test 1D diagrams.")

# Normalize persistence diagrams per diagram (births and deaths to [0, 1])
def normalize_diagrams_per_diagram(diagrams):
    """
    Normalize each persistence diagram independently so that births and deaths are in (0, 1) (strictly between 0 and 1).
    For each diagram, divide all births by the maximum birth and all deaths by the maximum death in that diagram,
    then scale to (0, 1) using a small epsilon margin.
    """
    eps = 1e-6
    norm_diagrams = []
    for dgm in diagrams:
        if len(dgm) == 0:
            norm_diagrams.append(dgm)
            continue
        max_birth = np.max(dgm[:, 0]) if np.max(dgm[:, 0]) > 0 else 1.0
        max_death = np.max(dgm[:, 1]) if np.max(dgm[:, 1]) > 0 else 1.0
        births_norm = dgm[:, 0] / max_birth
        deaths_norm = dgm[:, 1] / max_death
        # Scale to (0, 1) by avoiding exact 0 and 1
        births_norm = births_norm * (1 - 2 * eps) + eps
        deaths_norm = deaths_norm * (1 - 2 * eps) + eps
        norm_dgm = np.stack([births_norm, deaths_norm], axis=1)
        norm_diagrams.append(norm_dgm)
    return norm_diagrams

# Normalize persistence diagrams per diagram (births and deaths to [0, 1])
train_diagrams_0d = normalize_diagrams_per_diagram(train_diagrams_0d)
train_diagrams_1d = normalize_diagrams_per_diagram(train_diagrams_1d)
test_diagrams_0d = normalize_diagrams_per_diagram(test_diagrams_0d)
test_diagrams_1d = normalize_diagrams_per_diagram(test_diagrams_1d)

# %%
# ========================================================================= #
# Step 3: Convert Persistence Diagrams to Persistence Images (Parallelized)
# ========================================================================= #

# Configure the imager (adjust ranges and pixel_size as needed for your data)
pimgr = PersistenceImager(
    birth_range=(0.0, 1.0),
    pers_range=(0.0, 1.0),
    pixel_size=0.05,  # adjust as needed
    weight='persistence',
    kernel='gaussian'
)

# Fit the imager to all diagrams (train + test) for consistent scaling
all_diagrams = train_diagrams_0d + train_diagrams_1d + test_diagrams_0d + test_diagrams_1d
pimgr.fit(all_diagrams)

# Transform diagrams in parallel
print("\nTransforming train 0D diagrams to persistence images...")
train_images_0d = pimgr.transform(train_diagrams_0d, n_jobs=-1)
print(f"Transformed {len(train_images_0d)} train 0D diagrams.")

print("\nTransforming train 1D diagrams to persistence images...")
train_images_1d = pimgr.transform(train_diagrams_1d, n_jobs=-1)
print(f"Transformed {len(train_images_1d)} train 1D diagrams.")

print("\nTransforming test 0D diagrams to persistence images...")
test_images_0d = pimgr.transform(test_diagrams_0d, n_jobs=-1)
print(f"Transformed {len(test_images_0d)} test 0D diagrams.")

print("\nTransforming test 1D diagrams to persistence images...")
test_images_1d = pimgr.transform(test_diagrams_1d, n_jobs=-1)
print(f"Transformed {len(test_images_1d)} test 1D diagrams.")

# %%
# ========================================================================= #
# Step 4: Train Random Forest Classifier on Persistence Images
# ========================================================================= #

# Combine 0D and 1D persistence images for classifier input
def combine_images(images_0d, images_1d):
    arr_0d = np.array([img.flatten() for img in images_0d])
    arr_1d = np.array([img.flatten() for img in images_1d])
    return np.concatenate([arr_0d, arr_1d], axis=1)

X_train = combine_images(train_images_0d, train_images_1d)
X_test = combine_images(test_images_0d, test_images_1d)

y_train = np.array([train_labels[pid] for pid in train_point_clouds.keys()])
y_test = np.array([test_labels[pid] for pid in test_point_clouds.keys()])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nRandom Forest Classification Results (1D Persistence Images):")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# %%
# ========================================================================= #
# Step 5: Train CNN Classifier on Persistence Images
# ========================================================================= #

# CNN expects 4D input: (n_samples, height, width, channels)
# We'll concatenate 0D and 1D images along the channel axis
# def stack_images_for_cnn(images_0d, images_1d):
#     arr_0d = np.array(images_0d)
#     arr_1d = np.array(images_1d)
#     if arr_0d.ndim == 3:
#         arr_0d = arr_0d[..., np.newaxis]
#     if arr_1d.ndim == 3:
#         arr_1d = arr_1d[..., np.newaxis]
#     # Stack along channel axis
#     return np.concatenate([arr_0d, arr_1d], axis=-1)

# X_train_cnn = stack_images_for_cnn(train_images_0d, train_images_1d)
# X_test_cnn = stack_images_for_cnn(test_images_0d, test_images_1d)

# # Use the same labels as before
# y_train_cnn = y_train
# y_test_cnn = y_test

# # CNN model definition
# input_shape = X_train_cnn.shape[1:]
# cnn_model = models.Sequential([
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# print("\nTraining CNN on 1D persistence images...")
# cnn_model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=16, validation_split=0.1, verbose=2)

# print("\nEvaluating CNN on test set...")
# y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()

# print("\nCNN Classification Results (1D Persistence Images):")
# print(classification_report(y_test_cnn, y_pred_cnn))
# print(f"Accuracy: {accuracy_score(y_test_cnn, y_pred_cnn):.4f}")