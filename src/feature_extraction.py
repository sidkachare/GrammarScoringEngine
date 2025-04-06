import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_features(self, df):
        features_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio features"):
            audio_path = row["audio_path"]
            y, sr = librosa.load(audio_path, sr=self.sr)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = mfcc.mean(axis=1)

            zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
            rmse = librosa.feature.rms(y=y)[0].mean()
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = pitches[magnitudes > np.median(magnitudes)].mean() if np.any(magnitudes > 0) else 0.0

            features = {f"mfcc_{i+1}": val for i, val in enumerate(mfcc_mean)}
            features["zcr"] = zcr
            features["rmse"] = rmse
            features["spec_centroid"] = spec_centroid
            features["pitch"] = pitch
            features["audio_path"] = audio_path

            if "label" in df.columns:
                features["score"] = row["label"]

            features_list.append(features)

        return pd.DataFrame(features_list)


