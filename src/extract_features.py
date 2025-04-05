import pandas as pd
import numpy as np
import re
from pathlib import Path

import numpy as np
import pandas as pd

def extract_silence_features(df: pd.DataFrame) -> dict:
    """
    Extracts Participant-initiated silences (only between their own utterances)
    and reaction times after Ellie speaks.
    """
    df = df.sort_values('start_time').reset_index(drop=True)
    total_time = df['stop_time'].max() - df['start_time'].min()

    participant_silences = []
    reaction_times = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Participant → silence → Participant (Participant-initiated silence)
        if prev['speaker'] == 'Participant' and curr['speaker'] == 'Participant' and curr['start_time'] > prev['stop_time']:
            participant_silences.append(curr['start_time'] - prev['stop_time'])

        # Ellie → silence → Participant (Reaction time)
        if prev['speaker'] == 'Ellie' and curr['speaker'] == 'Participant' and curr['start_time'] > prev['stop_time']:
            reaction_times.append(curr['start_time'] - prev['stop_time'])

    features = {}

    # Participant initiated silence features
    if participant_silences:
        features["Total_Silence_Duration"] = sum(participant_silences)
        features["Avg_Silence_Duration"] = np.mean(participant_silences)
        features["Max_Silence_Duration"] = np.max(participant_silences)
        features["Std_Silence_Duration"] = np.std(participant_silences)
        features["Silence_Duration_Ratio"] = sum(participant_silences) / total_time
    else:
        features.update({
            "Total_Silence_Duration": 0,
            "Avg_Silence_Duration": 0,
            "Max_Silence_Duration": 0,
            "Std_Silence_Duration": 0,
            "Silence_Duration_Ratio": 0
        })

    # Reaction time features
    if reaction_times:
        features["Avg_Reaction_Time"] = np.mean(reaction_times)
        features["Max_Reaction_Time"] = np.max(reaction_times)
        features["Std_Reaction_Time"] = np.std(reaction_times)
        features["Long_Reaction_Times"] = sum(rt > 3.0 for rt in reaction_times)
    else:
        features.update({
            "Avg_Reaction_Time": 0,
            "Max_Reaction_Time": 0,
            "Std_Reaction_Time": 0,
            "Long_Reaction_Times": 0
        })

    return features

def extract_speech_rate_features(df: pd.DataFrame) -> dict:
    """
    Function to extract speech rate-related features.
    
    Args:
        df: DataFrame containing conversation data.
        
    Returns:
        dict: Speech rate-related features.
    """
    participant_df = df[df['speaker'] == 'Participant'].copy()
    participant_df['word_count'] = participant_df['value'].astype(str).apply(lambda x: len(x.split()))
    participant_df['speech_rate'] = participant_df['word_count'] / participant_df['duration']

    speech_rates = participant_df['speech_rate'].dropna()
    
    features = {
        "Avg_Speech_Rate": speech_rates.mean(),
        "Std_Speech_Rate": speech_rates.std(),
        "Max_Speech_Rate": speech_rates.max(),
        "Min_Speech_Rate": speech_rates.min()
    }
        
    # Calculate speech rate for different utterance lengths
    if len(participant_df) > 0:
        
        short_utterances = participant_df[participant_df['word_count'] <= 5]
        features["Short_Utterance_Speech_Rate"] = short_utterances['speech_rate'].mean() if len(short_utterances) > 0 else 0
        
        long_utterances = participant_df[participant_df['word_count'] > 15]
        features["Long_Utterance_Speech_Rate"] = long_utterances['speech_rate'].mean() if len(long_utterances) > 0 else 0
    
    return features

def extract_filler_features(df: pd.DataFrame) -> dict:
    """
    Function to extract filler-related features
    
    Args:
        df: DataFrame containing conversation data
        
    Returns:
        dict: Filler-related features
    """
    participant_df = df[df['speaker'] == 'Participant'].copy()
    
    fillers = ['um', 'uh', 'mm', 'hmm']
    emotion_cues = ['<sigh>', '<yawn>', '<laughter>', '<sniffle>', '<clears throat>']
    
    participant_df['num_fillers'] = participant_df['value'].astype(str).apply(
        lambda x: sum(len(re.findall(r'\b' + re.escape(f) + r'\b', x.lower())) for f in fillers)
    )
    
    participant_df['num_emotion_cues'] = participant_df['value'].astype(str).apply(
        lambda x: sum(x.lower().count(e.lower()) for e in emotion_cues)
    )
    
    participant_df['word_count'] = participant_df['value'].astype(str).apply(lambda x: len(x.split()))
    total_words = participant_df['word_count'].sum()
    
    features = {
        "Total_Filler_Count": participant_df['num_fillers'].sum(),
        "Avg_Fillers_per_Utterance": participant_df['num_fillers'].mean(),
        "Total_Emotion_Cue_Count": participant_df['num_emotion_cues'].sum(),
        "Filler_to_Word_Ratio": participant_df['num_fillers'].sum() / total_words if total_words > 0 else 0,
    }
    
    for e in emotion_cues:
        e_name = e.strip('<>')
        count = participant_df['value'].astype(str).str.count(re.escape(e)).sum()
        features[f'Emotion_{e_name}'] = count
    
    return features

def extract_turn_features(df: pd.DataFrame) -> dict:
    speakers = df['speaker'].values
    total_turns = sum(speakers[i] != speakers[i + 1] for i in range(len(speakers) - 1))
    ellie_to_participant = sum((speakers[i] == 'Ellie' and speakers[i + 1] == 'Participant') for i in range(len(speakers) - 1))

    return {
        "Num_Turns": ellie_to_participant,
        "Total_Turns": total_turns
    }

def extract_all_features(df: pd.DataFrame) -> dict:
    df = df.sort_values('start_time')
    df['duration'] = df['stop_time'] - df['start_time']
    df['word_count'] = df['value'].astype(str).apply(lambda x: len(x.split()))
    
    features = {
        "Num_Utterances_Ellie": (df['speaker'] == 'Ellie').sum(),
        "Num_Utterances_Participant": (df['speaker'] == 'Participant').sum(),
        "Total_Duration_Ellie": df.loc[df['speaker'] == 'Ellie', 'duration'].sum(),
        "Total_Duration_Participant": df.loc[df['speaker'] == 'Participant', 'duration'].sum(),
        "Avg_Utterance_Duration_Ellie": df.loc[df['speaker'] == 'Ellie', 'duration'].mean(),
        "Avg_Utterance_Duration_Participant": df.loc[df['speaker'] == 'Participant', 'duration'].mean(),
        "Total_Interview_Duration": df['stop_time'].max() - df['start_time'].min(),
        "Total_Word_Count_Ellie": df.loc[df['speaker'] == 'Ellie', 'word_count'].sum(),
        "Total_Word_Count_Participant": df.loc[df['speaker'] == 'Participant', 'word_count'].sum(),
        "Avg_Word_Count_Ellie": df.loc[df['speaker'] == 'Ellie', 'word_count'].mean(),
        "Avg_Word_Count_Participant": df.loc[df['speaker'] == 'Participant', 'word_count'].mean(),
    }
    
    features.update(extract_turn_features(df))
    features.update(extract_silence_features(df))
    features.update(extract_speech_rate_features(df))
    features.update(extract_filler_features(df))
    
    return features

def extract_utterance_features(transcript_dir: str) -> pd.DataFrame:
    transcript_path = Path(transcript_dir)
    transcript_files = sorted(transcript_path.glob('*_TRANSCRIPT.csv'))
    features = []

    for file_path in transcript_files:
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=["start_time", "stop_time", "speaker", "value"]).iloc[1:]
            df["start_time"] = df["start_time"].astype(float)
            df["stop_time"] = df["stop_time"].astype(float)
            
            participant_id = int(file_path.stem.split("_")[0])
            summary = extract_all_features(df)
            summary = {"Participant_ID": participant_id, **summary}
            
            features.append(summary)

        except Exception as e:
            print(f"[!] Error processing {file_path.name}: {e}")

    return pd.DataFrame(features)