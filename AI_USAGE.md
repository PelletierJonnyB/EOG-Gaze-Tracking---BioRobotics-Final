# AI USAGE

For each significant use, include:
• The tool used and the prompt/request
• What was generated
• Modifications you made
• What you learned
Include a reflection section: How did AI tools help your productivity? What were they not helpful
for? How did you verify AI-generated code was correct?

Use 1:
Tool: ChatGPT
Prompt: 
> Can you generate a sample code in python for plotting detected blinks if given these inputs:
> veog = df['veog'].values
> time = np.arange(len(veog)) / sample_rate
> blinks, processed = detect_blinks(veog, sample_rate=sample_rate)"

What was generated:

>import numpy as np
>import matplotlib.pyplot as plt
>
>#Inputs you already have
>veog = df['veog'].values
>time = np.arange(len(veog)) / sample_rate
>blinks, processed = detect_blinks(veog, sample_rate=sample_rate)
>
>#Create figure
>plt.figure(figsize=(12, 6))
>
>#Plot raw VEOG signal
>plt.plot(time, veog, label='Raw VEOG', alpha=0.5)
>
>#Plot processed signal (if available)
>if processed is not None:
>    plt.plot(time, processed, label='Processed VEOG', linewidth=2)
>
>#Plot blink markers
>if blinks is not None and len(blinks) > 0:
>    blink_times = time[blinks]
>    blink_values = veog[blinks]
>   
>   plt.scatter(blink_times, blink_values, 
>               color='red', label='Detected Blinks', zorder=5)
>
>#Labels and styling
>plt.xlabel('Time (seconds)')
>plt.ylabel('VEOG Signal')
>plt.title('Blink Detection from VEOG Signal')
>plt.legend()
>plt.grid(True)
>
>plt.tight_layout()
>plt.show()"

Modifications made: turned it into this function:

>def plot_blink_detection(df, sample_rate=SAMPLE_RATE):
>   veog = df['veog'].values
>    time = np.arange(len(veog)) / sample_rate
>    blinks, processed = detect_blinks(veog, sample_rate=sample_rate)
>    
>   fig, ax = plt.subplots(figsize=(14, 4))
>
>   ax.plot(time, veog, 'b-', linewidth=0.6, alpha=0.5, label='Raw VEOG')
>   ax.plot(time, processed, 'k-', linewidth=1.0, label='Processed VEOG')
> 
>  first = True
>   for start, end, peak in blinks:
>       ax.axvspan(start/sample_rate, end/sample_rate,
>                  color='red', alpha=0.3,
>                  label='Blink' if first else "")
>       ax.plot(peak/sample_rate, processed[peak], 'r^', markersize=8)
>       first = False
> 
>  ax.set_xlabel('Time (s)')
>   ax.set_ylabel('Amplitude (uV)')
>   ax.set_title(f'Blink Detection - Found {len(blinks)} blink(s)')
>   ax.legend()
> 
>  plt.tight_layout()
>   plt.show()
> 
>  return blinks"

  What I learned:
> I learned to better plot data using the plotting tools available in our libraries. Also borrowed code from lab 2 for this.

Use 2:
Tool: ChatGPT
Prompt: 
>can you fix modify this dataframe to be more printable? There is min, max, and peak-to-peak
>def extract_features(df, sample_rate=SAMPLE_RATE):
>   features = {}
>
>   for ch_name in ['heog', 'veog']:
>       if ch_name not in df.columns:
>           continue
>       raw = df[ch_name].values
>       processed = process_eog_signal(raw, sample_rate=sample_rate)
>       prefix = ch_name.upper()

What was generated:

>import numpy as np
>import pandas as pd
>
>def extract_features(df, sample_rate=SAMPLE_RATE):
>   rows = []
>
>    for ch_name in ['heog', 'veog']:
>        if ch_name not in df.columns:
>            continue
>      raw = df[ch_name].values
>      processed = process_eog_signal(raw, sample_rate=sample_rate)
>
>    row = {
>    "channel": ch_name.upper(),
>
>    #Raw signal stats
>    "raw_min": np.min(raw),
>    "raw_max": np.max(raw),
>    "raw_ptp": np.ptp(raw),  # peak-to-peak
>
>    #Processed signal stats
>    "proc_min": np.min(processed),
>    "proc_max": np.max(processed),
>    "proc_ptp": np.ptp(processed),
>  }
>
>    rows.append(row)
>
>    features_df = pd.DataFrame(rows)
>
>    return features_df

Modifications made: turned it into this function:

>def extract_features(df, sample_rate=SAMPLE_RATE):
>    features = {}
>
>    for ch_name in ['heog', 'veog']:
>        if ch_name not in df.columns:
>            continue
>
>    raw = df[ch_name].values
>    processed = process_eog_signal(raw, sample_rate=sample_rate)
>
>    prefix = ch_name.upper()
>
>    #Used AI to help get the data frame right here for printing and displaying
>    features[f'{prefix}_min'] = np.min(processed)
>    features[f'{prefix}_max'] = np.max(processed)
>    features[f'{prefix}_ptp'] = np.ptp(processed)
>
>    return features

  What I learned:
> I learned how to do the [f'{prefix}_min'] trick to get an adaptive name from both the AI and scouring the dataframes documentation.

Use 3:
Tool: ChatGPT
Prompt: 

>My graphs are empty. Can you fix that?
>MidH = DISTS[0]/2
>MidH_v = (features['HEOG_max'] + features['HEOG_min'])/2
>MidV_v = (features['VEOG_max'] + features['VEOG_min'])/2
>#print(MidH_v)
>#print(MidV_v)
>#computing calibrations and using casting since numpy functions work in radians 
>H_FOVp = np.degrees(np.arctan(MidH/DISTS[2]))
>H_FOVn = -np.degrees(np.arctan(MidH/DISTS[2]))
>#print(H_FOVn)
>V_FOVp = np.degrees(np.arctan(MidV/DISTS[2]))
>V_FOVn = -np.degrees(np.arctan(MidV/DISTS[2]))
>#print(V_FOVn)
>for ch_name in ['heog', 'veog']:
>    raw = df[ch_name].values
>    processed = process_eog_signal(raw, sample_rate=SAMPLE_RATE)

What was generated:

>import numpy as np
>
>def extract_features(df, sample_rate=SAMPLE_RATE):
>    features = {}
>
>    #Initialize in case a channel is missing
>    heog = None
>    veog = None
>
>    for ch_name in ['heog', 'veog']:
>        if ch_name not in df.columns:
>            continue
>
>    raw = df[ch_name].values
>    processed = process_eog_signal(raw, sample_rate=sample_rate)
>
>    prefix = ch_name.upper()
>
>    #Store processed signals
>        if ch_name == 'heog':
>            heog = processed
>        elif ch_name == 'veog':
>            veog = processed
>
>    #Feature extraction
>        features[f"{prefix}_min"] = np.min(processed)
>        features[f"{prefix}_max"] = np.max(processed)
>        features[f"{prefix}_ptp"] = np.ptp(processed)
>
>    return features, heog, veog

Modifications made: I added this part to my original function:

    if ch_name == 'heog':
        heog = processed
    elif ch_name == 'veog':
        veog = processed

What I learned:
>I learned that sometimes things need to be restated despite things already having been delcared. Code is strange sometimes.

Reflection section:
How did AI tools help your productivity?
>Yes they did

What were they not helpful for? 
>For bug fixing since debugging complex code is really annoying sometimes

How did you verify AI-generated code was correct?
>I ran it and saw if the result I wanted was generated.
