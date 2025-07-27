import pickle
import numpy as np

# Load the SHAP analysis file
with open('data/mfc/guncontrol/frame_prediction/shap_analysis_dev_0.pickle', 'rb') as f:
    shap_data = pickle.load(f)

print("Keys in SHAP data:", list(shap_data.keys()))
print()

# Inspect each key
for key, value in shap_data.items():
    print(f"{key}: {type(value)}")
    if key == 'shap_values':
        print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'No shape attribute'}")
        print(f"  Type: {type(value)}")
        if hasattr(value, '__len__'):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if hasattr(value[0], 'shape'):
                    print(f"  First element shape: {value[0].shape}")
                if hasattr(value[0], 'values'):
                    print(f"  Values shape: {value[0].values.shape}")
                    print(f"  Values sample: {value[0].values[:5] if len(value[0].values) > 5 else value[0].values}")
                if hasattr(value[0], 'data'):
                    print(f"  Data: {value[0].data}")
    elif isinstance(value, str):
        print(f"  Value: {value[:100]}{'...' if len(value) > 100 else ''}")
    else:
        print(f"  Value: {value}")
    print()