import pickle
import matplotlib.pyplot as plt
import shap

# Load the SHAP analysis file
with open('data/mfc/guncontrol/frame_prediction/shap_analysis_dev_0.pickle', 'rb') as f:
    shap_data = pickle.load(f)

shap_values = shap_data['shap_values']

print("SHAP values type:", type(shap_values))
print("SHAP values shape:", shap_values.shape)
print("SHAP values[0] type:", type(shap_values[0]))
print("SHAP values[0] shape:", shap_values[0].shape)

# Test different plotting approaches
print("\n--- Testing SHAP text plot ---")

# Try the original approach
try:
    fig = plt.figure(figsize=(12, 8))
    shap.plots.text(shap_values[0], display=False)
    plt.savefig('test_plot_1.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Original approach worked")
except Exception as e:
    print("✗ Original approach failed:", e)

# Try without display=False
try:
    fig = plt.figure(figsize=(12, 8))
    shap.plots.text(shap_values[0])
    plt.savefig('test_plot_2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ Without display=False worked")
except Exception as e:
    print("✗ Without display=False failed:", e)

# Try accessing the plot differently
try:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    shap.plots.text(shap_values[0], display=False)
    plt.savefig('test_plot_3.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ With explicit axis worked")
except Exception as e:
    print("✗ With explicit axis failed:", e)

# Check if we can access the plot content
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.text(shap_values[0], ax=ax)
    plt.savefig('test_plot_4.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ With explicit ax parameter worked")
except Exception as e:
    print("✗ With explicit ax parameter failed:", e)

print("\nGenerated test plots. Check their contents.")