import pandas as pd
import os

llama_2_7b_path = 'output/csv/meta-llama/llama-2-7b-hf/vlp_softmax_layers_llama_2_7b/'
llama_2_13b_path = 'output/csv/meta-llama/llama-2-13b-hf/vlp_softmax_layers_llama_2_13b/'

llama_2_7b_layers = 32
llama_2_13b_layers = 40

llama_2_7b_min_rows = []
llama_2_13b_min_rows = []

for i in range(0, llama_2_7b_layers):
    layer_str = str(i+1) + '/'
    llama_2_7b_layer_path = llama_2_7b_path + layer_str + 'metric.csv'
    
    if os.path.exists(llama_2_7b_layer_path):
        df = pd.read_csv(llama_2_7b_layer_path)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df_clean = df.dropna(subset=['value'])
        
        if not df_clean.empty:
            min_row = df_clean.loc[df_clean['value'].idxmin()]
            llama_2_7b_min_rows.append(min_row)
            print(f"Layer {i+1} - Min value: {min_row['value']}")
        else:
            print(f"Layer {i+1} - No valid values")
    else:
        print(f"Layer {i+1} - File not found: {llama_2_7b_layer_path}")

print("\n" + "="*50 + "\n")

for i in range(0, llama_2_13b_layers):
    layer_str = str(i+1) + '/'
    llama_2_13b_layer_path = llama_2_13b_path + layer_str + 'metric.csv'
    
    if os.path.exists(llama_2_13b_layer_path):
        df = pd.read_csv(llama_2_13b_layer_path)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df_clean = df.dropna(subset=['value'])
        
        if not df_clean.empty:
            min_row = df_clean.loc[df_clean['value'].idxmin()]
            llama_2_13b_min_rows.append(min_row)
            print(f"Layer {i+1} - Min value: {min_row['value']}")
        else:
            print(f"Layer {i+1} - No valid values")
    else:
        print(f"Layer {i+1} - File not found: {llama_2_13b_layer_path}")

# Convert to DataFrames for easier analysis
import matplotlib.pyplot as plt

if llama_2_7b_min_rows:
    llama_2_7b_results = pd.DataFrame(llama_2_7b_min_rows)
    print("\n\nLlama 2 7B Summary:")
    print(llama_2_7b_results)

if llama_2_13b_min_rows:
    llama_2_13b_results = pd.DataFrame(llama_2_13b_min_rows)
    print("\n\nLlama 2 13B Summary:")
    print(llama_2_13b_results)

width = 240 / 72
height = width * 0.4

# Create single plot
plt.figure(figsize=(width, height))

# Plot both models on the same axis
min_val_7b = None
min_val_13b = None

if llama_2_7b_min_rows:
    layers_7b = list(range(1, len(llama_2_7b_min_rows) + 1))
    values_7b = [row['value'] for row in llama_2_7b_min_rows]
    min_val_7b = min(values_7b)
    
    plt.plot(layers_7b, values_7b, marker='o', linewidth=0.8, markersize=2, label=f'Llama 2 7B    Final PPL: {min_val_7b:.2f}')

if llama_2_13b_min_rows:
    layers_13b = list(range(1, len(llama_2_13b_min_rows) + 1))
    values_13b = [row['value'] for row in llama_2_13b_min_rows]
    min_val_13b = min(values_13b)
    
    plt.plot(layers_13b, values_13b, marker='o', linewidth=0.8, markersize=2, label=f'Llama 2 13B    Final PPL: {min_val_13b:.2f}')

plt.ylabel('PPL', fontsize=8, labelpad=4)
plt.grid(True, alpha=0.3)

# Control legend position (x, y) in axis coordinates
legend_x = 1.031 # 0 = left, 1 = right
legend_y = 0.6 # 0 = bottom, 1 = top
plt.legend(fontsize=7, loc='upper right', bbox_to_anchor=(legend_x, legend_y), frameon=True, fancybox=True)

plt.tick_params(axis='both', labelsize=7)

# Set x-ticks every 5
ax = plt.gca()
max_layer = max(len(llama_2_7b_min_rows) if llama_2_7b_min_rows else 0, 
                len(llama_2_13b_min_rows) if llama_2_13b_min_rows else 0)
ax.set_xticks(range(0, max_layer + 1, 5))

# Set y-ticks every 1 from 5.4 to 6.2
ax.set_yticks([5.4, 5.6, 5.8, 6.0, 6.2])
ax.set_ylim(5.4, 6.2)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add minimum values as text


plt.savefig('figures/layer_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/layer_distribution.pdf', dpi=300, bbox_inches='tight')
print("\n\nPlot saved to figures/layer_distribution.png and figures/layer_distribution.pdf")
plt.show()