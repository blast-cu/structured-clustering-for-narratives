import pandas as pd
import pingouin as pg


def compute_icc(min_possible_score, max_possible_score):
    """
    Compute Intraclass Correlation Coefficient (ICC) for inter-annotator agreement.
    
    Transforms TSV data into the format required by pingouin and computes various ICC types.
    
    Args:
        min_possible_score (int): Minimum score in the Likert scale
        max_possible_score (int): Maximum score in the Likert scale
    """
    # Read the TSV file
    input_file = "./data/mfc/immigration/annotations.tsv"
    data = pd.read_csv(input_file, sep='\t', header=None, names=['annotator1', 'annotator2'])
    
    print(f"Loaded {len(data)} annotations from {input_file}")
    print(f"Annotator 1 range: {data['annotator1'].min()} - {data['annotator1'].max()}")
    print(f"Annotator 2 range: {data['annotator2'].min()} - {data['annotator2'].max()}")
    print(f"Using full scale range: {min_possible_score} - {max_possible_score}")
    
    # Transform data into long format required by pingouin
    # Each row represents one measurement (subject, rater, score)
    long_data = []
    
    for idx, row in data.iterrows():
        # Add measurement from annotator 1
        long_data.append({
            'Subject': idx,
            'Rater': 'Annotator1',
            'Score': row['annotator1']
        })
        # Add measurement from annotator 2
        long_data.append({
            'Subject': idx,
            'Rater': 'Annotator2', 
            'Score': row['annotator2']
        })
    
    # To account for the full Likert scale range, add phantom subjects with extreme scores
    # This ensures ICC calculation considers the full possible range
    n_subjects = len(data)
    
    # Add phantom subjects with minimum scores
    long_data.append({
        'Subject': n_subjects,
        'Rater': 'Annotator1',
        'Score': min_possible_score
    })
    long_data.append({
        'Subject': n_subjects,
        'Rater': 'Annotator2',
        'Score': min_possible_score
    })
    
    # Add phantom subjects with maximum scores
    long_data.append({
        'Subject': n_subjects + 1,
        'Rater': 'Annotator1',
        'Score': max_possible_score
    })
    long_data.append({
        'Subject': n_subjects + 1,
        'Rater': 'Annotator2',
        'Score': max_possible_score
    })
    
    # Convert to DataFrame
    df_long = pd.DataFrame(long_data)
    
    print(f"Transformed to long format: {len(df_long)} measurements for {df_long['Subject'].nunique()} subjects")
    
    # Compute ICC using pingouin
    # ICC(2,1) - Two-way random effects, single measurement, absolute agreement
    # ICC(2,k) - Two-way random effects, average measurement, absolute agreement  
    # ICC(3,1) - Two-way mixed effects, single measurement, consistency
    # ICC(3,k) - Two-way mixed effects, average measurement, consistency
    
    icc_results = pg.intraclass_corr(data=df_long, targets='Subject', raters='Rater', ratings='Score')
    
    # Extract specific ICC values
    icc_2_1 = icc_results[icc_results['Type'] == 'ICC2']['ICC'].iloc[0]  # ICC(2,1)
    icc_2_k = icc_results[icc_results['Type'] == 'ICC2k']['ICC'].iloc[0]  # ICC(2,k)
    icc_3_1 = icc_results[icc_results['Type'] == 'ICC3']['ICC'].iloc[0]  # ICC(3,1)
    icc_3_k = icc_results[icc_results['Type'] == 'ICC3k']['ICC'].iloc[0]  # ICC(3,k)
    
    # Pretty print results (multiplied by 100, 2 decimal places)
    print(f"\nIntraclass Correlation Coefficient (ICC) Results:")
    print(f"ICC(2,1) - Two-way random, single: {icc_2_1 * 100:.2f}")
    print(f"ICC(2,k) - Two-way random, average: {icc_2_k * 100:.2f}")
    print(f"ICC(3,1) - Two-way mixed, single: {icc_3_1 * 100:.2f}")
    print(f"ICC(3,k) - Two-way mixed, average: {icc_3_k * 100:.2f}")
    
    # Print results in tab-separated format (multiplied by 100, 2 decimal places)
    print(f"\nTab-separated results:")
    print(f"{icc_2_1 * 100:.2f}\t{icc_2_k * 100:.2f}\t{icc_3_1 * 100:.2f}\t{icc_3_k * 100:.2f}")
    
    # Print full results table for reference
    print(f"\nFull ICC Results Table:")
    print(icc_results.round(4))
    
    return icc_2_1, icc_2_k, icc_3_1, icc_3_k


def main():
    # Configure the score range here
    min_possible_score = 0  # Change this to your scale's minimum
    max_possible_score = 4  # Change this to your scale's maximum
    
    compute_icc(min_possible_score, max_possible_score)


if __name__ == "__main__":
    main()