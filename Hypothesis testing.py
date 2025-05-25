from scipy.stats import ttest_1samp

# Select post-2018 steel export data (2019 onward)
post_steel_exports = full_df.loc[full_df['Year'] > 2018, 'Steel_Exports']

# Test whether post-2018 exports differ significantly from 2018 level
t_stat, p_value = ttest_1samp(post_steel_exports, popmean=pre_tariff_steel)

print("\n=== Hypothesis Testing ===")
print("H₀: No change in steel exports post-2018")
print("H₁: Significant change in steel exports post-2018")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Result: Reject H₀ – Steel exports changed significantly after 2018.")
else:
    print("Result: Fail to reject H₀ – No statistically significant change.")