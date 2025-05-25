coeff_df = pd.DataFrame({
    'Feature': features.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=coeff_df, x='Coefficient', y='Feature', palette='viridis')
plt.title('Impact of Features on Trade Balance')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()
