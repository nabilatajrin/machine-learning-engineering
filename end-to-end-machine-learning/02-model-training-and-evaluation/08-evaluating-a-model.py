# Evaluate model using k-fold cross-validation
kf = KFold(n_splits=5)

# Compute the cross-validation score
score = cross_val_score(model, heart_disease_df_X, heart_disease_df_y, scoring='balanced_accuracy', cv=kf)
print(score)

# Get model predictions
y_pred = model.predict(heart_disease_df_X)

# Print confusion matrix
cm = confusion_matrix(heart_disease_df_y, y_pred)
print(cm)
